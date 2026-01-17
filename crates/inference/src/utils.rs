use image::{imageops, DynamicImage, GenericImageView, Rgb};
use ndarray::{s, Array, ArrayView3, Axis};
use protocol::types::Detection;
use std::cmp::Ordering;
use std::collections::HashMap;

pub type Tensor4D = Array<f32, ndarray::Dim<[usize; 4]>>;
pub type LetterboxResult = (Tensor4D, f32, f32, f32);

pub struct RescaleParams {
    pub scale: f32,
    pub pad_x: f32,
    pub pad_y: f32,
    pub orig_width: u32,
    pub orig_height: u32,
}

/// Resize image to square with padding, return normalized tensor and transform params
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn letterbox_nchw(img: &DynamicImage, target_size: u32, swap_bgr: bool) -> LetterboxResult {
    let (orig_width, orig_height) = img.dimensions();

    // Calculate scale to fit image in target_size square
    let scale =
        (target_size as f32 / orig_width as f32).min(target_size as f32 / orig_height as f32);
    let new_width = (orig_width as f32 * scale).round() as u32;
    let new_height = (orig_height as f32 * scale).round() as u32;

    // Calculate padding to center image
    let pad_x = ((target_size - new_width) / 2) as f32;
    let pad_y = ((target_size - new_height) / 2) as f32;

    // Resize and place on gray canvas
    let resized = img.resize_exact(new_width, new_height, imageops::FilterType::Triangle);
    let mut canvas = image::RgbImage::from_pixel(target_size, target_size, Rgb([114, 114, 114]));
    imageops::overlay(&mut canvas, &resized.to_rgb8(), pad_x as i64, pad_y as i64);

    // Convert to NCHW tensor [1, 3, H, W] with values normalized to [0, 1]
    let size = target_size as usize;
    let mut tensor = Array::zeros((1, 3, size, size));

    for (x, y, pixel) in canvas.enumerate_pixels() {
        let r = pixel.0[0];
        let g = pixel.0[1];
        let b = pixel.0[2];
        let (c0, c1, c2) = if swap_bgr { (b, g, r) } else { (r, g, b) };

        tensor[[0, 0, y as usize, x as usize]] = f32::from(c0) / 255.0;
        tensor[[0, 1, y as usize, x as usize]] = f32::from(c1) / 255.0;
        tensor[[0, 2, y as usize, x as usize]] = f32::from(c2) / 255.0;
    }

    (tensor, scale, pad_x, pad_y)
}

/// Parse YOLO model output into bounding boxes, scores, and class IDs
pub fn parse_yolo_output(
    output: ArrayView3<f32>,
    confidence_threshold: f32,
    iou_threshold: f32,
    allowed_class_ids: &[i64],
) -> (Vec<[f32; 4]>, Vec<f32>, Vec<i64>) {
    let predictions = output.index_axis(Axis(0), 0);
    let shape = predictions.shape();
    let preds =
        if shape[0] < shape[1] { predictions.t().to_owned() } else { predictions.to_owned() };

    let mut boxes = Vec::new();
    let mut scores = Vec::new();
    let mut class_ids = Vec::new();

    for prediction in preds.outer_iter() {
        if prediction.len() < 5 || prediction[4] < confidence_threshold {
            continue;
        }

        let objectness = prediction[4];
        let class_probs = prediction.slice(s![5..]);
        let (top_index, class_prob) = find_max(&class_probs);
        let confidence = objectness * class_prob;

        if confidence < confidence_threshold {
            continue;
        }

        #[allow(clippy::cast_possible_wrap)]
        let class_id = top_index as i64;

        if !allowed_class_ids.contains(&class_id) {
            continue;
        }

        // Convert from center format to corner format
        let center_x = prediction[0];
        let center_y = prediction[1];
        let width = prediction[2];
        let height = prediction[3];

        boxes.push([
            center_x - width / 2.0,
            center_y - height / 2.0,
            center_x + width / 2.0,
            center_y + height / 2.0,
        ]);
        scores.push(confidence);
        class_ids.push(class_id);
    }

    // Apply non-maximum suppression
    let keep_indices = non_maximum_suppression(&boxes, &scores, iou_threshold);

    let filtered_boxes: Vec<_> = keep_indices.iter().map(|&i| boxes[i]).collect();
    let filtered_scores: Vec<_> = keep_indices.iter().map(|&i| scores[i]).collect();
    let filtered_ids: Vec<_> = keep_indices.iter().map(|&i| class_ids[i]).collect();

    (filtered_boxes, filtered_scores, filtered_ids)
}

/// Convert model-space detections back to original image coordinates
#[allow(clippy::cast_precision_loss)]
pub fn rescale_detections(
    boxes: &[[f32; 4]],
    scores: &[f32],
    class_ids: &[i64],
    id_to_name: &HashMap<i64, &'static str>,
    params: &RescaleParams,
) -> Vec<Detection> {
    boxes
        .iter()
        .zip(scores.iter())
        .zip(class_ids.iter())
        .map(|((&[x1, y1, x2, y2], &score), &class_id)| {
            let class_name = id_to_name.get(&class_id).copied().unwrap_or("unknown");

            let x1 = ((x1 - params.pad_x) / params.scale).clamp(0.0, params.orig_width as f32);
            let y1 = ((y1 - params.pad_y) / params.scale).clamp(0.0, params.orig_height as f32);
            let x2 = ((x2 - params.pad_x) / params.scale).clamp(0.0, params.orig_width as f32);
            let y2 = ((y2 - params.pad_y) / params.scale).clamp(0.0, params.orig_height as f32);

            Detection {
                class: class_name.to_string(),
                bbox: [x1, y1, x2 - x1, y2 - y1],
                confidence: score,
            }
        })
        .collect()
}

/// Find index and value of maximum element
fn find_max(array: &ndarray::ArrayView1<f32>) -> (usize, f32) {
    array
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .map_or((0, f32::MIN), |(idx, &val)| (idx, val))
}

/// Non-maximum suppression to remove overlapping boxes
fn non_maximum_suppression(boxes: &[[f32; 4]], scores: &[f32], iou_threshold: f32) -> Vec<usize> {
    if boxes.is_empty() {
        return Vec::new();
    }

    // Sort by score descending
    let mut indices: Vec<usize> = (0..boxes.len()).collect();
    indices.sort_unstable_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(Ordering::Equal));

    let mut keep = Vec::new();
    let mut suppressed = vec![false; boxes.len()];

    for &current_idx in &indices {
        if suppressed[current_idx] {
            continue;
        }

        keep.push(current_idx);

        for &candidate_idx in &indices {
            if suppressed[candidate_idx] || candidate_idx == current_idx {
                continue;
            }

            if calculate_iou(boxes[current_idx], boxes[candidate_idx]) > iou_threshold {
                suppressed[candidate_idx] = true;
            }
        }
    }

    keep
}

/// Calculate Intersection over Union between two boxes
fn calculate_iou(box_a: [f32; 4], box_b: [f32; 4]) -> f32 {
    let [x1_a, y1_a, x2_a, y2_a] = box_a;
    let [x1_b, y1_b, x2_b, y2_b] = box_b;

    // Calculate intersection area
    let intersection_width = (x2_a.min(x2_b) - x1_a.max(x1_b)).max(0.0);
    let intersection_height = (y2_a.min(y2_b) - y1_a.max(y1_b)).max(0.0);
    let intersection_area = intersection_width * intersection_height;

    // Calculate union area
    let area_a = (x2_a - x1_a) * (y2_a - y1_a);
    let area_b = (x2_b - x1_b) * (y2_b - y1_b);
    let union_area = area_a + area_b - intersection_area;

    intersection_area / (union_area + 1e-6)
}
