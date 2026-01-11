use anyhow::Result;
use image::GenericImageView;
use ort::inputs;
use ort::session::Session;
use ort::value::TensorRef;
use protocol::InferenceResult;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use super::utils::{letterbox_nchw, parse_yolo_output, rescale_detections};

const IMAGE_SIZE: usize = 640;
const CONFIDENCE_THRESHOLD: f32 = 0.25;
const IOU_THRESHOLD: f32 = 0.45;
const BGR_MODE: bool = false;

pub struct OnnxDetector {
    session: Session,
    model_name: String,
    class_names: HashMap<i64, &'static str>,
    allowed_classes: Vec<i64>,
}

impl OnnxDetector {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let path = model_path.as_ref();
        let model_name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown").to_string();

        let session = Session::builder()?
            .with_intra_threads(4)?
            .with_inter_threads(2)?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level1)?
            .commit_from_file(path)?;

        let class_names = HashMap::from([
            (0, "person"),
            (1, "bicycle"),
            (2, "car"),
            (3, "motorcycle"),
            (5, "bus"),
            (6, "train"),
            (7, "truck"),
        ]);

        let allowed_classes = vec![0, 1, 2, 3, 5, 7];

        Ok(Self { session, model_name, class_names, allowed_classes })
    }

    pub fn detect(&mut self, image_bytes: &[u8]) -> Result<InferenceResult> {
        let start = Instant::now();

        let image = image::load_from_memory(image_bytes)?;
        let (original_width, original_height) = image.dimensions();

        // Preprocess: resize with padding and normalize
        let (input_tensor, scale, pad_x, pad_y) =
            letterbox_nchw(&image, IMAGE_SIZE as u32, BGR_MODE)?;

        // Run inference
        let input_ref = TensorRef::from_array_view(input_tensor.view())?;
        let outputs = self.session.run(inputs!["images" => input_ref])?;

        // Extract output tensor
        let output = outputs["output0"].try_extract_array::<f32>()?;
        let output_3d = output.into_dimensionality::<ndarray::Ix3>()?;

        // Parse YOLO output format
        let (boxes, scores, class_ids) = parse_yolo_output(
            output_3d,
            CONFIDENCE_THRESHOLD,
            IOU_THRESHOLD,
            &self.allowed_classes,
        );

        // Convert back to original image coordinates
        let detections = rescale_detections(
            &boxes,
            &scores,
            &class_ids,
            &self.class_names,
            scale,
            pad_x,
            pad_y,
            original_width,
            original_height,
        );

        Ok(InferenceResult {
            sequence_id: 0,
            processing_time_us: start.elapsed().as_micros() as u64,
            frame_size_bytes: image_bytes.len() as u32,
            detection_count: detections.len() as u32,
            detections,
            image_width: original_width,
            image_height: original_height,
            model_name: self.model_name.clone(),
            experiment_mode: "default".into(),
        })
    }
}
