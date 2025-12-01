use anyhow::Result;
use image::{DynamicImage, Rgba};
use imageproc::drawing::{draw_filled_rect_mut, draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use inference::engine::OnnxDetector;
use protocol::types::Detection;
use rusttype::{Font, Scale};
use std::collections::HashMap;
use std::path::PathBuf;
use tracing::{error, info};

static FONT_DATA: &[u8] = include_bytes!("../fonts/Roboto-Thin.ttf");

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        error!("Usage: {} <model.onnx> <image.jpg> [font_path.ttf]", args[0]);
        std::process::exit(1);
    }

    let model_path = PathBuf::from(&args[1]);
    let image_path = PathBuf::from(&args[2]);

    info!("Loading model: {:?}", model_path);
    let mut detector = OnnxDetector::new(&model_path)?;

    info!("Loading image: {:?}", image_path);
    let image_bytes = std::fs::read(&image_path)?;
    let mut img = image::load_from_memory(&image_bytes)?;

    info!("Running inference...");
    let result = detector.detect(&image_bytes)?;

    print_summary(&result.detections);

    if !result.detections.is_empty() {
        draw_detections(&mut img, &result.detections)?;
        img.save("output.jpg")?;
        info!("Saved annotated image to: output.jpg");
    }

    Ok(())
}

fn print_summary(detections: &[Detection]) {
    let mut class_counts: HashMap<&str, usize> = HashMap::new();
    for det in detections {
        *class_counts.entry(&det.class).or_insert(0) += 1;
    }

    info!("\n{:-^40}", " Summary ");
    info!("Total Objects: {}", detections.len());

    let mut classes: Vec<_> = class_counts.keys().collect();
    classes.sort();
    for class in classes {
        info!("{:<15}: {}", class, class_counts[class]);
    }
    info!("{:-^40}\n", "");
}

fn draw_detections(img: &mut DynamicImage, detections: &[Detection]) -> Result<()> {
    let font =
        Font::try_from_bytes(FONT_DATA).ok_or_else(|| anyhow::anyhow!("Failed to load font"))?;

    let scale = Scale::uniform(24.0);

    for det in detections {
        let (x, y, w, h) = (
            det.bbox[0] as i32,
            det.bbox[1] as i32,
            det.bbox[2].max(1.0) as u32,
            det.bbox[3].max(1.0) as u32,
        );

        let seed = det.class.bytes().map(|b| b as u32).sum::<u32>();
        let color = Rgba([((seed * 50) % 255) as u8, ((seed * 100) % 255) as u8, 200, 255]);

        draw_hollow_rect_mut(img, Rect::at(x, y).of_size(w, h), color);

        let label = format!("{} {:.0}%", det.class, det.confidence * 100.0);
        let (text_w, text_h) = imageproc::drawing::text_size(scale, &font, &label);

        let label_y = (y - text_h).max(0);
        let bg_rect =
            Rect::at(x, label_y).of_size((text_w + 8).max(1) as u32, text_h.max(1) as u32);

        draw_filled_rect_mut(img, bg_rect, color);
        draw_text_mut(img, Rgba([255, 255, 255, 255]), x + 4, label_y, scale, &font, &label);
    }
    Ok(())
}
