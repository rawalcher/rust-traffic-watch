use anyhow::{Context, Result};
use image::codecs::jpeg::JpegEncoder;
use image::ImageEncoder;
use inference::engine::OnnxDetector;
use protocol::types::{
    tiers::{codec_name, res_folder},
    EncodingSpec,
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

use crate::annotate::draw_detections;
use crate::state::{LatestFrame, SharedState};

pub struct LoopConfig {
    pub model_path: PathBuf,
    pub model_name: String,
    pub spec: EncodingSpec,
    pub fps_cap: Option<u64>,
}

pub fn run_inference_loop(cfg: &LoopConfig, state: &Arc<SharedState>) -> Result<()> {
    let frame_dir = PathBuf::from("../../services/roadside-unit/testImages")
        .join(res_folder(cfg.spec.resolution))
        .join(codec_name(cfg.spec.codec));

    info!("Loading frames from {:?}", frame_dir);
    let mut frames: Vec<PathBuf> = std::fs::read_dir(&frame_dir)
        .with_context(|| format!("Failed to read {:?}", frame_dir.display()))?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .filter(|p| {
            // Match only the selected tier
            p.file_stem()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.ends_with(&format!("_{:?}", cfg.spec.tier)))
        })
        .collect();
    frames.sort();

    if frames.is_empty() {
        anyhow::bail!("No frames found in {:?} for tier {:?}", frame_dir.display(), cfg.spec.tier);
    }
    info!("Found {} frames", frames.len());

    info!("Loading model from {:?}", cfg.model_path);
    let mut detector = OnnxDetector::new(&cfg.model_path)?;

    let min_frame_duration = cfg.fps_cap.map(|fps| Duration::from_micros(1_000_000 / fps));
    let mut idx: usize = 0;

    loop {
        let loop_start = Instant::now();
        let path = &frames[idx];

        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                warn!("Failed reading {:?}: {e}", path);
                idx = (idx + 1) % frames.len();
                continue;
            }
        };

        let inference_start = Instant::now();
        let detection_result = match detector.detect(&bytes) {
            Ok(r) => r,
            Err(e) => {
                warn!("Inference failed on {:?}: {e}", path);
                idx = (idx + 1) % frames.len();
                continue;
            }
        };
        let inference_us = u64::try_from(inference_start.elapsed().as_micros()).unwrap_or(0);

        let jpeg_bytes = match annotate_to_jpeg(&bytes, &detection_result.detections) {
            Ok(b) => b,
            Err(e) => {
                warn!("Annotation failed: {e}");
                idx = (idx + 1) % frames.len();
                continue;
            }
        };

        let mut class_counts: HashMap<String, u32> = HashMap::new();
        for d in &detection_result.detections {
            *class_counts.entry(d.class.clone()).or_insert(0) += 1;
        }

        let summary = class_counts
            .iter()
            .map(|(class, n)| format!("{n} {class}"))
            .collect::<Vec<_>>()
            .join(", ");

        info!(
            "frame {:04} | {:>3}ms | {}",
            idx,
            inference_us / 1000,
            if summary.is_empty() { "no detections".to_string() } else { summary }
        );

        {
            let mut latest = state.latest.write();
            *latest = LatestFrame {
                jpeg_bytes,
                frame_number: idx as u64,
                inference_us,
                detection_count: detection_result.detection_count,
                class_counts,
                model_name: cfg.model_name.clone(),
            };
        }
        {
            let mut n = state.total_inferences.write();
            *n += 1;
        }

        debug!(idx, inference_ms = inference_us / 1000, "frame done");

        idx = (idx + 1) % frames.len();

        if let Some(min_dur) = min_frame_duration {
            let elapsed = loop_start.elapsed();
            if elapsed < min_dur {
                std::thread::sleep(min_dur.checked_sub(elapsed).unwrap());
            }
        }
    }
}

fn annotate_to_jpeg(
    input_bytes: &[u8],
    detections: &[protocol::types::Detection],
) -> Result<Vec<u8>> {
    let mut img = image::load_from_memory(input_bytes)?;
    draw_detections(&mut img, detections)?;

    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let mut out = Vec::with_capacity(128 * 1024);
    let encoder = JpegEncoder::new_with_quality(&mut out, 80);
    encoder.write_image(&rgb, w, h, image::ColorType::Rgb8)?;
    Ok(out)
}
