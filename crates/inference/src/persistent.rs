use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tracing::info;

use crate::engine::OnnxDetector;
use protocol::{current_timestamp_micros, FrameMessage, InferenceResult};

pub struct PersistentOnnxDetector {
    detector: OnnxDetector,
    model_name: String,
    models_dir: PathBuf,
    last_activity: Instant,
    start_time: Instant,
    inference_count: usize,
}

impl PersistentOnnxDetector {
    /// # Errors
    /// # Panics
    pub fn new(model_name: String, models_dir: PathBuf) -> Result<Self> {
        info!("Loading ONNX model '{}'", model_name);
        let model_path = Self::resolve_model_path(&model_name, &models_dir)?;
        let detector = OnnxDetector::new(model_path.to_str().unwrap())?;

        Ok(Self {
            detector,
            model_name,
            models_dir,
            last_activity: Instant::now(),
            start_time: Instant::now(),
            inference_count: 0,
        })
    }

    /// # Errors
    /// # Panics
    pub fn reload_model(&mut self, new_model: &str) -> Result<()> {
        info!("Reloading ONNX model to '{}'", new_model);
        let model_path = Self::resolve_model_path(new_model, &self.models_dir)?;
        let new_detector = OnnxDetector::new(model_path.to_str().unwrap())?;

        self.detector = new_detector;
        self.model_name = new_model.to_string();

        self.inference_count = 0;
        self.last_activity = Instant::now();
        self.start_time = Instant::now();

        Ok(())
    }

    fn resolve_model_path(model_name: &str, models_dir: &Path) -> Result<PathBuf> {
        let exact_path = models_dir.join(format!("{model_name}.onnx"));
        if exact_path.exists() {
            return Ok(exact_path);
        }

        let with_ext = if Path::new(model_name)
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("onnx"))
        {
            models_dir.join(model_name)
        } else {
            models_dir.join(format!("{model_name}.onnx"))
        };

        if with_ext.exists() {
            return Ok(with_ext);
        }

        anyhow::bail!(
            "Model file not found: {:?} or {:?}",
            exact_path.display(),
            with_ext.display()
        );
    }

    /// # Errors
    /// # Panics
    pub fn detect_objects(&mut self, image_bytes: &[u8], _mode: &str) -> Result<InferenceResult> {
        self.last_activity = Instant::now();
        self.inference_count += 1;

        // Direct access to detector, no mutex overhead
        self.detector.detect(image_bytes)
    }

    #[must_use]
    pub fn stats(&self) -> DetectorStats {
        let uptime = self.start_time.elapsed();
        let idle_time = self.last_activity.elapsed();

        DetectorStats {
            model_name: self.model_name.clone(),
            inference_count: self.inference_count,
            uptime_secs: uptime.as_secs(),
            idle_secs: idle_time.as_secs(),
        }
    }

    pub fn shutdown(&self) {
        let stats = self.stats();
        info!(
            "Shutting down detector '{}': {} inferences in {}s",
            stats.model_name, stats.inference_count, stats.uptime_secs
        );
    }
}

#[derive(Debug, Clone)]
pub struct DetectorStats {
    pub model_name: String,
    pub inference_count: usize,
    pub uptime_secs: u64,
    pub idle_secs: u64,
}

/// # Errors
pub fn perform_onnx_inference_with_counts(
    frame_message: &mut FrameMessage,
    detector: &mut PersistentOnnxDetector,
) -> Result<InferenceResult> {
    let image_bytes = &frame_message.frame.frame_data;
    frame_message.timing.inference_start = Some(current_timestamp_micros());
    let mut result =
        detector.detect_objects(image_bytes, "default").context("ONNX inference failed")?;
    frame_message.timing.inference_complete = Some(current_timestamp_micros());
    result.sequence_id = frame_message.sequence_id;
    Ok(result)
}
