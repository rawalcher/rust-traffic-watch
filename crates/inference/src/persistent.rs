use anyhow::{Context, Result};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tracing::info;

use crate::engine::OnnxDetector;
use protocol::{FrameMessage, InferenceResult};

pub struct PersistentOnnxDetector {
    detector: Arc<Mutex<OnnxDetector>>,
    model_name: String,
    models_dir: PathBuf,
    last_activity: Arc<Mutex<Instant>>,
    start_time: Instant,
    inference_count: Arc<Mutex<usize>>,
}

impl PersistentOnnxDetector {
    pub fn new(model_name: String, models_dir: PathBuf) -> Result<Self> {
        info!("Loading ONNX model '{}'", model_name);
        let model_path = Self::resolve_model_path(&model_name, &models_dir)?;
        let detector = OnnxDetector::new(model_path.to_str().unwrap())?;

        Ok(Self {
            detector: Arc::new(Mutex::new(detector)),
            model_name,
            models_dir,
            last_activity: Arc::new(Mutex::new(Instant::now())),
            start_time: Instant::now(),
            inference_count: Arc::new(Mutex::new(0)),
        })
    }

    pub fn reload_model(&mut self, new_model: &str) -> Result<()> {
        info!("Reloading ONNX model to '{}'", new_model);
        let model_path = Self::resolve_model_path(new_model, &self.models_dir)?;
        let new_detector = OnnxDetector::new(model_path.to_str().unwrap())?;

        *self.detector.lock().unwrap() = new_detector;
        self.model_name = new_model.to_string();

        *self.inference_count.lock().unwrap() = 0;
        *self.last_activity.lock().unwrap() = Instant::now();
        self.start_time = Instant::now();

        Ok(())
    }

    fn resolve_model_path(model_name: &str, models_dir: &PathBuf) -> Result<PathBuf> {
        let exact_path = models_dir.join(format!("{}.onnx", model_name));
        if exact_path.exists() {
            return Ok(exact_path);
        }

        let with_ext = if model_name.ends_with(".onnx") {
            models_dir.join(model_name)
        } else {
            models_dir.join(format!("{}.onnx", model_name))
        };

        if with_ext.exists() {
            return Ok(with_ext);
        }

        anyhow::bail!("Model file not found: {:?} or {:?}", exact_path, with_ext)
    }

    pub fn detect_objects(&self, image_bytes: &[u8], _mode: &str) -> Result<InferenceResult> {
        {
            let mut last = self.last_activity.lock().unwrap();
            *last = Instant::now();
        }
        {
            let mut count = self.inference_count.lock().unwrap();
            *count += 1;
        }

        let mut detector = self.detector.lock().unwrap();
        detector.detect(image_bytes)
    }

    pub fn stats(&self) -> DetectorStats {
        let count = *self.inference_count.lock().unwrap();
        let uptime = self.start_time.elapsed();
        let last_activity = *self.last_activity.lock().unwrap();
        let idle_time = last_activity.elapsed();

        DetectorStats {
            model_name: self.model_name.clone(),
            inference_count: count,
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

pub fn perform_onnx_inference_with_counts(
    frame_message: &FrameMessage,
    detector: &PersistentOnnxDetector,
) -> Result<InferenceResult> {
    let image_bytes = &frame_message.frame.frame_data;
    let mut result = detector
        .detect_objects(image_bytes, "default")
        .context("ONNX inference failed")?;
    result.sequence_id = frame_message.sequence_id;
    Ok(result)
}
