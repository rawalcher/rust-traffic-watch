use crate::constants::{DEFAULT_DURATION_SECONDS, DEFAULT_FPS};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Copy, Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceId {
    Pi,
    Jetson,
}

impl fmt::Display for DeviceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceId::Pi => write!(f, "Pi"),
            DeviceId::Jetson => write!(f, "Jetson"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Message {
    Hello(DeviceId),
    Frame(FrameMessage),
    Result(InferenceMessage),
    Control(ControlMessage),
    Pulse(TimingMetadata),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlMessage {
    StartExperiment { config: ExperimentConfig },
    Shutdown,
    ReadyToStart,
    BeginExperiment,
    PreheatingComplete,
    DataConnectionReady,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TimingMetadata {
    pub sequence_id: u64,
    pub frame_number: u64,
    pub pi_hostname: String,
    pub pi_capture_start: Option<u64>,
    pub pi_sent_to_jetson: Option<u64>,
    pub jetson_received: Option<u64>,
    pub jetson_inference_start: Option<u64>,
    pub jetson_inference_complete: Option<u64>,
    pub jetson_sent_result: Option<u64>,
    pub controller_sent_pulse: Option<u64>,
    pub controller_received: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameMessage {
    pub sequence_id: u64,
    pub frame_data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub timing: TimingMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceMessage {
    pub sequence_id: u64,
    pub inference: InferenceResult,
    pub timing: TimingMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    pub sequence_id: u64,
    pub detections: Vec<Detection>,
    pub processing_time_us: u64,
    pub frame_size_bytes: u32,
    pub detection_count: u32,
    pub image_width: u32,
    pub image_height: u32,
    pub model_name: String,
    pub experiment_mode: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection {
    pub class: String,
    pub bbox: [f32; 4],
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    pub experiment_id: String,
    pub model_name: String,
    pub mode: ExperimentMode,
    pub duration_seconds: u64,
    pub fixed_fps: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExperimentMode {
    LocalOnly,
    Offload,
}

impl ExperimentConfig {
    pub fn new(experiment_id: String, mode: ExperimentMode, model_name: String) -> Self {
        Self {
            experiment_id,
            model_name,
            mode,
            duration_seconds: DEFAULT_DURATION_SECONDS,
            fixed_fps: DEFAULT_FPS,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectCounts {
    pub cars: u32,
    pub trucks: u32,
    pub buses: u32,
    pub motorcycles: u32,
    pub bicycles: u32,
    pub pedestrians: u32,
    pub total_vehicles: u32,
    pub total_objects: u32,
}

pub fn current_timestamp_micros() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as u64
}

pub fn get_hostname() -> String {
    std::env::var("HOSTNAME")
        .or_else(|_| {
            std::process::Command::new("hostname")
                .output()
                .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
                .map_err(|_| "unknown")
        })
        .unwrap_or_else(|_| "unknown-pi".to_string())
}
