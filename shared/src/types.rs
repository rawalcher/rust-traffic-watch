use serde::{Deserialize, Serialize};
use crate::constants::{DEFAULT_DURATION_SECONDS, DEFAULT_FPS};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TimingPayload {
    pub sequence_id: u64,
    pub frame_data: Option<Vec<u8>>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub pi_hostname: String,
    pub pi_capture_start: Option<u64>,
    pub pi_sent_to_jetson: Option<u64>,
    pub jetson_received: Option<u64>,
    pub jetson_inference_start: Option<u64>,
    pub jetson_inference_complete: Option<u64>,
    pub jetson_sent_result: Option<u64>,
    pub controller_received: Option<u64>,
}

#[derive(Debug, Clone)]
pub enum ThroughputMode {
    High,
    Fps,
}

pub struct FrameThroughputController {
    mode: ThroughputMode,
}

impl FrameThroughputController {
    pub fn new(mode: ThroughputMode) -> Self {
        Self { mode }
    }

    pub fn set_mode(&mut self, mode: ThroughputMode) {
        self.mode = mode;
    }

    pub fn get_frame_skip(&self) -> u64 {
        match self.mode {
            ThroughputMode::High => 3,
            ThroughputMode::Fps => 30,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ExperimentMode {
    LocalOnly,
    Offload,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ExperimentConfig {
    pub experiment_id: String,
    pub model_name: String,
    pub mode: ExperimentMode,
    pub duration_seconds: u64,
    pub fixed_fps: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ControlMessage {
    StartExperiment { config: ExperimentConfig },
    ProcessingResult(ProcessingResult),
    PreheatingComplete,
    ReadyToStart,
    BeginExperiment,
    Shutdown,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum NetworkMessage {
    Control(ControlMessage),
    Frame(TimingPayload),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ProcessingResult {
    pub timing: TimingPayload,
    pub inference: InferenceResult,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct InferenceResult {
    pub sequence_id: u64,
    pub detections: Vec<Detection>,
    pub processing_time_us: u64,
    pub pi_hostname: String,
    pub frame_size_bytes: u32,
    pub detection_count: u32,
    pub image_width: u32,
    pub image_height: u32,
    pub model_name: String,
    pub experiment_mode: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Detection {
    pub class: String,
    pub bbox: [f32; 4], // x, y, width, height
    pub confidence: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
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

impl TimingPayload {
    pub fn new(sequence_id: u64) -> Self {
        let hostname = get_hostname();
        Self {
            sequence_id,
            frame_data: None,
            width: None,
            height: None,
            pi_hostname: hostname,
            pi_capture_start: Some(current_timestamp_micros()),
            pi_sent_to_jetson: None,
            jetson_received: None,
            jetson_inference_start: None,
            jetson_inference_complete: None,
            jetson_sent_result: None,
            controller_received: None,
        }
    }

    pub fn add_frame_data(&mut self, data: Vec<u8>, width: u32, height: u32) {
        self.frame_data = Some(data);
        self.width = Some(width);
        self.height = Some(height);
    }

    pub fn pi_overhead_us(&self) -> Option<u64> {
        match (self.pi_capture_start, self.pi_sent_to_jetson) {
            (Some(start), Some(sent)) => Some(sent - start),
            _ => None,
        }
    }

    pub fn network_latency_us(&self) -> Option<u64> {
        match (self.pi_sent_to_jetson, self.jetson_received) {
            (Some(sent), Some(received)) => Some(received - sent),
            _ => None,
        }
    }

    pub fn jetson_processing_us(&self) -> Option<u64> {
        match (self.jetson_inference_start, self.jetson_inference_complete) {
            (Some(start), Some(complete)) => Some(complete - start),
            _ => None,
        }
    }

    pub fn total_latency_us(&self) -> Option<u64> {
        match (self.pi_capture_start, self.controller_received) {
            (Some(start), Some(received)) => Some(received - start),
            _ => None,
        }
    }
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