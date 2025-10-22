use crate::{DeviceId, ExperimentConfig};
use codec::types::Frame;
use serde::{Deserialize, Serialize};
use std::fmt;

impl fmt::Display for DeviceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceId::RoadsideUnit(id) => write!(f, "RSU-{id:04}"),
            DeviceId::ZoneProcessor(id) => write!(f, "ZP-{id:04}"),
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
    ConfigureExperiment { config: ExperimentConfig },
    Shutdown,
    ReadyToStart,
    BeginExperiment,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TimingMetadata {
    pub sequence_id: u64,
    pub frame_number: u64,
    pub pi_hostname: String,
    pub pi_capture_start: Option<u64>,
    pub pi_sent_to_jetson: Option<u64>,
    pub jetson_received: Option<u64>,
    pub jetson_sent_result: Option<u64>,
    pub controller_sent_pulse: Option<u64>,
    pub controller_received: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameMessage {
    pub sequence_id: u64,
    pub timing: TimingMetadata,
    pub frame: Frame,
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
