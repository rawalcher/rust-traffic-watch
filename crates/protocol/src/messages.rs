use crate::types::{Detection, ExperimentConfig, ExperimentMode, Frame};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Copy, Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceId {
    RoadsideUnit(u8),
    ZoneProcessor(u8),
}

impl fmt::Display for DeviceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RoadsideUnit(id) => write!(f, "RSU-{id:04}"),
            Self::ZoneProcessor(id) => write!(f, "ZP-{id:04}"),
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
pub struct TimingMetadata {
    pub sequence_id: u64,
    pub frame_number: u64,
    pub source_device: DeviceId,
    pub mode: ExperimentMode,

    // Controller timestamps
    pub controller_sent_pulse: Option<u64>,
    pub controller_received: Option<u64>,

    // Roadside Unit (RSU) timestamps - always populated
    pub capture_start: Option<u64>,
    pub encode_complete: Option<u64>,
    pub send_start: Option<u64>,

    // Zone Processor (ZP) timestamps - only for remote mode
    pub receive_start: Option<u64>,
    pub queued_for_inference: Option<u64>,
    pub inference_start: Option<u64>,
    pub inference_complete: Option<u64>,
    pub send_result: Option<u64>,
}

impl TimingMetadata {
    #[must_use]
    pub const fn new_with_mode(
        device_id: DeviceId,
        sequence_id: u64,
        frame_number: u64,
        mode: ExperimentMode,
    ) -> Self {
        Self {
            // no default bc of device_id
            source_device: device_id,
            sequence_id,
            frame_number,
            mode,
            controller_sent_pulse: None,
            controller_received: None,
            capture_start: None,
            encode_complete: None,
            send_start: None,
            receive_start: None,
            queued_for_inference: None,
            inference_start: None,
            inference_complete: None,
            send_result: None,
        }
    }

    #[must_use]
    pub const fn is_local_mode(&self) -> bool {
        matches!(self.mode, ExperimentMode::Local)
    }

    #[must_use]
    pub const fn is_remote_mode(&self) -> bool {
        matches!(self.mode, ExperimentMode::Offload)
    }

    #[must_use]
    pub const fn total_latency(&self) -> Option<u64> {
        match (self.controller_sent_pulse, self.controller_received) {
            (Some(start), Some(end)) => Some(end.saturating_sub(start)),
            _ => None,
        }
    }

    #[must_use]
    pub const fn rsu_overhead(&self) -> Option<u64> {
        match (self.capture_start, self.send_start) {
            (Some(start), Some(end)) => Some(end.saturating_sub(start)),
            _ => None,
        }
    }

    #[must_use]
    pub const fn network_latency(&self) -> Option<u64> {
        if self.is_local_mode() {
            return None;
        }

        match (self.total_latency(), self.rsu_overhead(), self.zp_processing(), self.zp_queueing())
        {
            (Some(total), Some(rsu), Some(zp_processing), Some(zp_queue)) => Some(
                total.saturating_sub(rsu).saturating_sub(zp_processing).saturating_sub(zp_queue),
            ),
            _ => None,
        }
    }

    #[must_use]
    pub const fn zp_processing(&self) -> Option<u64> {
        if self.is_local_mode() {
            return None;
        }

        match (self.receive_start, self.send_result) {
            (Some(recv), Some(send)) => Some(send.saturating_sub(recv)),
            _ => None,
        }
    }

    #[must_use]
    pub const fn zp_queueing(&self) -> Option<u64> {
        if self.is_local_mode() {
            return None;
        }
        match (self.queued_for_inference, self.receive_start) {
            (Some(queued), Some(start)) => Some(start.saturating_sub(queued)),
            _ => None,
        }
    }

    #[must_use]
    pub const fn inference_time(&self) -> Option<u64> {
        match (self.inference_start, self.inference_complete) {
            (Some(start), Some(end)) => Some(end.saturating_sub(start)),
            _ => None,
        }
    }
}

/// # Panics
#[must_use]
pub fn current_timestamp_micros() -> u64 {
    u64::try_from(
        SystemTime::now().duration_since(UNIX_EPOCH).expect("Time went backwards").as_micros(),
    )
    .expect("Timestamp overflow (not expected until the year 584,554)")
}
