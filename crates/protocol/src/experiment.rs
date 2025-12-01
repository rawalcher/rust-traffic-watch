use codec::types::EncodingSpec;
use crate::constants::{DEFAULT_DURATION_SECONDS, SEND_FPS};
use serde::{Deserialize, Serialize};
use strum_macros::Display;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    pub experiment_id: String,
    pub model_name: String,
    pub mode: ExperimentMode,
    pub encoding_spec: EncodingSpec,
    pub duration_seconds: u64,
    pub fixed_fps: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Display)]
pub enum ExperimentMode {
    Local,
    Offload,
}

impl ExperimentConfig {
    pub fn new(
        experiment_id: String,
        mode: ExperimentMode,
        model_name: String,
        encoding_spec: EncodingSpec,
    ) -> Self {
        Self {
            experiment_id,
            model_name,
            mode,
            encoding_spec,
            duration_seconds: DEFAULT_DURATION_SECONDS,
            fixed_fps: SEND_FPS,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
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
