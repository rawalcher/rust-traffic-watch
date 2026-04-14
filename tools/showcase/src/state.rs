use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Default)]
pub struct LatestFrame {
    pub jpeg_bytes: Vec<u8>,
    pub frame_number: u64,
    pub inference_us: u64,
    pub detection_count: u32,
    pub class_counts: HashMap<String, u32>,
    pub model_name: String,
}

pub struct SharedState {
    pub latest: RwLock<LatestFrame>,
    pub started: Instant,
    pub total_inferences: RwLock<u64>,
    pub device_label: String,
}

impl SharedState {
    pub fn new(device_label: String) -> Arc<Self> {
        Arc::new(Self {
            latest: RwLock::new(LatestFrame::default()),
            started: Instant::now(),
            total_inferences: RwLock::new(0),
            device_label,
        })
    }
}
