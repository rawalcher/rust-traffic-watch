use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FrameMessage {
    pub sequence_id: u64,
    pub timestamp: u64,
    pub frame_data: Vec<u8>,
    pub width: u32,
    pub height: u32,
}