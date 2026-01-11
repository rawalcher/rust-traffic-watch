use std::time::Duration;

pub const CONTROLLER_PORT: u16 = 9090;
pub const ZONE_PROCESSOR_PORT: u16 = 9092;

pub const CONTROLLER_ADDRESS: &str = "10.0.0.110";
pub const ZONE_PROCESSOR_ADDRESS: &str = "10.0.0.178";

pub const DEFAULT_MODEL: &str = "yolov5n";
pub const DEFAULT_DURATION_SECONDS: u64 = 10;
pub const DEFAULT_SEND_FPS: u64 = 1;
pub const SOURCE_FPS: u64 = 30;
pub const DEFAULT_RSU_COUNT: u8 = 1;

/// Maximum frame sequence number before wrapping
pub const MAX_FRAME_SEQUENCE: u64 = 30000;

pub const FRAME_WIDTH: u32 = 1920;
pub const FRAME_HEIGHT: u32 = 1080;

/// PNG compression levels (1–9, higher = more compression)
/// [Best, Default, Fast]
pub const PNG_ZLIB_LEVEL: [u8; 3] = [6, 3, 1];

/// JPEG quality levels (0–100, higher = better quality)
/// [Best, Default, Fast]
pub const JPEG_QUALITY: [u8; 3] = [90, 75, 60];

/// WebP lossy quality levels (0–100, higher = better quality)
/// [Best, Default, Fast]
pub const WEBP_LOSSY_QUALITY: [f32; 3] = [90.0, 75.0, 60.0];

/// WebP lossless compression effort (0–6, higher = more compression)
/// [Best, Default, Fast]
pub const WEBP_LOSSLESS_METHOD: [i32; 3] = [6, 3, 0];

/// # Panics
#[must_use]
pub fn fps_to_interval(fps: u64) -> Duration {
    assert!(fps > 0, "FPS must not be zero");
    Duration::from_micros(1_000_000 / fps)
}

/// # Panics
#[must_use]
pub fn compute_skip(send_fps: u64) -> u64 {
    assert!(send_fps > 0, "send_fps must not be zero");
    assert!(SOURCE_FPS >= send_fps, "send_fps must not exceed SOURCE_FPS");
    assert_eq!(SOURCE_FPS % send_fps, 0, "send_fps must divide SOURCE_FPS ({SOURCE_FPS}) exactly");

    SOURCE_FPS / send_fps
}

#[must_use]
pub fn controller_address() -> String {
    format!("{CONTROLLER_ADDRESS}:{CONTROLLER_PORT}")
}

#[must_use]
pub fn controller_bind_address() -> String {
    format!("0.0.0.0:{CONTROLLER_PORT}")
}

#[must_use]
pub fn zone_processor_address() -> String {
    format!("{ZONE_PROCESSOR_ADDRESS}:{ZONE_PROCESSOR_PORT}")
}

#[must_use]
pub fn zone_processor_bind_address() -> String {
    format!("0.0.0.0:{ZONE_PROCESSOR_PORT}")
}
