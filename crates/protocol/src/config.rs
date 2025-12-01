use std::time::Duration;

pub const CONTROLLER_PORT: u16 = 9090;
pub const JETSON_PORT: u16 = 9092;

pub const CONTROLLER_ADDRESS: &str = "localhost";
pub const JETSON_ADDRESS: &str = "localhost";
pub const PI_ADDRESS: &str = "localhost";

// echo 120 | sudo tee /sys/devices/pwm-fan/target_pwm

pub const DEFAULT_MODEL: &str = "yolov5n";
pub const DEFAULT_DURATION_SECONDS: u64 = 10;
pub const SEND_FPS: u64 = 1;
pub const SOURCE_FPS: u64 = 30;

pub const MAX_FRAME_SEQUENCE: u64 = 30000;
pub const FRAME_WIDTH: u32 = 1920;
pub const FRAME_HEIGHT: u32 = 1080;

/// will be removed
pub const PYTHON_VENV_PATH: &str = "python3";
pub const INFERENCE_PYTORCH_PATH: &str = "python/inference_pytorch.py";
pub const INFERENCE_TENSORRT_PATH: &str = "python/inference_tensorrt.py";
/// will be removed

#[must_use]
pub fn fps_to_interval(fps: u64) -> Duration {
    assert!(fps > 0, "FPS must not be zero");
    Duration::from_micros(1_000_000 / fps)
}

pub fn compute_skip(send_fps: u64) -> u64 {
    assert!(send_fps > 0, "send_fps must not be zero");
    assert!(SOURCE_FPS >= send_fps, "send_fps must not exceed source_fps");
    assert_eq!(SOURCE_FPS % send_fps, 0, "send_fps must divide source_fps exactly");

    SOURCE_FPS / send_fps
}

pub fn controller_address() -> String {
    format!("{}:{}", CONTROLLER_ADDRESS, CONTROLLER_PORT)
}

pub fn controller_bind_address() -> String {
    format!("0.0.0.0:{}", CONTROLLER_PORT)
}

pub fn jetson_address() -> String {
    format!("{}:{}", JETSON_ADDRESS, JETSON_PORT)
}

pub fn jetson_bind_address() -> String {
    format!("0.0.0.0:{}", JETSON_PORT)
}

// PNG: zlib compression levels (1–9, higher = more compression)
pub const PNG_ZLIB_LEVEL: [u8; 3] = [6, 3, 1];

// JPEG: quality scale (0–100, higher = better quality)
pub const JPEG_QUALITY: [u8; 3] = [90, 75, 60];

// WebP Lossy: quality scale (0–100)
pub const WEBP_LOSSY_QUALITY: [f32; 3] = [90.0, 75.0, 60.0];

// WebP Lossless: method 0–6 (compression effort)
pub const WEBP_LOSSLESS_METHOD: [i32; 3] = [6, 3, 0];
