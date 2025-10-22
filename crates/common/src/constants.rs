// TODO: SPLIT UP MORE

use codec::types::{ImageCodecKind, ImageResolutionType};

pub const CONTROLLER_PORT: u16 = 9090;
pub const JETSON_PORT: u16 = 9092;

pub const CONTROLLER_ADDRESS: &str = "10.0.0.20";
pub const JETSON_ADDRESS: &str = "10.0.0.21";
pub const PI_ADDRESS: &str = "10.0.0.25";

// echo 120 | sudo tee /sys/devices/pwm-fan/target_pwm

pub const DEFAULT_MODEL: &str = "yolov5n";
pub const DEFAULT_DURATION_SECONDS: u64 = 10;
pub const DEFAULT_SEND_FPS: f32 = 1.0;
pub const SOURCE_FPS: f32 = 30.0;

pub const MAX_FRAME_SEQUENCE: u64 = 30000;
pub const FRAME_WIDTH: u32 = 1920;
pub const FRAME_HEIGHT: u32 = 1080;

pub const PYTHON_VENV_PATH: &str = "python3";
pub const INFERENCE_PYTORCH_PATH: &str = "python/inference_pytorch.py";
pub const INFERENCE_TENSORRT_PATH: &str = "python/inference_tensorrt.py";

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

pub fn get_frame_skip() -> Result<u64, &'static str> {
    let skip = (SOURCE_FPS / DEFAULT_SEND_FPS).ceil() as u64;
    Ok(skip.max(1))
}

pub fn res_folder(res: ImageResolutionType) -> &'static str {
    match res {
        ImageResolutionType::FHD => "FHD",
        ImageResolutionType::HD => "HD",
        ImageResolutionType::Letterbox => "640",
    }
}

pub fn codec_folder(codec: ImageCodecKind) -> &'static str {
    match codec {
        ImageCodecKind::JpgLossy => "jpg",
        ImageCodecKind::PngLossless => "png",
        ImageCodecKind::WebpLossy => "webp",
        ImageCodecKind::WebpLossless => "webp",
    }
}

pub fn codec_ext(codec: ImageCodecKind) -> &'static str {
    match codec {
        ImageCodecKind::JpgLossy => "jpg",
        ImageCodecKind::PngLossless => "png",
        ImageCodecKind::WebpLossy | ImageCodecKind::WebpLossless => "webp",
    }
}
