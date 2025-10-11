use crate::{ImageCodecKind, ImageResolutionType};

pub const CONTROLLER_PORT: u16 = 9090;
pub const JETSON_PORT: u16 = 9092;

pub const CONTROLLER_ADDRESS: &str = "10.0.0.20";
pub const JETSON_ADDRESS: &str = "10.0.0.21";
pub const PI_ADDRESS: &str = "10.0.0.25";

// echo 120 | sudo tee /sys/devices/pwm-fan/target_pwm

pub const DEFAULT_MODEL: &str = "yolov5n";
pub const DEFAULT_DURATION_SECONDS: u64 = 180;
pub const DEFAULT_SEND_FPS: f32 = 1.0;
pub const SOURCE_FPS: f32 = 30.0;

pub const MAX_FRAME_SEQUENCE: u64 = 30000;
pub const FRAME_WIDTH: u32 = 1920;
pub const FRAME_HEIGHT: u32 = 1080;

pub const PYTHON_VENV_PATH: &str = "python3";
pub const INFERENCE_PYTORCH_PATH: &str = "python/inference_pytorch.py";
pub const INFERENCE_TENSORRT_PATH: &str = "python/inference_tensorrt.py";

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Tier {
    T1,  // High quality / slower compression
    T2,  // Balanced
    T3   // Low quality / fast compression
}

impl Tier {
    pub const ALL: [Tier; 3] = [Tier::T1, Tier::T2, Tier::T3];

    #[inline]
    pub fn idx(self) -> usize {
        match self {
            Tier::T1 => 0,
            Tier::T2 => 1,
            Tier::T3 => 2
        }
    }
}

// PNG: Lossless compression - zlib levels
pub const PNG_ZLIB_LEVEL: [u8; 3] = [
    6,
    3,
    1
];

// JPEG: Lossy - quality scale 0-100
pub const JPEG_QUALITY: [u8; 3] = [
    90,
    75,
    60
];

// WebP Lossy: quality scale 0-100
pub const WEBP_LOSSY_QUALITY: [f32; 3] = [
    90.0,
    75.0,
    60.0
];

// WebP Lossless: method 0-6 (compression effort)
pub const WEBP_LOSSLESS_METHOD: [i32; 3] = [
    6,  // T1: Best compression
    3,  // T2: Balanced
    0   // T3: Fast (minimal compression)
];

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

pub mod tiers {
    use super::*;

    #[inline]
    pub fn png_level(t: Tier) -> u8 {
        PNG_ZLIB_LEVEL[t.idx()]
    }

    #[inline]
    pub fn png_compression(t: Tier) -> png::Compression {
        match png_level(t) {
            1 => png::Compression::Fast,
            3 => png::Compression::Default,
            6 => png::Compression::Default,
            _ => png::Compression::Default,
        }
    }

    #[inline]
    pub fn jpeg_quality(t: Tier) -> u8 {
        JPEG_QUALITY[t.idx()]
    }

    #[inline]
    pub fn webp_lossy_quality(t: Tier) -> f32 {
        WEBP_LOSSY_QUALITY[t.idx()]
    }

    #[inline]
    pub fn webp_lossless_method(t: Tier) -> i32 {
        WEBP_LOSSLESS_METHOD[t.idx()]
    }
}

pub fn res_folder(res: ImageResolutionType) -> &'static str {
    match res {
        ImageResolutionType::FHD => "FHD",
        ImageResolutionType::HD => "HD",
        ImageResolutionType::LETTERBOX => "640",
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