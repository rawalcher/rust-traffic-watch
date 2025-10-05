use crate::{ImageCodecKind, ImageResolutionType};

pub const CONTROLLER_PORT: u16 = 9090;
pub const JETSON_PORT: u16 = 9092;

pub const CONTROLLER_ADDRESS: &str = "10.0.0.20";
pub const JETSON_ADDRESS: &str = "10.0.0.21";
pub const PI_ADDRESS: &str = "10.0.0.25";

// echo 120 | sudo tee /sys/devices/pwm-fan/target_pwm

pub const DEFAULT_MODEL: &str = "yolov5n";
pub const DEFAULT_DURATION_SECONDS: u64 = 180;

// at 24 fps or higher rust starts to behave weirdly
pub const DEFAULT_SEND_FPS: f32 = 1.0;
pub const SOURCE_FPS: f32 = 30.0;

pub const MAX_FRAME_SEQUENCE: u64 = 30000;
pub const FRAME_WIDTH: u32 = 1920;
pub const FRAME_HEIGHT: u32 = 1080;

pub const PYTHON_VENV_PATH: &str = "python3";
pub const INFERENCE_PYTORCH_PATH: &str = "python/inference_pytorch.py";
pub const INFERENCE_TENSORRT_PATH: &str = "python/inference_tensorrt.py";

// image compression
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Tier { T1, T2, T3 }

impl Tier {
    pub const ALL: [Tier; 3] = [Tier::T1, Tier::T2, Tier::T3];
    pub fn idx(self) -> usize { match self { Tier::T1 => 0, Tier::T2 => 1, Tier::T3 => 2 } }
}

pub const PNG_ZLIB_LEVEL: [u8; 3] = [6, 9, 0];

pub const JPEG_QUALITY: [u8; 3] = [85, 75, 65];

pub const WEBP_LOSSY_QUALITY: [u8; 3] = [85, 75, 65];

pub const WEBP_LOSSLESS_EFFORT: [u8; 3] = [6, 9, 0];

// pi and jetson connect here
pub fn controller_address() -> String {
    format!("{}:{}", CONTROLLER_ADDRESS, CONTROLLER_PORT)
}
pub fn controller_bind_address() -> String {
    format!("0.0.0.0:{}", CONTROLLER_PORT)
}

// pi connects here
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
    use super::{Tier, PNG_ZLIB_LEVEL, JPEG_QUALITY, WEBP_LOSSY_QUALITY, WEBP_LOSSLESS_EFFORT};

    #[inline]
    pub fn png_level(t: Tier) -> u8 {
        PNG_ZLIB_LEVEL[t.idx()]
    }
    #[inline]
    pub fn png_compression(t: Tier) -> png::Compression {
        match png_level(t) {
            0 => png::Compression::Fast,
            9 => png::Compression::Best,
            _ => png::Compression::Default,
        }
    }

    #[inline]
    pub fn jpeg_quality(t: Tier) -> u8 {
        JPEG_QUALITY[t.idx()]
    }

    #[inline]
    pub fn webp_lossy_quality(t: Tier) -> f32 {
        WEBP_LOSSY_QUALITY[t.idx()] as f32
    }

    #[inline]
    pub fn webp_lossless_method(t: Tier) -> i32 {
        let z = WEBP_LOSSLESS_EFFORT[t.idx()];
        z.min(6) as i32
    }
}

pub fn res_folder(res: ImageResolutionType) -> &'static str {
    match res {
        ImageResolutionType::FHD => "fhd",
        ImageResolutionType::HD => "hd",
        ImageResolutionType::LETTERBOX => "640",
    }
}

pub fn codec_folder(codec: ImageCodecKind) -> &'static str {
    match codec {
        ImageCodecKind::JpegLossy => "jpeg",
        ImageCodecKind::PngLossless => "png",
        ImageCodecKind::WebpLossy => "webp",
        ImageCodecKind::WebpLossless => "webp-lossless",
    }
}

pub fn codec_ext(codec: ImageCodecKind) -> &'static str {
    match codec {
        ImageCodecKind::JpegLossy => "jpg",
        ImageCodecKind::PngLossless => "png",
        ImageCodecKind::WebpLossy | ImageCodecKind::WebpLossless => "webp",
    }
}