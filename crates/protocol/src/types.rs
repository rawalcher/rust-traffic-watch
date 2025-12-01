use crate::config::{
    DEFAULT_DURATION_SECONDS, JPEG_QUALITY, PNG_ZLIB_LEVEL, SEND_FPS, WEBP_LOSSLESS_METHOD,
    WEBP_LOSSY_QUALITY,
};
use image::codecs::png;
use serde::{Deserialize, Serialize};
use strum_macros::Display;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Display)]
pub enum Tier {
    T1, // Highest Quality/Best Compression
    T2, // Default/Medium Quality
    T3, // Lowest Quality/Fastest Compression
}

impl Tier {
    pub const ALL: [Self; 3] = [Self::T1, Self::T2, Self::T3];

    #[must_use]
    pub const fn idx(self) -> usize {
        match self {
            Self::T1 => 0,
            Self::T2 => 1,
            Self::T3 => 2,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Display)]
pub enum ImageCodecKind {
    JpgLossy,
    PngLossless,
    WebpLossy,
    WebpLossless,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Display)]
pub enum ImageResolutionType {
    FHD,
    HD,
    Letterbox,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EncodingSpec {
    pub codec: ImageCodecKind,
    pub tier: Tier,
    pub resolution: ImageResolutionType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Frame {
    pub frame_data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub encoding: EncodingSpec,
}

#[derive(Debug, Clone, Serialize, Deserialize, Display)]
pub enum ExperimentMode {
    Local,
    Offload,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    pub experiment_id: String,
    pub model_name: String,
    pub mode: ExperimentMode,
    pub encoding_spec: EncodingSpec,
    pub duration_seconds: u64,
    pub fixed_fps: u64,
}

impl ExperimentConfig {
    #[must_use]
    pub const fn new(
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection {
    pub class: String,
    pub bbox: [f32; 4],
    pub confidence: f32,
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

pub mod tiers {
    use super::{
        png, ImageCodecKind, ImageResolutionType, Tier, JPEG_QUALITY,
        PNG_ZLIB_LEVEL, WEBP_LOSSLESS_METHOD, WEBP_LOSSY_QUALITY,
    };

    #[inline]
    #[must_use]
    pub const fn png_level(t: Tier) -> u8 {
        PNG_ZLIB_LEVEL[t.idx()]
    }

    #[inline]
    #[must_use]
    pub const fn png_compression(t: Tier) -> png::CompressionType {
        match png_level(t) {
            1 => png::CompressionType::Fast,
            6 => png::CompressionType::Best,
            _ => png::CompressionType::Default,
        }
    }

    #[inline]
    #[must_use]
    pub const fn jpeg_quality(t: Tier) -> u8 {
        JPEG_QUALITY[t.idx()]
    }

    #[inline]
    #[must_use]
    pub const fn webp_lossy_quality(t: Tier) -> f32 {
        WEBP_LOSSY_QUALITY[t.idx()]
    }

    #[inline]
    #[must_use]
    pub const fn webp_lossless_method(t: Tier) -> i32 {
        WEBP_LOSSLESS_METHOD[t.idx()]
    }

    #[must_use]
    pub const fn res_folder(res: ImageResolutionType) -> &'static str {
        match res {
            ImageResolutionType::FHD => "FHD",
            ImageResolutionType::HD => "HD",
            ImageResolutionType::Letterbox => "640",
        }
    }

    #[must_use]
    pub const fn codec_name(codec: ImageCodecKind) -> &'static str {
        match codec {
            ImageCodecKind::JpgLossy => "jpg",
            ImageCodecKind::PngLossless => "png",
            ImageCodecKind::WebpLossy | ImageCodecKind::WebpLossless => "webp",
        }
    }
}
