use serde::{Deserialize, Serialize};
use strum_macros::Display;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Frame {
    pub frame_data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub encoding: EncodingSpec,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Display)]
pub enum ImageCodecKind {
    JpgLossy,
    PngLossless,
    WebpLossy,
    WebpLossless,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Display)]
pub enum Tier {
    T1,
    T2,
    T3,
}
impl Tier {
    pub const ALL: [Tier; 3] = [Tier::T1, Tier::T2, Tier::T3];

    #[inline(always)]
    pub const fn idx(self) -> usize {
        match self {
            Tier::T1 => 0,
            Tier::T2 => 1,
            Tier::T3 => 2,
        }
    }
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

// PNG: zlib compression levels (1–9, higher = more compression)
const PNG_ZLIB_LEVEL: [u8; 3] = [6, 3, 1];

// JPEG: quality scale (0–100, higher = better quality)
const JPEG_QUALITY: [u8; 3] = [90, 75, 60];

// WebP Lossy: quality scale (0–100)
const WEBP_LOSSY_QUALITY: [f32; 3] = [90.0, 75.0, 60.0];

// WebP Lossless: method 0–6 (compression effort)
const WEBP_LOSSLESS_METHOD: [i32; 3] = [6, 3, 0];

pub mod tiers {
    use super::*;
    use image::codecs::png;

    #[inline]
    pub fn png_level(t: Tier) -> u8 {
        PNG_ZLIB_LEVEL[t.idx()]
    }

    #[inline]
    pub fn png_compression(t: Tier) -> png::CompressionType {
        match png_level(t) {
            1 => png::CompressionType::Fast,
            3 => png::CompressionType::Default,
            6 => png::CompressionType::Best,
            _ => png::CompressionType::Default,
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
