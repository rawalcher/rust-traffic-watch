pub mod decoder;
pub mod encoder;
pub mod types;

use crate::decoder::decompress_to_rgb;
use crate::encoder::{encode_jpeg, encode_png, encode_webp_lossless, encode_webp_lossy};
use crate::types::{EncodingSpec, Frame, ImageCodecKind};
use anyhow::{Context, Result};
use image::GenericImageView;
use std::{fs, path::Path};

pub struct ImageCodec;

impl ImageCodec {
    pub fn compress_from_path(path: &Path, spec: EncodingSpec) -> Result<Frame> {
        let bytes = fs::read(path).with_context(|| format!("read {:?}", path))?;
        Self::compress_from_bytes(&bytes, spec)
    }

    pub fn compress_from_bytes(src_bytes: &[u8], spec: EncodingSpec) -> Result<Frame> {
        let img = image::load_from_memory(src_bytes).context("decode input")?;
        let (w, h) = img.dimensions();

        let out = match spec.codec {
            ImageCodecKind::PngLossless => encode_png(&img, spec.tier)?,
            ImageCodecKind::JpgLossy => encode_jpeg(&img, spec.tier)?,
            ImageCodecKind::WebpLossy => encode_webp_lossy(&img, spec.tier)?,
            ImageCodecKind::WebpLossless => encode_webp_lossless(&img, spec.tier)?,
        };

        Ok(Frame {
            frame_data: out,
            width: w,
            height: h,
            encoding: spec,
        })
    }

    pub fn decompress_to_rgb(src_bytes: &[u8]) -> Result<(u32, u32, Vec<u8>)> {
        decompress_to_rgb(src_bytes)
    }
}
