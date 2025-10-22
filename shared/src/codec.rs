use anyhow::{Context, Result};
use anyhow::anyhow;
use std::{fs, path::Path};
use image::{DynamicImage, GenericImageView};
use crate::constants::{Tier};
use crate::Frame;
use crate::types::{ImageCodecKind, EncodingSpec};

pub struct ImageCodec;

impl ImageCodec {
    pub fn compress_from_path(path: &Path, spec: EncodingSpec) -> Result<Frame> {
        let bytes: Vec<u8> = fs::read(path)
            .with_context(|| format!("read {:?}", path))?;
        Self::compress_from_bytes(bytes.as_slice(), spec)
    }
}

fn encode_png(img: &DynamicImage, tier: Tier) -> Result<Vec<u8>> {
    use png::{Encoder};
    use crate::constants::tiers;

    let mut buf = Vec::new();
    {
        let rgba = img.to_rgba8();
        let (w, h) = (rgba.width(), rgba.height());

        let mut enc = Encoder::new(&mut buf, w, h);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);

        let comp = tiers::png_compression(tier);
        enc.set_compression(comp);

        let mut writer = enc.write_header()?;
        writer.write_image_data(&rgba)?;
    }
    Ok(buf)
}

fn encode_jpeg(img: &DynamicImage, tier: Tier) -> Result<Vec<u8>> {
    use crate::constants::tiers;
    use mozjpeg::{ColorSpace, Compress, ScanMode};

    let q = tiers::jpeg_quality(tier).min(100);
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);

    let mut c = Compress::new(ColorSpace::JCS_RGB);
    c.set_size(w, h);
    c.set_quality(q as f32);
    c.set_scan_optimization_mode(ScanMode::Auto);
    c.set_progressive_mode();
    c.set_optimize_scans(true);

    let mut c = c.start_compress(Vec::new())?;
    c.write_scanlines(&rgb).expect("PANIC");
    let jpeg = c.finish()?;
    Ok(jpeg)
}


fn encode_webp_lossless(img: &DynamicImage, tier: Tier) -> Result<Vec<u8>> {
    use webp::{Encoder, WebPConfig};

    let rgba = img.to_rgba8();
    let (w, h) = (rgba.width(), rgba.height());
    let enc = Encoder::from_rgba(&rgba, w, h);

    let mut cfg = WebPConfig::new().unwrap();
    cfg.lossless = 1;
    cfg.method   = crate::constants::tiers::webp_lossless_method(tier); // 0..=6
    cfg.quality  = 100.0;
    cfg.alpha_compression = 0;

    let mem = enc.encode_advanced(&cfg)
        .map_err(|e| anyhow!("libwebp lossless encode failed: {:?}", e))?;
    Ok(mem.to_vec())
}

fn encode_webp_lossy(img: &DynamicImage, tier: Tier) -> Result<Vec<u8>> {
    use webp::{Encoder, WebPConfig};

    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let enc = Encoder::from_rgb(&rgb, w, h);

    let mut cfg = WebPConfig::new().unwrap();
    cfg.lossless = 0;
    cfg.quality  = crate::constants::tiers::webp_lossy_quality(tier);
    cfg.method   = 4;

    let mem = enc.encode_advanced(&cfg)
        .map_err(|e| anyhow!("libwebp lossy encode failed: {:?}", e))?;
    Ok(mem.to_vec())
}

#[cfg(test)]
mod bench {
    use super::*;
    use std::time::Instant;

    #[test]
    fn benchmark_encoding() {
        let img = DynamicImage::new_rgb8(1920, 1080);

        for _ in 0..3 {
            let _ = encode_jpeg(&img, Tier::T2);
        }

        let start = Instant::now();
        for _ in 0..10 {
            let _ = encode_jpeg(&img, Tier::T2);
        }
        let elapsed = start.elapsed();

        println!("Average JPEG encode time: {:?}", elapsed / 10);
    }
}