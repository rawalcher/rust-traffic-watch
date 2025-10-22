use crate::types::{Tier, tiers};
use anyhow::{Result, anyhow};
use image::{DynamicImage, ExtendedColorType, ImageEncoder};

pub fn encode_png(img: &DynamicImage, tier: Tier) -> Result<Vec<u8>> {
    use image::codecs::png::{CompressionType, FilterType, PngEncoder};

    let rgba = img.to_rgba8();
    let (w, h) = (rgba.width(), rgba.height());

    let (compression, filter) = match tier {
        Tier::T1 => (CompressionType::Best, FilterType::Adaptive),
        Tier::T2 => (CompressionType::Default, FilterType::Adaptive),
        Tier::T3 => (CompressionType::Fast, FilterType::NoFilter),
    };

    let mut buf = Vec::new();
    let encoder = PngEncoder::new_with_quality(&mut buf, compression, filter);
    encoder.write_image(
        &rgba,
        w,
        h,
        ExtendedColorType::from(image::ColorType::Rgba8),
    )?;
    Ok(buf)
}

pub fn encode_jpeg(img: &DynamicImage, tier: Tier) -> Result<Vec<u8>> {
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
    c.write_scanlines(&rgb)?;
    Ok(c.finish()?)
}

pub fn encode_webp_lossless(img: &DynamicImage, tier: Tier) -> Result<Vec<u8>> {
    use webp::{Encoder, WebPConfig};

    let rgba = img.to_rgba8();
    let (w, h) = (rgba.width(), rgba.height());
    let enc = Encoder::from_rgba(&rgba, w, h);

    let mut cfg = WebPConfig::new().unwrap();
    cfg.lossless = 1;
    cfg.method = tiers::webp_lossless_method(tier);
    cfg.quality = 100.0;
    cfg.alpha_compression = 0;

    let mem = enc
        .encode_advanced(&cfg)
        .map_err(|e| anyhow!("libwebp lossless encode failed: {:?}", e))?;
    Ok(mem.to_vec())
}

pub fn encode_webp_lossy(img: &DynamicImage, tier: Tier) -> Result<Vec<u8>> {
    use webp::{Encoder, WebPConfig};

    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let enc = Encoder::from_rgb(&rgb, w, h);

    let mut cfg = WebPConfig::new().unwrap();
    cfg.lossless = 0;
    cfg.quality = tiers::webp_lossy_quality(tier);
    cfg.method = 4;

    let mem = enc
        .encode_advanced(&cfg)
        .map_err(|e| anyhow!("libwebp lossy encode failed: {:?}", e))?;
    Ok(mem.to_vec())
}
