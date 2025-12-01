use anyhow::{anyhow, Result};
use image::{
    imageops::FilterType, DynamicImage, ExtendedColorType, GenericImageView, ImageEncoder,
};
use protocol::types::{tiers, ImageCodecKind, ImageResolutionType, Tier};

pub fn process_and_encode(
    img: &DynamicImage,
    target_res: ImageResolutionType,
    codec: ImageCodecKind,
    tier: Tier,
) -> Result<Vec<u8>> {
    let resized = resize_image(img, target_res);

    match codec {
        ImageCodecKind::JpgLossy => encode_jpeg(&resized, tier),
        ImageCodecKind::PngLossless => encode_png(&resized, tier),
        ImageCodecKind::WebpLossy => encode_webp_lossy(&resized, tier),
        ImageCodecKind::WebpLossless => encode_webp_lossless(&resized, tier),
    }
}

fn resize_image(img: &DynamicImage, target_res: ImageResolutionType) -> DynamicImage {
    match target_res {
        ImageResolutionType::FHD => img.clone(),
        ImageResolutionType::HD => img.resize_exact(1280, 720, FilterType::Lanczos3),
        ImageResolutionType::Letterbox => letterbox_resize(img, 640, 640),
    }
}

fn letterbox_resize(img: &DynamicImage, target_w: u32, target_h: u32) -> DynamicImage {
    let (orig_w, orig_h) = img.dimensions();

    // Prevent NaN / Inf / divide-by-zero
    if orig_w == 0 || orig_h == 0 || target_w == 0 || target_h == 0 {
        // Return original or empty in worst case, or handle error.
        // For now, assuming panic or upstream checks, but let's be safe:
        return img.resize_exact(target_w, target_h, FilterType::Lanczos3);
    }

    let scale_w = target_w as f32 / orig_w as f32;
    let scale_h = target_h as f32 / orig_h as f32;
    let scale = scale_w.min(scale_h);

    let new_w = (orig_w as f32 * scale).round() as u32;
    let new_h = (orig_h as f32 * scale).round() as u32;

    let resized = img.resize_exact(new_w, new_h, FilterType::Lanczos3);
    let mut canvas = DynamicImage::new_rgb8(target_w, target_h);

    let offset_x = (target_w - new_w) / 2;
    let offset_y = (target_h - new_h) / 2;

    image::imageops::overlay(&mut canvas, &resized, offset_x as i64, offset_y as i64);

    canvas.into()
}

fn encode_png(img: &DynamicImage, tier: Tier) -> Result<Vec<u8>> {
    use image::codecs::png::{CompressionType, FilterType as PngFilterType, PngEncoder};

    let rgba = img.to_rgba8();
    let (w, h) = (rgba.width(), rgba.height());

    let (compression, filter) = match tier {
        Tier::T1 => (CompressionType::Best, PngFilterType::Adaptive),
        Tier::T2 => (CompressionType::Default, PngFilterType::Adaptive),
        Tier::T3 => (CompressionType::Fast, PngFilterType::NoFilter),
    };

    let mut buf = Vec::new();
    let encoder = PngEncoder::new_with_quality(&mut buf, compression, filter);
    encoder.write_image(&rgba, w, h, ExtendedColorType::from(image::ColorType::Rgba8))?;
    Ok(buf)
}

fn encode_jpeg(img: &DynamicImage, tier: Tier) -> Result<Vec<u8>> {
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

fn encode_webp_lossless(img: &DynamicImage, tier: Tier) -> Result<Vec<u8>> {
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

fn encode_webp_lossy(img: &DynamicImage, tier: Tier) -> Result<Vec<u8>> {
    use webp::{Encoder, WebPConfig};

    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let enc = Encoder::from_rgb(&rgb, w, h);

    let mut cfg = WebPConfig::new().unwrap();
    cfg.lossless = 0;
    cfg.quality = tiers::webp_lossy_quality(tier);
    cfg.method = 4;

    let mem =
        enc.encode_advanced(&cfg).map_err(|e| anyhow!("libwebp lossy encode failed: {:?}", e))?;
    Ok(mem.to_vec())
}
