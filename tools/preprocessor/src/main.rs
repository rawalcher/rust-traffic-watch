use anyhow::{Context, Result};
use protocol::types::{
    tiers::{codec_name, res_folder}, ImageCodecKind, ImageResolutionType,
    Tier,
};
use rayon::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use tracing::{error, info};

const OUTPUT_DIR: &str = "services/roadside-unit/testImages";

struct ConversionTask {
    input_path: PathBuf,
    output_base: PathBuf,
    frame_number: String,
}

struct ConversionStats {
    total: usize,
    succeeded: AtomicUsize,
    failed: AtomicUsize,
}

impl ConversionStats {
    const fn new(total: usize) -> Self {
        Self { total, succeeded: AtomicUsize::new(0), failed: AtomicUsize::new(0) }
    }

    fn report_success(&self) {
        let current = self.succeeded.fetch_add(1, Ordering::Relaxed) + 1;
        if current.is_multiple_of(50) || current == self.total {
            let percent = (current as f32 / self.total as f32) * 100.0;
            info!("Progress: {}/{} ({:.1}%)", current, self.total, percent);
        }
    }

    fn report_failure(&self) {
        self.failed.fetch_add(1, Ordering::Relaxed);
    }

    fn print_summary(&self) {
        info!("=== Conversion Summary ===");
        info!("  Total:     {}", self.total);
        info!("  Succeeded: {}", self.succeeded.load(Ordering::Relaxed));
        info!("  Failed:    {}", self.failed.load(Ordering::Relaxed));
        info!("==========================\n");
    }
}

fn discover_input_images(dir: &Path) -> Result<Vec<ConversionTask>> {
    Ok(fs::read_dir(dir)?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .map(|p| {
            let frame_number =
                p.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown").to_owned();
            ConversionTask { input_path: p, output_base: PathBuf::from(OUTPUT_DIR), frame_number }
        })
        .collect())
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        error!("Usage: {} <input_directory>", args[0]);
        std::process::exit(1);
    }

    let input_dir = PathBuf::from(&args[1]);
    if !input_dir.exists() {
        error!("Input directory not found");
        std::process::exit(1);
    }

    let tasks = discover_input_images(&input_dir)?;
    if tasks.is_empty() {
        error!("No images found.");
        std::process::exit(1);
    }

    let resolutions =
        [ImageResolutionType::FHD, ImageResolutionType::HD, ImageResolutionType::Letterbox];
    let codecs = [
        ImageCodecKind::JpgLossy,
        ImageCodecKind::PngLossless,
        ImageCodecKind::WebpLossy,
        ImageCodecKind::WebpLossless,
    ];
    let tiers = [Tier::T1, Tier::T2, Tier::T3];

    let total_ops = tasks.len() * resolutions.len() * codecs.len() * tiers.len();
    let stats = ConversionStats::new(total_ops);

    info!("Starting {} conversions on {} images...", total_ops, tasks.len());
    fs::create_dir_all(OUTPUT_DIR)?;

    tasks.par_iter().for_each(|task| {
        let img_result = image::open(&task.input_path)
            .with_context(|| format!("Failed to open: {}", task.input_path.display()));

        match img_result {
            Ok(img) => {
                for &res in &resolutions {
                    let resized_img = image_processing::resize_image(&img, res);

                    for &codec in &codecs {
                        for &tier in &tiers {
                            match image_processing::encode_only(&resized_img, codec, tier) {
                                Ok(bytes) => {
                                    let res_dir = task.output_base.join(res_folder(res));
                                    let codec_dir = res_dir.join(codec_name(codec));
                                    let _ = fs::create_dir_all(&codec_dir);

                                    let output_file = codec_dir.join(format!(
                                        "{}_{:?}.{}",
                                        task.frame_number,
                                        tier,
                                        codec_name(codec)
                                    ));

                                    if fs::write(output_file, bytes).is_ok() {
                                        stats.report_success();
                                    } else {
                                        stats.report_failure();
                                    }
                                }
                                Err(e) => {
                                    error!("Encoding failed: {}", e);
                                    stats.report_failure();
                                }
                            }
                        }
                    }
                }
            }
            Err(e) => {
                error!("{}", e);
                for _ in 0..(resolutions.len() * codecs.len() * tiers.len()) {
                    stats.report_failure();
                }
            }
        }
    });

    stats.print_summary();
    Ok(())
}

pub mod image_processing {
    use super::{ImageCodecKind, ImageResolutionType, Result, Tier};
    use image::{
        imageops::FilterType, DynamicImage, ExtendedColorType, GenericImageView, ImageEncoder,
    };

    #[must_use]
    pub fn resize_image(img: &DynamicImage, target_res: ImageResolutionType) -> DynamicImage {
        match target_res {
            ImageResolutionType::FHD => img.clone(),
            ImageResolutionType::HD => img.resize_exact(1280, 720, FilterType::CatmullRom),
            ImageResolutionType::Letterbox => letterbox_resize(img, 640, 640),
        }
    }

    /// # Errors
    pub fn encode_only(img: &DynamicImage, codec: ImageCodecKind, tier: Tier) -> Result<Vec<u8>> {
        match codec {
            ImageCodecKind::JpgLossy => encode_jpeg(img, tier),
            ImageCodecKind::PngLossless => encode_png(img, tier),
            ImageCodecKind::WebpLossy => encode_webp_lossy(img, tier),
            ImageCodecKind::WebpLossless => encode_webp_lossless(img, tier),
        }
    }

    fn letterbox_resize(img: &DynamicImage, target_w: u32, target_h: u32) -> DynamicImage {
        let (orig_w, orig_h) = img.dimensions();
        if orig_w == 0 || orig_h == 0 {
            return DynamicImage::new_rgb8(target_w, target_h);
        }

        let scale = (target_w as f32 / orig_w as f32).min(target_h as f32 / orig_h as f32);
        let new_w = (orig_w as f32 * scale).round() as u32;
        let new_h = (orig_h as f32 * scale).round() as u32;

        let resized = img.resize_exact(new_w, new_h, FilterType::CatmullRom);
        let mut canvas = DynamicImage::new_rgb8(target_w, target_h);

        let offset_x = (target_w - new_w) / 2;
        let offset_y = (target_h - new_h) / 2;

        image::imageops::overlay(&mut canvas, &resized, i64::from(offset_x), i64::from(offset_y));
        canvas
    }

    fn encode_png(img: &DynamicImage, tier: Tier) -> Result<Vec<u8>> {
        use image::codecs::png::{CompressionType, FilterType as PngFilterType, PngEncoder};
        let rgba = img.to_rgba8();
        let (w, h) = rgba.dimensions();

        let (compression, filter) = match tier {
            Tier::T1 => (CompressionType::Best, PngFilterType::Adaptive),
            Tier::T2 => (CompressionType::Default, PngFilterType::Adaptive),
            Tier::T3 => (CompressionType::Fast, PngFilterType::NoFilter),
        };

        let mut buf = Vec::new();
        let encoder = PngEncoder::new_with_quality(&mut buf, compression, filter);
        encoder.write_image(&rgba, w, h, ExtendedColorType::Rgba8)?;
        Ok(buf)
    }

    fn encode_jpeg(img: &DynamicImage, tier: Tier) -> Result<Vec<u8>> {
        use mozjpeg::{ColorSpace, Compress, ScanMode};
        let q = protocol::types::tiers::jpeg_quality(tier).min(100);
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);

        let mut c = Compress::new(ColorSpace::JCS_RGB);
        c.set_size(w, h);
        c.set_quality(f32::from(q));
        // Optimization: Disable progressive for Tier 3 for speed
        if tier != Tier::T3 {
            c.set_scan_optimization_mode(ScanMode::Auto);
            c.set_progressive_mode();
        }

        let mut c = c.start_compress(Vec::new())?;
        c.write_scanlines(&rgb)?;
        Ok(c.finish()?)
    }

    fn encode_webp_lossless(img: &DynamicImage, tier: Tier) -> Result<Vec<u8>> {
        let rgba = img.to_rgba8();
        let (w, h) = rgba.dimensions();
        let enc = webp::Encoder::from_rgba(&rgba, w, h);

        let mut cfg = webp::WebPConfig::new().map_err(|()| anyhow::anyhow!("WebP config error"))?;
        cfg.lossless = 1;
        cfg.method = protocol::types::tiers::webp_lossless_method(tier);

        let mem = enc.encode_advanced(&cfg).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        Ok(mem.to_vec())
    }

    fn encode_webp_lossy(img: &DynamicImage, tier: Tier) -> Result<Vec<u8>> {
        let rgb = img.to_rgb8();
        let (w, h) = rgb.dimensions();
        let enc = webp::Encoder::from_rgb(&rgb, w, h);

        let mut cfg = webp::WebPConfig::new().map_err(|()| anyhow::anyhow!("WebP config error"))?;
        cfg.quality = protocol::types::tiers::webp_lossy_quality(tier);
        cfg.method = 4;

        let mem = enc.encode_advanced(&cfg).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        Ok(mem.to_vec())
    }
}
