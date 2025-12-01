use anyhow::Context as AnyhowContext;
use anyhow::Result;
use codec::types::{EncodingSpec, ImageCodecKind, ImageResolutionType, Tier};
use codec::ImageCodec;
use common::constants::{codec_name, res_folder};
use image::{imageops::FilterType, DynamicImage, GenericImageView};
use log::{error, info};
use rayon::prelude::*;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

struct ConversionTask {
    input_path: PathBuf,
    output_base: PathBuf,
    frame_number: String,
}

struct ConversionStats {
    total: usize,
    succeeded: usize,
    failed: usize,
}

const OUTPUT_DIR: &str = "services/roadside-unit/sample";

impl ConversionStats {
    const fn new(total: usize) -> Self {
        Self {
            total,
            succeeded: 0,
            failed: 0,
        }
    }

    fn report_success(&mut self) {
        self.succeeded += 1;

        if self.succeeded.is_multiple_of(50) || self.succeeded == self.total {
            let percent = (self.succeeded as f32 / self.total as f32) * 100.0;

            info!(
                "Progress: {}/{} ({:.1}%)",
                self.succeeded, self.total, percent
            );
        }
    }

    fn report_failure(&mut self) {
        self.failed += 1;
    }

    fn print_summary(&self) {
        info!("\n=================================================================");
        info!("Conversion Summary:");
        info!("  Total:     {}", self.total);
        info!("  Succeeded: {}", self.succeeded);
        info!("  Failed:    {}", self.failed);
        info!("=================================================================\n");
    }
}

// ---------------- IMAGE RESIZING ----------------

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
    assert!(orig_w > 0 && orig_h > 0);
    assert!(target_w > 0 && target_h > 0);

    let scale_w = target_w as f32 / orig_w as f32;
    let scale_h = target_h as f32 / orig_h as f32;
    let scale = scale_w.min(scale_h);

    // Explicit rounding — no truncation bias
    let new_w = (orig_w as f32 * scale).round() as u32;
    let new_h = (orig_h as f32 * scale).round() as u32;

    let resized = img.resize_exact(new_w, new_h, FilterType::Lanczos3);

    let mut canvas = DynamicImage::new_rgb8(target_w, target_h);

    let offset_x = (target_w - new_w) / 2;
    let offset_y = (target_h - new_h) / 2;

    image::imageops::overlay(&mut canvas, &resized, offset_x as i64, offset_y as i64);

    canvas
}

// ---------------- SINGLE CONVERSION ----------------

fn process_single_image(
    task: &ConversionTask,
    resolution: ImageResolutionType,
    codec: ImageCodecKind,
    tier: Tier,
) -> Result<(), Box<dyn Error>> {
    let img = image::open(&task.input_path)
        .with_context(|| format!("Failed to open image: {}", task.input_path.display()))?;

    let resized = resize_image(&img, resolution);

    let mut temp_png = Vec::new();
    {
        use image::ImageFormat;
        use std::io::Cursor;

        resized
            .write_to(&mut Cursor::new(&mut temp_png), ImageFormat::Png)
            .context("Failed to encode to temporary PNG")?;
    }

    let spec = EncodingSpec {
        codec,
        tier,
        resolution,
    };

    let frame = ImageCodec::compress_from_bytes(&temp_png, spec)
        .context("Failed to compress with ImageCodec")?;

    let res_dir = task.output_base.join(res_folder(resolution));
    let codec_dir = res_dir.join(codec_name(codec));
    fs::create_dir_all(&codec_dir)?;

    let tier_suffix = format!("{:?}", tier);

    let output_file =
        codec_dir.join(format!("{}_{}.{}", task.frame_number, tier_suffix, codec_name(codec)));

    fs::write(&output_file, frame.frame_data)
        .with_context(|| format!("Failed to write output: {}", output_file.display()))?;

    Ok(())
}

// ---------------- IMAGE DISCOVERY ----------------

fn discover_input_images(dir: &Path) -> Result<Vec<ConversionTask>> {
    Ok(fs::read_dir(dir)?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .map(|p| {
            let frame_number = p
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_owned();

            ConversionTask {
                input_path: p,
                output_base: PathBuf::from(OUTPUT_DIR),
                frame_number,
            }
        })
        .collect())
}

// ---------------- MAIN ----------------

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        error!("Usage: {} <input_directory>", args[0]);
        error!("Example:");
        error!("  {} ./source_images/", args[0]);
        std::process::exit(1);
    }

    let input_dir = PathBuf::from(&args[1]);

    if !input_dir.exists() || !input_dir.is_dir() {
        error!("Input directory does not exist: {}", input_dir.display());
        std::process::exit(1);
    }

    info!("=================================================================");
    info!("Image Preprocessor for Rust Traffic Watch");
    info!("=================================================================");
    info!("Input directory: {}", input_dir.display());
    info!("Output directory: {}\n", OUTPUT_DIR);

    info!("Discovering input images...");
    let tasks = discover_input_images(&input_dir)?;

    if tasks.is_empty() {
        error!("No images found in input directory");
        std::process::exit(1);
    }

    info!("Found {} images", tasks.len());

    let resolutions = vec![
        ImageResolutionType::FHD,
        ImageResolutionType::HD,
        ImageResolutionType::Letterbox,
    ];

    let codecs = vec![
        ImageCodecKind::JpgLossy,
        ImageCodecKind::PngLossless,
        ImageCodecKind::WebpLossy,
        ImageCodecKind::WebpLossless,
    ];

    let tiers = vec![Tier::T1, Tier::T2, Tier::T3];

    let total_combinations = resolutions.len() * codecs.len() * tiers.len();
    let total_conversions = tasks.len() * total_combinations;

    info!("\nConfiguration:");
    info!("  Resolutions: {}", resolutions.len());
    info!("  Codecs: {}", codecs.len());
    info!("  Tiers: {}", tiers.len());
    info!("  Conversions/image: {total_combinations}");
    info!("  Total conversions: {total_conversions}");

    fs::create_dir_all(OUTPUT_DIR)?;

    let stats = Arc::new(Mutex::new(ConversionStats::new(total_conversions)));

    info!("Starting parallel conversion...\n");

    let mut work_items = Vec::new();
    for resolution in &resolutions {
        for codec in &codecs {
            for tier in &tiers {
                for task in &tasks {
                    work_items.push((task, *resolution, *codec, *tier));
                }
            }
        }
    }

    work_items.par_iter().for_each(|(task, resolution, codec, tier)| {
        match process_single_image(task, *resolution, *codec, *tier) {
            Ok(()) => {
                let mut stats = stats.lock().unwrap();
                stats.report_success();
            }
            Err(e) => {
                error!("Failed: {} → {e}", task.input_path.display());
                let mut stats = stats.lock().unwrap();
                stats.report_failure();
            }
        }
    });

    let stats = stats.lock().unwrap();
    stats.print_summary();

    if stats.failed > 0 {
        info!("Some conversions failed.");
    } else {
        info!("All conversions completed successfully!");
    }

    Ok(())
}
