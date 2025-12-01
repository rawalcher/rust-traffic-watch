mod image_processing;

use anyhow::{Context, Result};
use protocol::types::{
    tiers::{codec_name, res_folder}, ImageCodecKind, ImageResolutionType,
    Tier,
};
use rayon::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tracing::{error, info};

const OUTPUT_DIR: &str = "services/roadside-unit/testImages";

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

impl ConversionStats {
    const fn new(total: usize) -> Self {
        Self { total, succeeded: 0, failed: 0 }
    }

    fn report_success(&mut self) {
        self.succeeded += 1;
        if self.succeeded % 50 == 0 || self.succeeded == self.total {
            let percent = (self.succeeded as f32 / self.total as f32) * 100.0;
            info!("Progress: {}/{} ({:.1}%)", self.succeeded, self.total, percent);
        }
    }

    fn report_failure(&mut self) {
        self.failed += 1;
    }

    fn print_summary(&self) {
        info!("=== Conversion Summary ===");
        info!("  Total:     {}", self.total);
        info!("  Succeeded: {}", self.succeeded);
        info!("  Failed:    {}", self.failed);
        info!("==========================\n");
    }
}

fn process_single_task(
    task: &ConversionTask,
    resolution: ImageResolutionType,
    codec: ImageCodecKind,
    tier: Tier,
) -> Result<()> {
    let img = image::open(&task.input_path)
        .with_context(|| format!("Failed to open: {}", task.input_path.display()))?;

    let compressed_bytes = image_processing::process_and_encode(&img, resolution, codec, tier)
        .context("Processing failed")?;

    let res_dir = task.output_base.join(res_folder(resolution));
    let codec_dir = res_dir.join(codec_name(codec));
    fs::create_dir_all(&codec_dir)?;

    let tier_suffix = format!("{:?}", tier);
    let output_file =
        codec_dir.join(format!("{}_{}.{}", task.frame_number, tier_suffix, codec_name(codec)));

    fs::write(&output_file, compressed_bytes)?;
    Ok(())
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
        vec![ImageResolutionType::FHD, ImageResolutionType::HD, ImageResolutionType::Letterbox];
    let codecs = vec![
        ImageCodecKind::JpgLossy,
        ImageCodecKind::PngLossless,
        ImageCodecKind::WebpLossy,
        ImageCodecKind::WebpLossless,
    ];
    let tiers = vec![Tier::T1, Tier::T2, Tier::T3];

    let total_ops = tasks.len() * resolutions.len() * codecs.len() * tiers.len();
    let stats = Arc::new(Mutex::new(ConversionStats::new(total_ops)));

    info!("Starting {} conversions on {} images...", total_ops, tasks.len());
    fs::create_dir_all(OUTPUT_DIR)?;

    let mut work_items = Vec::with_capacity(total_ops);
    for task in &tasks {
        for res in &resolutions {
            for codec in &codecs {
                for tier in &tiers {
                    work_items.push((task, *res, *codec, *tier));
                }
            }
        }
    }

    work_items.par_iter().for_each(|(task, res, codec, tier)| {
        match process_single_task(task, *res, *codec, *tier) {
            Ok(_) => stats.lock().unwrap().report_success(),
            Err(e) => {
                error!("Task failed: {}", e);
                stats.lock().unwrap().report_failure();
            }
        }
    });

    stats.lock().unwrap().print_summary();
    Ok(())
}
