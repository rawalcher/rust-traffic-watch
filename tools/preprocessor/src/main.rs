use anyhow::Context as AnyhowContext;
use codec::types::{EncodingSpec, ImageCodecKind, ImageResolutionType, Tier};
use codec::ImageCodec;
use common::constants::{codec_ext, codec_folder, res_folder};
use image::{imageops::FilterType, DynamicImage, GenericImageView};
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

impl ConversionStats {
    fn new(total: usize) -> Self {
        Self {
            total,
            succeeded: 0,
            failed: 0,
        }
    }

    fn report_success(&mut self) {
        self.succeeded += 1;
        if self.succeeded % 50 == 0 || self.succeeded == self.total {
            println!("Progress: {}/{} ({:.1}%)",
                     self.succeeded, self.total,
                     (self.succeeded as f32 / self.total as f32) * 100.0
            );
        }
    }

    fn report_failure(&mut self) {
        self.failed += 1;
    }

    fn print_summary(&self) {
        println!("\n=================================================================");
        println!("Conversion Summary:");
        println!("  Total:     {}", self.total);
        println!("  Succeeded: {}", self.succeeded);
        println!("  Failed:    {}", self.failed);
        println!("=================================================================\n");
    }
}

fn resize_image(img: &DynamicImage, target_res: ImageResolutionType) -> DynamicImage {
    match target_res {
        ImageResolutionType::FHD => {
            img.clone()
        }
        ImageResolutionType::HD => {
            img.resize_exact(1280, 720, FilterType::Lanczos3)
        }
        ImageResolutionType::Letterbox => {
            letterbox_resize(img, 640, 640)
        }
    }
}

fn letterbox_resize(img: &DynamicImage, target_w: u32, target_h: u32) -> DynamicImage {
    let (orig_w, orig_h) = img.dimensions();

    let scale = (target_w as f32 / orig_w as f32).min(target_h as f32 / orig_h as f32);
    let new_w = (orig_w as f32 * scale) as u32;
    let new_h = (orig_h as f32 * scale) as u32;

    let resized = img.resize_exact(new_w, new_h, FilterType::Lanczos3);

    let mut canvas = DynamicImage::new_rgb8(target_w, target_h);

    let offset_x = (target_w - new_w) / 2;
    let offset_y = (target_h - new_h) / 2;

    image::imageops::overlay(&mut canvas, &resized, offset_x as i64, offset_y as i64);

    canvas
}

fn process_single_image(
    task: &ConversionTask,
    resolution: ImageResolutionType,
    codec: ImageCodecKind,
    tier: Tier,
) -> Result<(), Box<dyn Error>> {
    let img = image::open(&task.input_path)
        .with_context(|| format!("Failed to open image: {:?}", task.input_path))?;

    let resized = resize_image(&img, resolution);

    let mut temp_png = Vec::new();
    {
        use std::io::Cursor;
        use image::ImageFormat;
        resized.write_to(&mut Cursor::new(&mut temp_png), ImageFormat::Png)
            .context("Failed to encode to temporary PNG")?;
    }

    let spec = EncodingSpec {
        codec,
        tier,
        resolution,
    };

    let frame = ImageCodec::compress_from_bytes(&temp_png, spec)
        .context("Failed to compress with ImageCodec")?;

    // Build output path: {output_base}/{resolution}/{codec}/{frame_name}_{tier}.{ext}
    let res_dir = task.output_base.join(res_folder(resolution));
    let codec_dir = res_dir.join(codec_folder(codec));
    fs::create_dir_all(&codec_dir)?;

    let tier_suffix = format!("{:?}", tier);
    let output_file = codec_dir.join(format!("{}_{}.{}",
                                             task.frame_number,
                                             tier_suffix,
                                             codec_ext(codec)
    ));

    fs::write(&output_file, frame.frame_data)
        .with_context(|| format!("Failed to write output: {:?}", output_file))?;

    Ok(())
}

fn discover_input_images(input_dir: &Path) -> Result<Vec<ConversionTask>, Box<dyn Error>> {
    let mut tasks = Vec::new();

    for entry in fs::read_dir(input_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            if let Some(ext) = path.extension() {
                if ext == "jpg" || ext == "jpeg" || ext == "JPG" || ext == "JPEG" {
                    if let Some(stem) = path.file_stem() {
                        let frame_number = stem.to_string_lossy().to_string();

                        tasks.push(ConversionTask {
                            input_path: path.clone(),
                            output_base: PathBuf::from("roadside-unit/sample"),
                            frame_number,
                        });
                    }
                }
            }
        }
    }

    tasks.sort_by(|a, b| a.frame_number.cmp(&b.frame_number));
    Ok(tasks)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <input_directory>", args[0]);
        eprintln!("\nExample:");
        eprintln!("  {} ./source_images/", args[0]);
        eprintln!("\nThis will process all JPG images in the input directory and generate:");
        eprintln!("  - 3 resolutions: FHD (1920x1080), HD (1280x720), LETTERBOX (640x640)");
        eprintln!("  - 3 quality tiers: T1 (high), T2 (balanced), T3 (fast)");
        eprintln!("  - 4 codecs: JPG, PNG, WebP Lossy, WebP Lossless");
        eprintln!("\nOutput: pi-sender/sample/{{resolution}}/{{codec}}/{{filename}}.{{ext}}");
        std::process::exit(1);
    }

    let input_dir = PathBuf::from(&args[1]);

    if !input_dir.exists() || !input_dir.is_dir() {
        eprintln!("Error: Input directory does not exist: {:?}", input_dir);
        std::process::exit(1);
    }

    println!("=================================================================");
    println!("Image Preprocessor for Rust Traffic Watch");
    println!("=================================================================");
    println!("Input directory: {:?}", input_dir);
    println!("Output directory: pi-sender/sample/");
    println!();

    println!("Discovering input images...");
    let tasks = discover_input_images(&input_dir)?;

    if tasks.is_empty() {
        eprintln!("Error: No JPG images found in input directory");
        std::process::exit(1);
    }

    println!("Found {} images to process", tasks.len());

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

    println!("\nConfiguration:");
    println!("  Resolutions: {} (FHD, HD, Letterbox)", resolutions.len());
    println!("  Codecs: {} (JPG, PNG, WebP Lossy, WebP Lossless)", codecs.len());
    println!("  Quality Tiers: {} (T1, T2, T3)", tiers.len());
    println!("  Total combinations per image: {}", total_combinations);
    println!("  Total conversions: {}", total_conversions);
    println!();

    fs::create_dir_all("roadside-unit/sample")?;

    let stats = Arc::new(Mutex::new(ConversionStats::new(total_conversions)));

    println!("Starting parallel batch conversion...\n");

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
            Ok(_) => {
                let mut stats = stats.lock().unwrap();
                stats.report_success();
            }
            Err(e) => {
                eprintln!("  Failed to process {:?}: {}", task.input_path, e);
                let mut stats = stats.lock().unwrap();
                stats.report_failure();
            }
        }
    });

    let stats = stats.lock().unwrap();
    stats.print_summary();

    if stats.failed > 0 {
        println!("Some conversions failed.");
    } else {
        println!("All conversions completed successfully!");
    }

    Ok(())
}