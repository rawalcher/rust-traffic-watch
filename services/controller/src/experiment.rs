// services/controller/src/experiment.rs

use clap::{Args, Parser, Subcommand};
use std::error::Error;
use tokio::time::{sleep, Duration, Instant};
use tracing::{error, info};

use protocol::config::{
    DEFAULT_DURATION_SECONDS, DEFAULT_MODEL, DEFAULT_RSU_COUNT, DEFAULT_SEND_FPS,
};
use protocol::types::{
    EncodingSpec, ExperimentConfig, ExperimentMode, ImageCodecKind, ImageResolutionType, Tier,
};

use crate::service::ControllerHarness;

#[derive(Parser, Debug, Clone)]
#[command(author, version, about = "Experiment Controller CLI")]
pub struct Cli {
    #[command(subcommand)]
    pub command: RunMode,

    #[arg(long, global = true, conflicts_with = "remote_only")]
    pub local_only: bool,

    #[arg(long, global = true, conflicts_with = "local_only")]
    pub remote_only: bool,
}

#[derive(Subcommand, Debug, Clone)]
pub enum RunMode {
    /// Run a single specific model and FPS configuration
    Single(SingleArgs),

    /// Predefined fast test
    Quick,

    /// Full comprehensive suite: all models, all codecs, all tiers, all resolutions, 3 RSUs
    Full,

    /// Custom comprehensive test with specified parameters (all codecs, tiers, resolutions)
    Custom(CustomArgs),
}

#[derive(Args, Debug, Clone)]
pub struct SingleArgs {
    #[arg(long, default_value = DEFAULT_MODEL)]
    pub model: String,

    #[arg(long, default_value_t = DEFAULT_SEND_FPS)]
    pub fps: u64,

    #[arg(long, default_value_t = DEFAULT_RSU_COUNT)]
    pub rsu_count: u8,

    #[arg(long, default_value_t = DEFAULT_DURATION_SECONDS)]
    pub duration: u64,
}

#[derive(Args, Debug, Clone)]
pub struct CustomArgs {
    #[arg(long, default_value = "yolov5n")]
    pub model: String,

    #[arg(long, default_value_t = 15)]
    pub fps: u64,

    #[arg(long, default_value_t = 3)]
    pub rsu_count: u8,

    #[arg(long, default_value_t = 60)]
    pub duration: u64,
}

struct SuiteDefinition {
    name: String,
    models: Vec<String>,
    fps_values: Vec<u64>,
    codecs: Vec<ImageCodecKind>,
    tiers: Vec<Tier>,
    resolutions: Vec<ImageResolutionType>,
    modes: Vec<ExperimentMode>,
    duration: u64,
    rsu_count: u8,
}

pub async fn execute(
    cli: Cli,
    harness: &ControllerHarness,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let modes = match (cli.local_only, cli.remote_only) {
        (true, _) => vec![ExperimentMode::Local],
        (_, true) => vec![ExperimentMode::Offload],
        _ => vec![ExperimentMode::Local, ExperimentMode::Offload],
    };

    let def = match cli.command {
        RunMode::Single(a) => SuiteDefinition {
            name: "single".into(),
            models: vec![a.model],
            fps_values: vec![a.fps],
            codecs: vec![ImageCodecKind::JpgLossy],
            tiers: vec![Tier::T2],
            resolutions: vec![ImageResolutionType::FHD],
            modes,
            duration: a.duration,
            rsu_count: a.rsu_count,
        },
        RunMode::Quick => SuiteDefinition {
            name: "quick".into(),
            models: vec!["yolov5n".into()],
            fps_values: vec![1, 10],
            codecs: vec![ImageCodecKind::JpgLossy],
            tiers: vec![Tier::T2],
            resolutions: vec![ImageResolutionType::FHD],
            modes,
            duration: 10,
            rsu_count: 2,
        },
        RunMode::Full => SuiteDefinition {
            name: "full".into(),
            models: vec!["yolov5n".into(), "yolov5s".into(), "yolov5m".into()],
            fps_values: vec![1, 5, 10, 15],
            codecs: vec![
                ImageCodecKind::JpgLossy,
                ImageCodecKind::WebpLossy,
                ImageCodecKind::PngLossless,
            ],
            tiers: vec![Tier::T1, Tier::T2, Tier::T3],
            resolutions: vec![
                ImageResolutionType::FHD,
                ImageResolutionType::HD,
                ImageResolutionType::Letterbox,
            ],
            modes,
            duration: 30,
            rsu_count: 3,
        },
        RunMode::Custom(a) => SuiteDefinition {
            name: "custom".into(),
            models: vec![a.model],
            fps_values: vec![a.fps],
            codecs: vec![
                ImageCodecKind::JpgLossy,
                ImageCodecKind::WebpLossy,
                ImageCodecKind::WebpLossless,
                ImageCodecKind::PngLossless,
            ],
            tiers: vec![Tier::T1, Tier::T2, Tier::T3],
            resolutions: vec![
                ImageResolutionType::FHD,
                ImageResolutionType::HD,
                ImageResolutionType::Letterbox,
            ],
            modes,
            duration: a.duration,
            rsu_count: a.rsu_count,
        },
    };

    run_engine(def, harness).await
}

async fn run_engine(
    def: SuiteDefinition,
    harness: &ControllerHarness,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let total = def.models.len()
        * def.fps_values.len()
        * def.codecs.len()
        * def.tiers.len()
        * def.resolutions.len()
        * def.modes.len();

    info!("Starting {} suite: {} experiments total", def.name, total);
    let start_time = Instant::now();
    let mut current = 0;

    for model in &def.models {
        for fps in &def.fps_values {
            for codec in &def.codecs {
                for tier in &def.tiers {
                    for res in &def.resolutions {
                        for mode in &def.modes {
                            current += 1;
                            let id = format!(
                                "{mode}_{model}_{codec:?}_{tier:?}_{res:?}_{fps}fps_{}",
                                def.name
                            );

                            log_status(current, total, &id, start_time);

                            let mut config = ExperimentConfig::new(
                                id.clone(),
                                mode.clone(),
                                def.rsu_count,
                                model.clone(),
                                EncodingSpec { codec: *codec, tier: *tier, resolution: *res },
                            );
                            config.fixed_fps = *fps;
                            config.duration_seconds = def.duration;

                            if let Err(e) = harness.run_controller_with_retry(config, 3).await {
                                error!("Experiment failed: {e}");
                            }
                            sleep(Duration::from_millis(100)).await;
                        }
                    }
                }
            }
        }
    }

    info!("Suite '{}' finished in {}s", def.name, start_time.elapsed().as_secs());
    Ok(())
}

#[allow(clippy::cast_precision_loss)]
fn log_status(current: usize, total: usize, id: &str, start: Instant) {
    let elapsed = start.elapsed().as_secs();
    let progress = (current as f64 / total as f64) * 100.0;

    info!("[{current}/{total}] {progress:.1}% | Elapsed: {elapsed}s | Running: {id}");
}
