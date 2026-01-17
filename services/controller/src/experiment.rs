use clap::{Args, Parser, Subcommand};
use std::error::Error;
use tokio::time::{sleep, Duration};
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

    /// Run a full automated test suite with custom parameters
    Suite(SuiteArgs),

    /// Predefined fast test
    Quick,

    /// Advanced high-stress suite: all models, all tiers, long duration
    Advanced,

    /// Full comprehensive suite: all models, all codecs, all tiers, all resolutions, 3 RSUs
    Full,
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
pub struct SuiteArgs {
    #[arg(long, value_delimiter = ',', default_values = &["yolov5n", "yolov5s", "yolov5m"])]
    pub models: Vec<String>,

    #[arg(long, value_delimiter = ',', default_values = &["1", "5", "10"])]
    pub fps_values: Vec<u64>,

    #[arg(long, default_value_t = DEFAULT_DURATION_SECONDS)]
    pub duration: u64,

    #[arg(long, default_value_t = 1)]
    pub rsu_count: u8,
}

pub async fn execute(
    cli: Cli,
    harness: &ControllerHarness,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let modes = if cli.local_only {
        vec![ExperimentMode::Local]
    } else if cli.remote_only {
        vec![ExperimentMode::Offload]
    } else {
        vec![ExperimentMode::Local, ExperimentMode::Offload]
    };

    match cli.command {
        RunMode::Single(args) => run_single_experiment(args, modes, harness).await,
        RunMode::Suite(args) => run_test_suite(args, modes, harness).await,
        RunMode::Quick => {
            let quick_args = SuiteArgs {
                models: vec!["yolov5n".into()],
                fps_values: vec![1, 10],
                duration: 30,
                rsu_count: 2,
            };
            run_test_suite(quick_args, modes, harness).await
        }
        RunMode::Advanced => {
            let advanced_args = SuiteArgs {
                models: vec![
                    "yolov5n".into(),
                    "yolov5s".into(),
                    "yolov5m".into(),
                    "yolov5l".into(),
                ],
                fps_values: vec![1, 5, 10, 20, 30],
                duration: 600,
                rsu_count: 2,
            };
            run_test_suite(advanced_args, modes, harness).await
        }
        RunMode::Full => run_full_comprehensive_suite(modes, harness).await,
    }
}

async fn run_single_experiment(
    args: SingleArgs,
    modes: Vec<ExperimentMode>,
    harness: &ControllerHarness,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let encoding = EncodingSpec {
        codec: ImageCodecKind::JpgLossy,
        tier: Tier::T2,
        resolution: ImageResolutionType::FHD,
    };

    for mode in modes {
        let experiment_id = format!("{}_{}_{}fps_single", mode, args.model, args.fps);

        let mut config = ExperimentConfig::new(
            experiment_id.clone(),
            mode,
            args.rsu_count,
            args.model.clone(),
            encoding.clone(),
        );
        config.fixed_fps = args.fps;
        config.duration_seconds = args.duration;

        info!("Starting single experiment: {}", experiment_id);
        harness.run_controller(config).await?;
        sleep(Duration::from_secs(2)).await;
    }
    Ok(())
}

async fn run_test_suite(
    args: SuiteArgs,
    modes: Vec<ExperimentMode>,
    harness: &ControllerHarness,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let total = args.models.len() * args.fps_values.len() * modes.len();
    let mut current = 0;

    info!("Starting suite: {} total experiments", total);

    for model in &args.models {
        for fps in &args.fps_values {
            for mode in &modes {
                current += 1;
                let experiment_id = format!("{mode}_{model}_{fps}fps_suite");

                info!("[{}/{}] Running: {}", current, total, experiment_id);

                let mut config = ExperimentConfig::new(
                    experiment_id.clone(),
                    mode.clone(),
                    args.rsu_count,
                    model.clone(),
                    EncodingSpec {
                        codec: ImageCodecKind::JpgLossy,
                        tier: Tier::T2,
                        resolution: ImageResolutionType::FHD,
                    },
                );
                config.fixed_fps = *fps;
                config.duration_seconds = args.duration;

                match harness.run_controller_with_retry(config, 3).await {
                    Ok(()) => info!("Done: {}", experiment_id),
                    Err(e) => error!("Failed {}: {}", experiment_id, e),
                }
            }
        }
    }
    Ok(())
}

async fn run_full_comprehensive_suite(
    modes: Vec<ExperimentMode>,
    harness: &ControllerHarness,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let models = vec!["yolov5n".to_string(), "yolov5s".to_string(), "yolov5m".to_string()];

    let fps_values = vec![1, 5, 10, 15];

    let codecs = vec![
        ImageCodecKind::JpgLossy,
        ImageCodecKind::PngLossless,
        ImageCodecKind::WebpLossy,
        ImageCodecKind::WebpLossless,
    ];

    let tiers = vec![Tier::T1, Tier::T2, Tier::T3];

    let resolutions =
        vec![ImageResolutionType::FHD, ImageResolutionType::HD, ImageResolutionType::Letterbox];

    let duration = 30;
    let rsu_count = 3;

    let total = models.len()
        * fps_values.len()
        * codecs.len()
        * tiers.len()
        * resolutions.len()
        * modes.len();

    info!("========================================");
    info!("FULL COMPREHENSIVE SUITE (SMOKE TEST)");
    info!("========================================");
    info!("Models: {}", models.len());
    info!("FPS values: {}", fps_values.len());
    info!("Codecs: {}", codecs.len());
    info!("Tiers: {}", tiers.len());
    info!("Resolutions: {}", resolutions.len());
    info!("Modes: {}", modes.len());
    info!("RSU count: {}", rsu_count);
    info!("Duration per experiment: {}s (smoke test)", duration);
    info!("----------------------------------------");
    info!("TOTAL EXPERIMENTS: {}", total);
    info!(
        "ESTIMATED TIME: {:.1} hours ({:.0} minutes)",
        (total as f64 * duration as f64) / 3600.0,
        (total as f64 * duration as f64) / 60.0
    );
    info!("========================================");

    let mut current = 0;
    let start_time = std::time::Instant::now();

    for model in &models {
        for fps in &fps_values {
            for codec in &codecs {
                for tier in &tiers {
                    for resolution in &resolutions {
                        for mode in &modes {
                            current += 1;

                            let experiment_id = format!(
                                "{}_{}_{:?}_{:?}_{:?}_{}fps_{}rsu_full",
                                mode, model, codec, tier, resolution, fps, rsu_count
                            );

                            let elapsed = start_time.elapsed().as_secs();
                            let rate = if current > 1 {
                                elapsed as f64 / (current - 1) as f64
                            } else {
                                0.0
                            };
                            let remaining_experiments = total - current;
                            let eta_seconds = (remaining_experiments as f64 * rate) as u64;

                            info!("========================================");
                            info!("[{}/{}] Running: {}", current, total, experiment_id);
                            info!(
                                "Progress: {:.1}% | Elapsed: {}h {}m | ETA: {}h {}m",
                                (current as f64 / total as f64) * 100.0,
                                elapsed / 3600,
                                (elapsed % 3600) / 60,
                                eta_seconds / 3600,
                                (eta_seconds % 3600) / 60
                            );
                            info!("========================================");

                            let encoding_spec = EncodingSpec {
                                codec: *codec,
                                tier: *tier,
                                resolution: *resolution,
                            };

                            let mut config = ExperimentConfig::new(
                                experiment_id.clone(),
                                mode.clone(),
                                rsu_count,
                                model.clone(),
                                encoding_spec,
                            );
                            config.fixed_fps = *fps;
                            config.duration_seconds = duration;

                            match harness.run_controller_with_retry(config, 3).await {
                                Ok(()) => {
                                    info!("Completed: {}", experiment_id);
                                }
                                Err(e) => {
                                    error!("Failed {}: {}", experiment_id, e);
                                    error!("Continuing to next experiment...");
                                }
                            }

                            if current < total {
                                sleep(Duration::from_millis(100)).await;
                            }
                        }
                    }
                }
            }
        }
    }

    let total_time = start_time.elapsed();
    info!("========================================");
    info!("FULL SUITE SMOKE TEST COMPLETE");
    info!("========================================");
    info!("Total experiments: {}", total);
    info!(
        "Total time: {}h {}m {}s",
        total_time.as_secs() / 3600,
        (total_time.as_secs() % 3600) / 60,
        total_time.as_secs() % 60
    );
    info!("Average per experiment: {:.1}s", total_time.as_secs() as f64 / total as f64);
    info!("========================================");
    info!("NOTE: This was a 30-second smoke test.");
    info!("Review results to select configurations for full production runs.");
    info!("========================================");

    Ok(())
}
