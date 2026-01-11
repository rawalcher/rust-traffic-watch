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
                duration: 60,
                rsu_count: 1,
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
                let experiment_id = format!("{}_{}_{}fps_suite", mode, model, fps);

                info!("[{}/{}] Running: {}", current, total, experiment_id);

                let config = ExperimentConfig::new(
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

                match harness.run_controller_with_retry(config, 3).await {
                    Ok(()) => info!("Done: {}", experiment_id),
                    Err(e) => error!("Failed {}: {}", experiment_id, e),
                }

                if current < total {
                    sleep(Duration::from_secs(5)).await;
                }
            }
        }
    }
    Ok(())
}
