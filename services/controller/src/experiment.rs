use std::error::Error;
use tokio::time::{sleep, Duration};
use tracing::{error, info};

use codec::types::ImageCodecKind::{JpgLossy, PngLossless, WebpLossless, WebpLossy};
use codec::types::ImageResolutionType::{Letterbox, FHD, HD};
use codec::types::{EncodingSpec, ImageCodecKind, ImageResolutionType, Tier};
use common::constants::{DEFAULT_DURATION_SECONDS, DEFAULT_MODEL, SEND_FPS};
use protocol::{ExperimentConfig, ExperimentMode};

use super::service::ControllerHarness;

fn parse_codec(s: &str) -> Option<ImageCodecKind> {
    match s.trim().to_ascii_lowercase().as_str() {
        "jpg" | "jpglossy" => Some(JpgLossy),
        "png" | "pnglossless" => Some(PngLossless),
        "webplossy" => Some(WebpLossy),
        "webplossless" => Some(WebpLossless),
        _ => None,
    }
}

fn parse_resolution(s: &str) -> Option<ImageResolutionType> {
    match s.trim().to_ascii_lowercase().as_str() {
        "fhd" | "1080p" => Some(FHD),
        "hd" | "720p" => Some(HD),
        "letterbox" | "lb" => Some(Letterbox),
        _ => None,
    }
}

fn parse_tier(s: &str) -> Option<Tier> {
    match s.trim().to_ascii_lowercase().as_str() {
        "t1" => Some(Tier::T1),
        "t2" => Some(Tier::T2),
        "t3" => Some(Tier::T3),
        _ => None,
    }
}

#[derive(Debug)]
pub struct TestConfig {
    pub models: Vec<String>,
    pub fps_values: Vec<u64>,
    pub modes: Vec<ExperimentMode>,
    pub duration_seconds: u64,
    pub codecs: Vec<ImageCodecKind>,
    pub tiers: Vec<Tier>,
    pub resolutions: Vec<ImageResolutionType>,
    pub num_roadside_units: u32,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            models: vec!["yolov5n".into(), "yolov5s".into(), "yolov5m".into()],
            fps_values: vec![1, 5, 10],
            modes: vec![ExperimentMode::Local, ExperimentMode::Offload],
            duration_seconds: DEFAULT_DURATION_SECONDS,
            codecs: vec![JpgLossy, WebpLossy, PngLossless, WebpLossless],
            tiers: vec![Tier::T1, Tier::T2, Tier::T3],
            resolutions: vec![FHD, HD, Letterbox],
            num_roadside_units: 1,
        }
    }
}

impl TestConfig {
    pub fn parse_args(args: &[String]) -> Self {
        let mut config = Self::default();

        for arg in args {
            match arg.as_str() {
                a if a.starts_with("--models=") => {
                    config.models =
                        a.trim_start_matches("--models=").split(',').map(String::from).collect();
                }
                a if a.starts_with("--fps=") => {
                    config.fps_values = a
                        .trim_start_matches("--fps=")
                        .split(',')
                        .filter_map(|s| s.parse::<u64>().ok())
                        .collect();
                }
                a if a.starts_with("--duration=") => {
                    if let Ok(d) = a.trim_start_matches("--duration=").parse::<u64>() {
                        config.duration_seconds = d;
                    }
                }
                a if a.starts_with("--codecs=") => {
                    config.codecs = a
                        .trim_start_matches("--codecs=")
                        .split(',')
                        .filter_map(parse_codec)
                        .collect();
                }
                a if a.starts_with("--tiers=") => {
                    config.tiers = a
                        .trim_start_matches("--tiers=")
                        .split(',')
                        .filter_map(parse_tier)
                        .collect();
                }
                a if a.starts_with("--resolutions=") => {
                    config.resolutions = a
                        .trim_start_matches("--resolutions=")
                        .split(',')
                        .filter_map(parse_resolution)
                        .collect();
                }
                a if a.starts_with("--num-roadside-units=") => {
                    if let Ok(n) = a.trim_start_matches("--num-roadside-units=").parse::<u32>() {
                        config.num_roadside_units = n;
                    }
                }
                "--local-only" => config.modes = vec![ExperimentMode::Local],
                "--remote-only" => config.modes = vec![ExperimentMode::Offload],

                "--quick" => {
                    config.duration_seconds = 60;
                    config.models = vec!["yolov5n".into()];
                    config.fps_values = vec![1, 10];
                    config.codecs = vec![JpgLossy];
                    config.tiers = vec![Tier::T2];
                    config.resolutions = vec![FHD];
                    config.num_roadside_units = 1;
                }
                _ => {}
            }
        }

        config.num_roadside_units = config.num_roadside_units.max(1);

        config
    }

    pub fn total_experiments(&self) -> usize {
        self.models.len()
            * self.fps_values.len()
            * self.modes.len()
            * self.codecs.len()
            * self.tiers.len()
            * self.resolutions.len()
    }

    pub fn estimated_time(&self) -> Duration {
        Duration::from_secs((self.duration_seconds + 10) * self.total_experiments() as u64)
    }
}

pub async fn run_single_experiment(
    args: &[String],
    harness: &ControllerHarness,
    num_roadside_units: u32,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let model_name = args.iter().find(|arg| arg.starts_with("--model=")).map_or_else(
        || DEFAULT_MODEL.to_string(),
        |arg| arg.trim_start_matches("--model=").to_string(),
    );

    let fps = args
        .iter()
        .find(|arg| arg.starts_with("--fps="))
        .and_then(|arg| arg.trim_start_matches("--fps=").parse::<u64>().ok())
        .unwrap_or(SEND_FPS);

    let modes = match args.iter().find(|a| *a == "--local" || *a == "--remote") {
        Some(flag) if flag == "--local" => vec![ExperimentMode::Local],
        Some(flag) if flag == "--remote" => vec![ExperimentMode::Offload],
        _ => vec![ExperimentMode::Local, ExperimentMode::Offload],
    };
    let codec = args
        .iter()
        .find(|a| a.starts_with("--codec="))
        .and_then(|a| parse_codec(a.trim_start_matches("--codec=")))
        .unwrap_or(JpgLossy);

    let resolution = args
        .iter()
        .find(|a| a.starts_with("--resolution="))
        .and_then(|a| parse_resolution(a.trim_start_matches("--resolution=")))
        .unwrap_or(FHD);

    let tier = args
        .iter()
        .find(|a| a.starts_with("--tier="))
        .and_then(|a| parse_tier(a.trim_start_matches("--tier=")))
        .unwrap_or(Tier::T2);

    let encoding = EncodingSpec { codec, tier, resolution };

    for mode in modes {
        let experiment_id = format!("{mode}_{model_name}_{fps}fps_{codec}_{tier}_{resolution}");

        let mut config = ExperimentConfig::new(
            experiment_id.clone(),
            mode,
            model_name.clone(),
            encoding.clone(),
        );
        config.fixed_fps = fps;

        info!("Starting single experiment: {}", experiment_id);
        harness.run_controller(config, num_roadside_units).await?;
        sleep(Duration::from_secs(2)).await;
    }
    Ok(())
}

pub async fn run_test_suite(
    test_config: TestConfig,
    harness: &ControllerHarness,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    info!("=================================================");
    info!("Starting automated test suite");
    info!("Models: {:?}", test_config.models);
    info!("FPS values: {:?}", test_config.fps_values);
    info!("Modes: {:?}", test_config.modes);
    info!("Codecs: {:?}", test_config.codecs);
    info!("Tiers: {:?}", test_config.tiers);
    info!("Resolutions: {:?}", test_config.resolutions);
    info!("Number of RSUs (N): {}", test_config.num_roadside_units);
    info!("Duration per test: {} seconds", test_config.duration_seconds);
    info!("Total experiments: {}", test_config.total_experiments());
    info!("Estimated total time: {:?}", test_config.estimated_time());
    info!("=================================================");

    let mut current = 0;
    let total = test_config.total_experiments();
    let num_roadside_units = test_config.num_roadside_units;

    for model in &test_config.models {
        for fps in &test_config.fps_values {
            for mode in &test_config.modes {
                for codec in &test_config.codecs {
                    for tier in &test_config.tiers {
                        for resolution in &test_config.resolutions {
                            current += 1;

                            let experiment_id =
                                format!("{mode}_{model}_{fps}fps_{codec}_{tier}_{resolution}",);

                            info!("Experiment {}/{}: {}", current, total, experiment_id);

                            let mut config = ExperimentConfig::new(
                                experiment_id.clone(),
                                mode.clone(),
                                model.clone(),
                                EncodingSpec {
                                    codec: *codec,
                                    tier: *tier,
                                    resolution: *resolution,
                                },
                            );
                            config.fixed_fps = *fps;
                            config.duration_seconds = test_config.duration_seconds;

                            match harness
                                .run_controller_with_retry(config, num_roadside_units, 3)
                                .await
                            {
                                Ok(()) => info!("Experiment {} complete", experiment_id),
                                Err(e) => error!("Experiment {} failed: {}", experiment_id, e),
                            }

                            if current < total {
                                info!("Waiting 10 seconds before next experiment...");
                                sleep(Duration::from_secs(10)).await;
                            }
                        }
                    }
                }
            }
        }
    }

    info!("=================================================");
    info!("Test suite complete");
    info!("=================================================");
    Ok(())
}
