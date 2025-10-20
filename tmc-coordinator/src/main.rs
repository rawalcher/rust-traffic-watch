use std::{env, error::Error};

use csv::Writer;
use tokio::sync::{mpsc, watch, Mutex};
use tokio::time::{sleep, timeout, Duration, Instant};
use tracing::{debug, error, info, warn};

use std::sync::Arc;

use shared::connection::{
    get_device_sender, start_controller_listener, wait_for_device_readiness, wait_for_devices, Role,
};
use shared::constants::*;
use shared::current_timestamp_micros;
use shared::types::*;
use shared::DeviceId::{self, Pi};
use shared::ImageCodecKind::{JpgLossy, PngLossless, WebpLossless, WebpLossy};
use shared::ImageResolutionType::{FHD, HD, LETTERBOX};

struct ControllerHarness {
    active_sink_tx: watch::Sender<Option<mpsc::UnboundedSender<InferenceMessage>>>,
    _forwarder_task: tokio::task::JoinHandle<()>,
    _listener_task: tokio::task::JoinHandle<()>,
}

impl ControllerHarness {
    async fn new() -> Self {
        let (raw_tx, mut raw_rx) = mpsc::unbounded_channel::<InferenceMessage>();

        let listener_task = tokio::spawn(async move {
            if let Err(e) = start_controller_listener(Role::Controller {
                result_handler: raw_tx,
            })
            .await
            {
                warn!("tmc-coordinator listener exited with error: {:?}", e);
            }
        });

        let (active_sink_tx, active_sink_rx) =
            watch::channel::<Option<mpsc::UnboundedSender<InferenceMessage>>>(None);

        let forwarder_task = tokio::spawn(async move {
            let sink_rx = active_sink_rx;
            while let Some(msg) = raw_rx.recv().await {
                if let Some(sink) = sink_rx.borrow().clone() {
                    let _ = sink.send(msg);
                }
            }
        });

        Self {
            active_sink_tx,
            _forwarder_task: forwarder_task,
            _listener_task: listener_task,
        }
    }

    async fn run_controller_with_retry(
        &self,
        config: ExperimentConfig,
        max_retries: u32,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let mut attempt = 0;

        loop {
            attempt += 1;

            match self.run_controller(config.clone()).await {
                Ok(()) => {
                    return Ok(());
                }
                Err(e) if attempt <= max_retries => {
                    if e.to_string().contains("disconnected")
                        || e.to_string().contains("writer channel closed")
                    {
                        warn!(
                            "Device disconnected. Restarting experiment in 30s (attempt {}/{})",
                            attempt + 1,
                            max_retries + 1
                        );
                        sleep(Duration::from_secs(30)).await;
                        continue;
                    } else {
                        return Err(e);
                    }
                }
                Err(e) => {
                    return Err(e);
                }
            }
        }
    }

    async fn run_controller(
        &self,
        config: ExperimentConfig,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        info!("Starting experiment: {}", config.experiment_id);

        shared::connection::clear_ready_devices().await;

        let (inference_tx, mut inference_rx) = mpsc::unbounded_channel::<InferenceMessage>();
        let results: Arc<Mutex<Vec<InferenceMessage>>> = Arc::new(Mutex::new(Vec::new()));
        let results_clone = Arc::clone(&results);

        let _ = self.active_sink_tx.send(Some(inference_tx.clone()));

        let (stop_tx, mut stop_rx) = watch::channel(false);
        let result_collection_task = tokio::spawn(async move {
            let mut count = 0usize;
            loop {
                tokio::select! {
                    _ = stop_rx.changed() => {
                        if *stop_rx.borrow() { break; }
                    }
                    maybe = inference_rx.recv() => {
                        match maybe {
                            Some(mut result) => {
                                result.timing.controller_received = Some(current_timestamp_micros());
                                count += 1;
                                info!("Controller received result {} for sequence_id: {}", count, result.sequence_id);
                                results_clone.lock().await.push(result);
                            }
                            None => break, // sender dropped
                        }
                    }
                }
            }
            debug!("Result collection task finished with {} results", count);
        });

        let controller_listener_task_note = "listener runs globally; not per-run";

        let required_devices = match config.mode {
            ExperimentMode::LocalOnly => vec![Pi],
            ExperimentMode::Offload => vec![Pi, DeviceId::Jetson],
        };

        wait_for_devices(&required_devices).await;

        let msg = Message::Control(ControlMessage::ConfigureExperiment {
            config: config.clone(),
        });
        for id in &required_devices {
            if let Some(sender) = get_device_sender(id).await {
                sender.send(msg.clone())?;
            }
        }

        wait_for_device_readiness(&required_devices).await;

        let begin_msg = Message::Control(ControlMessage::BeginExperiment);
        for id in &required_devices {
            if let Some(sender) = get_device_sender(id).await {
                sender.send(begin_msg.clone())?;
            }
        }

        let start = Instant::now();
        let mut sequence_id: u64 = 0;
        let mut frame_number: u64 = 1;
        let pulse_interval = Duration::from_millis((1000.0 / config.fixed_fps) as u64);
        let mut expected_results = 0u64;

        let frame_skip = (SOURCE_FPS / config.fixed_fps).round() as u64;
        let frame_skip = frame_skip.max(1);

        while start.elapsed().as_secs() < config.duration_seconds {
            if let Some(sender) = get_device_sender(&Pi).await {
                let mut timing = TimingMetadata::default();
                timing.sequence_id = sequence_id;
                timing.controller_sent_pulse = Some(current_timestamp_micros());
                timing.frame_number = frame_number;

                sender.send(Message::Pulse(timing))?;
                info!("Sent pulse {} to Pi", sequence_id);

                sequence_id += 1;
                frame_number = (frame_number + frame_skip - 1) % MAX_FRAME_SEQUENCE + 1;
                expected_results += 1;
            }
            sleep(pulse_interval).await;
        }

        info!(
            "Finished sending {} pulses. Waiting 5 seconds for results...",
            expected_results
        );
        sleep(Duration::from_secs(5)).await;

        let final_count_now = results.lock().await.len();
        info!(
            "Collected {} results out of {} pulses sent ({:.1}% success rate)",
            final_count_now,
            expected_results,
            (final_count_now as f32 / expected_results as f32) * 100.0
        );

        let shutdown = Message::Control(ControlMessage::Shutdown);
        for id in &required_devices {
            if let Some(sender) = get_device_sender(id).await {
                let _ = sender.send(shutdown.clone());
            }
        }

        let _ = stop_tx.send(true);

        let _ = self.active_sink_tx.send(None);

        drop(inference_tx);

        match timeout(Duration::from_secs(2), result_collection_task).await {
            Ok(joined) => {
                if let Err(e) = joined {
                    warn!("Result collection task error: {:?}", e);
                }
            }
            Err(_) => {
                warn!("Result collector did not stop in time; it will end soon.");
            }
        }

        info!("Shutdown complete ({})", controller_listener_task_note);

        let locked_results = results.lock().await;
        let final_count = locked_results.len();
        debug!(
            "Generating CSV with {} results for experiment {}",
            final_count, config.experiment_id
        );

        generate_analysis_csv(&locked_results, &config.experiment_id, &config)?;
        info!(
            "Experiment {} completed with {} results saved",
            config.experiment_id, final_count
        );
        Ok(())
    }
}

#[derive(Debug)]
struct TestConfig {
    models: Vec<String>,
    fps_values: Vec<f32>,
    modes: Vec<ExperimentMode>,
    duration_seconds: u64,
    codecs: Vec<ImageCodecKind>,
    tiers: Vec<Tier>,
    resolutions: Vec<ImageResolutionType>,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            models: vec!["yolov5n".into(), "yolov5s".into(), "yolov5m".into()],
            fps_values: vec![1.0, 5.0, 10.0],
            modes: vec![ExperimentMode::LocalOnly, ExperimentMode::Offload],
            duration_seconds: DEFAULT_DURATION_SECONDS,
            codecs: vec![JpgLossy, WebpLossy, PngLossless, WebpLossless],
            tiers: vec![Tier::T1, Tier::T2, Tier::T3],
            resolutions: vec![FHD, HD, LETTERBOX],
        }
    }
}

impl TestConfig {
    fn parse_args(args: &[String]) -> Self {
        let mut config = Self::default();

        for arg in args {
            match arg.as_str() {
                a if a.starts_with("--models=") => {
                    config.models = a
                        .trim_start_matches("--models=")
                        .split(',')
                        .map(String::from)
                        .collect();
                }
                a if a.starts_with("--fps=") => {
                    config.fps_values = a
                        .trim_start_matches("--fps=")
                        .split(',')
                        .filter_map(|s| s.parse::<f32>().ok())
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
                "--local-only" => config.modes = vec![ExperimentMode::LocalOnly],
                "--remote-only" => config.modes = vec![ExperimentMode::Offload],

                "--quick" => {
                    config.duration_seconds = 60;
                    config.models = vec!["yolov5n".into()];
                    config.fps_values = vec![1.0, 10.0];
                    config.codecs = vec![JpgLossy];
                    config.tiers = vec![Tier::T2];
                    config.resolutions = vec![FHD];
                }
                _ => {}
            }
        }

        config
    }

    fn total_experiments(&self) -> usize {
        self.models.len()
            * self.fps_values.len()
            * self.modes.len()
            * self.codecs.len()
            * self.tiers.len()
            * self.resolutions.len()
    }

    fn estimated_time(&self) -> Duration {
        Duration::from_secs((self.duration_seconds + 10) * self.total_experiments() as u64)
    }
}

async fn run_single_experiment(
    args: &[String],
    harness: &ControllerHarness,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let model_name = args
        .iter()
        .find(|arg| arg.starts_with("--model="))
        .map(|arg| arg.trim_start_matches("--model=").to_string())
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());

    let fps = args
        .iter()
        .find(|arg| arg.starts_with("--fps="))
        .and_then(|arg| arg.trim_start_matches("--fps=").parse::<f32>().ok())
        .unwrap_or(DEFAULT_SEND_FPS);

    let modes = match args.iter().find(|a| *a == "--local" || *a == "--remote") {
        Some(flag) if flag == "--local" => vec![ExperimentMode::LocalOnly],
        Some(flag) if flag == "--remote" => vec![ExperimentMode::Offload],
        _ => vec![ExperimentMode::LocalOnly, ExperimentMode::Offload],
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

    let encoding = EncodingSpec {
        codec,
        tier,
        resolution,
    };

    for mode in modes {
        let experiment_id = format!(
            "{:?}_{}_{}fps_{:?}_{:?}_{:?}",
            mode, model_name, fps, codec, tier, resolution
        );

        let mut config = ExperimentConfig::new(
            experiment_id.clone(),
            mode,
            model_name.clone(),
            encoding.clone(),
        );
        config.fixed_fps = fps;

        info!("Starting single experiment: {}", experiment_id);
        harness.run_controller(config).await?;
        sleep(Duration::from_secs(2)).await;
    }
    Ok(())
}

async fn run_test_suite(
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
    info!(
        "Duration per test: {} seconds",
        test_config.duration_seconds
    );
    info!("Total experiments: {}", test_config.total_experiments());
    info!("Estimated total time: {:?}", test_config.estimated_time());
    info!("=================================================");

    let mut current = 0;
    let total = test_config.total_experiments();

    for model in &test_config.models {
        for fps in &test_config.fps_values {
            for mode in &test_config.modes {
                for codec in &test_config.codecs {
                    for tier in &test_config.tiers {
                        for resolution in &test_config.resolutions {
                            current += 1;

                            let experiment_id = format!(
                                "{:?}_{}_{}fps_{:?}_{:?}_{:?}",
                                mode, model, fps, codec, tier, resolution
                            );

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

                            match harness.run_controller_with_retry(config, 3).await {
                                Ok(_) => info!("Experiment {} complete", experiment_id),
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

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    tracing_subscriber::fmt::init();
    let args: Vec<String> = env::args().collect();

    let harness = ControllerHarness::new().await;

    if args.iter().any(|a| a.starts_with("--model=")) {
        run_single_experiment(&args, &harness).await
    } else {
        let test_config = TestConfig::parse_args(&args);
        run_test_suite(test_config, &harness).await
    }
}

fn parse_codec(s: &str) -> Option<ImageCodecKind> {
    match s.trim().to_ascii_lowercase().as_str() {
        "jpg" | "jpglossy" => Some(JpgLossy),
        "webplossy" => Some(WebpLossy),
        "png" | "pnglossless" => Some(PngLossless),
        "webglossless" => Some(WebpLossless),
        _ => None,
    }
}

fn parse_resolution(s: &str) -> Option<ImageResolutionType> {
    match s.trim().to_ascii_lowercase().as_str() {
        "fhd" | "1080p" => Some(FHD),
        "hd" | "720p" => Some(HD),
        "letterbox" | "lb" => Some(LETTERBOX),
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
fn generate_analysis_csv(
    results: &[InferenceMessage],
    experiment_id: &str,
    config: &ExperimentConfig,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    if results.is_empty() {
        warn!("No results to save for experiment: {}", experiment_id);
        return Ok(());
    }

    std::fs::create_dir_all("logs")?;
    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let filename = format!("logs/experiment_{}_{}.csv", experiment_id, timestamp);

    debug!("Saving {} results to {}", results.len(), filename);
    let mut writer = Writer::from_path(&filename)?;

    writer.write_record(&[
        "sequence_id",
        "frame_number",
        "pi_hostname",
        "pi_capture_start",
        "pi_sent_to_jetson",
        "jetson_received",
        "jetson_sent_result",
        "controller_sent_pulse",
        "controller_received",
        "pi_overhead_us",
        "jetson_overhead_us",
        "network_latency_us",
        "total_latency_us",
        "inference_us",
        "frame_size_bytes",
        "detection_count",
        "image_width",
        "image_height",
        "model_name",
        "experiment_mode",
        "codec",
        "tier",
        "resolution",
    ])?;

    for r in results {
        let t = &r.timing;
        let i = &r.inference;

        let pi_overhead = calc_diff(t.pi_sent_to_jetson, t.pi_capture_start);
        let jetson_overhead = calc_diff(t.jetson_sent_result, t.jetson_received);
        let total_latency = calc_diff(t.controller_received, t.controller_sent_pulse);

        let network_latency = match config.mode {
            ExperimentMode::Offload => total_latency.saturating_sub(pi_overhead + jetson_overhead),
            ExperimentMode::LocalOnly => total_latency.saturating_sub(i.processing_time_us),
        };

        writer.write_record(&[
            t.sequence_id.to_string(),
            t.frame_number.to_string(),
            t.pi_hostname.clone(),
            opt_str(t.pi_capture_start),
            opt_str(t.pi_sent_to_jetson),
            opt_str(t.jetson_received),
            opt_str(t.jetson_sent_result),
            opt_str(t.controller_sent_pulse),
            opt_str(t.controller_received),
            pi_overhead.to_string(),
            jetson_overhead.to_string(),
            network_latency.to_string(),
            total_latency.to_string(),
            i.processing_time_us.to_string(),
            i.frame_size_bytes.to_string(),
            i.detection_count.to_string(),
            i.image_width.to_string(),
            i.image_height.to_string(),
            i.model_name.clone(),
            i.experiment_mode.clone(),
            format!("{:?}", config.encoding_spec.codec),
            format!("{:?}", config.encoding_spec.tier),
            format!("{:?}", config.encoding_spec.resolution),
        ])?;
    }

    writer.flush()?;
    info!(
        "Analysis saved to {} with {} records",
        filename,
        results.len()
    );
    Ok(())
}

fn opt_str<T: ToString>(opt: Option<T>) -> String {
    opt.map(|v| v.to_string()).unwrap_or_default()
}

fn calc_diff(a: Option<u64>, b: Option<u64>) -> u64 {
    match (a, b) {
        (Some(a), Some(b)) if a >= b => a - b,
        _ => 0,
    }
}
