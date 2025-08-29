// src/main.rs

use std::{env, error::Error};

use csv::Writer;
use tokio::sync::{mpsc, watch, Mutex};
use tokio::time::{sleep, timeout, Duration, Instant};
use tracing::{info, warn};

use std::sync::Arc;

use shared::connection::{
    get_device_sender, start_controller_listener, wait_for_device_readiness, wait_for_devices, Role,
};
use shared::constants::*;
use shared::current_timestamp_micros;
use shared::types::*;
use shared::DeviceId::{self, Pi};

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
                warn!("controller listener exited with error: {:?}", e);
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
            info!("Result collection task finished with {} results", count);
        });

        let controller_listener_task_note = "listener runs globally; not per-run";

        let required_devices = match config.mode {
            ExperimentMode::LocalOnly => vec![Pi],
            ExperimentMode::Offload => vec![Pi, DeviceId::Jetson],
        };

        wait_for_devices(&required_devices).await;

        let msg = Message::Control(ControlMessage::StartExperiment {
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

        while start.elapsed().as_secs() < config.duration_seconds {
            if let Some(sender) = get_device_sender(&Pi).await {
                let mut timing = TimingMetadata::default();
                timing.sequence_id = sequence_id;
                timing.controller_sent_pulse = Some(current_timestamp_micros());
                timing.frame_number = frame_number;

                sender.send(Message::Pulse(timing))?;
                info!("Sent pulse {} to Pi", sequence_id);

                sequence_id += 1;
                frame_number = (frame_number + 30 - 1) % MAX_FRAME_SEQUENCE + 1;
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
        info!(
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
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            models: vec![
                "yolov5n".to_string(),
                "yolov5s".to_string(),
                "yolov5m".to_string(),
            ],
            fps_values: vec![1.0, 5.0, 10.0, 15.0],
            modes: vec![ExperimentMode::LocalOnly, ExperimentMode::Offload],
            duration_seconds: DEFAULT_DURATION_SECONDS,
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
                "--local-only" => config.modes = vec![ExperimentMode::LocalOnly],
                "--remote-only" => config.modes = vec![ExperimentMode::Offload],
                "--quick" => {
                    config.duration_seconds = 60;
                    config.models = vec!["yolov5n".to_string()];
                    config.fps_values = vec![1.0, 10.0];
                }
                _ => {}
            }
        }

        config
    }

    fn total_experiments(&self) -> usize {
        self.models.len() * self.fps_values.len() * self.modes.len()
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

    for mode in modes {
        let experiment_id = format!("{:?}_{}_{}fps", mode, model_name, fps);
        let mut config = ExperimentConfig::new(experiment_id.clone(), mode, model_name.clone());
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
                current += 1;

                info!("=================================================");
                info!(
                    "Experiment {}/{}: Model={}, FPS={}, Mode={:?}",
                    current, total, model, fps, mode
                );
                info!("=================================================");

                let experiment_id = format!("{:?}_{}_{}fps", mode, model, fps);
                let mut config =
                    ExperimentConfig::new(experiment_id.clone(), mode.clone(), model.clone());
                config.fixed_fps = *fps;
                config.duration_seconds = test_config.duration_seconds;

                match harness.run_controller(config).await {
                    Ok(_) => info!("✓ Experiment {} completed", experiment_id),
                    Err(e) => {
                        eprintln!("✗ Experiment {} failed: {}", experiment_id, e);
                    }
                }

                if current < total {
                    info!("Waiting 15 seconds before next experiment...");
                    sleep(Duration::from_secs(15)).await; // extra cleanup time
                }
            }
        }
    }

    info!("=================================================");
    info!("All experiments completed! Results in logs/");
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

    info!("Saving {} results to {}", results.len(), filename);
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
