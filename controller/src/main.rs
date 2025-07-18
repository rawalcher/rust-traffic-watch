use std::{env, error::Error};
use std::sync::{Arc, Mutex};
use tokio::time::{sleep, Duration, Instant};
use tracing::{info};

use csv::Writer;
use log::debug;
use tokio::sync::mpsc;
use shared::constants::*;
use shared::connection::{get_device_sender, start_controller_listener, wait_for_device_readiness, wait_for_devices, Role};
use shared::types::*;
use shared::current_timestamp_micros;
use shared::DeviceId::Pi;

pub async fn run_controller(config: ExperimentConfig) -> Result<(), Box<dyn Error + Send + Sync>> {
    info!("Starting experiment: {}", config.experiment_id);

    let (inference_tx, mut inference_rx) = mpsc::unbounded_channel::<InferenceMessage>();

    let results: Arc<Mutex<Vec<InferenceMessage>>> = Arc::new(Mutex::new(Vec::new()));
    let results_clone = Arc::clone(&results);

    tokio::spawn(async move {
        while let Some(mut result) = inference_rx.recv().await {
            result.timing.controller_received = Some(current_timestamp_micros());
            info!("Controller received result for sequence_id: {}", result.sequence_id);
            results_clone.lock().unwrap().push(result);
        }
    });

    tokio::spawn(start_controller_listener(Role::Controller {
        result_handler: inference_tx,
    }));
    
    if matches!(config.mode, ExperimentMode::Offload){
        wait_for_devices(&[DeviceId::Jetson, Pi]).await;
    }
    if matches!(config.mode, ExperimentMode::LocalOnly){
        wait_for_devices(&[Pi]).await;

    }
    
    let msg = Message::Control(ControlMessage::StartExperiment { config: config.clone() });
    for id in &[Pi, DeviceId::Jetson] {
        debug!("Preparing to send StartExperiment");
        if matches!(config.mode, ExperimentMode::LocalOnly) && *id == DeviceId::Jetson {
            continue;
        }
        if let Some(sender) = get_device_sender(id).await {
            sender.send(msg.clone())?;
        }
    }

    if matches!(config.mode, ExperimentMode::Offload){
        wait_for_device_readiness(&[DeviceId::Jetson, Pi]).await;
    }
    if matches!(config.mode, ExperimentMode::LocalOnly){
        wait_for_device_readiness(&[Pi]).await;
    }

    let begin_msg = Message::Control(ControlMessage::BeginExperiment);
    for id in &[Pi, DeviceId::Jetson] {
        if matches!(config.mode, ExperimentMode::LocalOnly) && *id == DeviceId::Jetson {
            continue;
        }
        if let Some(sender) = get_device_sender(id).await {
            sender.send(begin_msg.clone())?;
        }
    }

    let start = Instant::now();
    let mut sequence_id: u64 = 0;
    let mut frame_number: u64 = 1;

    while start.elapsed().as_secs() < config.duration_seconds {
        if let Some(sender) = get_device_sender(&Pi).await {
            let mut timing = TimingMetadata::default();
            timing.sequence_id = sequence_id;
            timing.controller_sent_pulse = Some(current_timestamp_micros());
            timing.frame_number = frame_number;
            let pulse = Message::Pulse(timing);
            sender.send(pulse)?;
            info!("Sent pulse {} to Pi", sequence_id);
            sequence_id += 1;
            frame_number += 30;
            if frame_number >= 30000 {
                // wrap to prevent file not found
                frame_number = 1;
            }
        }
        sleep(Duration::from_millis((1000.0 / config.fixed_fps) as u64)).await;
    }
    
    sleep(Duration::from_millis(2500)).await;
    
    let shutdown = Message::Control(ControlMessage::Shutdown);
    for id in &[Pi, DeviceId::Jetson] {
        if let Some(sender) = get_device_sender(id).await {
            let _ = sender.send(shutdown.clone());
        }
    }

    info!("Shutdown complete");

    let locked_results = results.lock().unwrap();
    generate_analysis_csv(&locked_results, &config.experiment_id, &config)?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    tracing_subscriber::fmt::init();

    let args: Vec<String> = env::args().collect();
    let model_name = args.iter()
        .find(|arg| arg.starts_with("--model="))
        .map(|arg| arg.trim_start_matches("--model=").to_string())
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());

    let fps = args.iter()
        .find(|arg| arg.starts_with("--fps="))
        .map(|arg| arg.trim_start_matches("--fps=").parse::<f32>().unwrap_or(DEFAULT_FPS))
        .unwrap_or(DEFAULT_FPS);

    let modes = match args.iter().find(|a| a == &&"--local".to_string() || a == &&"--remote".to_string()) {
        Some(flag) if flag == "--local" => vec![ExperimentMode::LocalOnly],
        Some(flag) if flag == "--remote" => vec![ExperimentMode::Offload],
        _ => vec![ExperimentMode::LocalOnly, ExperimentMode::Offload],
    };

    for mode in modes {
        let experiment_id = format!("{:?}_{}_{}fps", mode, model_name, fps);
        let mut config = ExperimentConfig::new(experiment_id.clone(), mode.clone(), model_name.clone());
        config.fixed_fps = fps;
        info!("Starting experiment with {} FPS", fps);
        run_controller(config).await?;
        sleep(Duration::from_secs(2)).await;
    }

    Ok(())
}

fn generate_analysis_csv(results: &Vec<InferenceMessage>, experiment_id: &str, config: &ExperimentConfig) -> Result<(), Box<dyn Error + Send + Sync>> {
    if results.is_empty() {
        return Ok(());
    }

    info!("Generating analysis csv");

    std::fs::create_dir_all("logs")?;
    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let filename = format!("logs/experiment_{}_{}.csv", experiment_id, timestamp);
    let mut writer = Writer::from_path(&filename)?;

    let headers = [
        "sequence_id", "frame_number", "pi_hostname", "pi_capture_start", "pi_sent_to_jetson",
        "jetson_received", "jetson_inference_start", "jetson_inference_complete",
        "jetson_sent_result", "controller_sent_pulse", "controller_received",
        "pi_overhead_us", "jetson_inference_us", "jetson_overhead_us", "network_latency_us", "total_latency_us",
        "processing_latency_pi_us",
        "frame_size_bytes", "detection_count", "image_width", "image_height",
        "model_name", "experiment_mode",
    ];
    writer.write_record(&headers)?;

    for r in results {
        let t = &r.timing;
        let i = &r.inference;

        let pi_overhead = opt_diff_val(t.pi_sent_to_jetson, t.pi_capture_start);
        let jetson_inference = opt_diff_val(t.jetson_inference_complete, t.jetson_inference_start);
        let jetson_overhead = opt_diff_val(t.jetson_sent_result, t.jetson_received);
        let total_latency = opt_diff_val(t.controller_received, t.controller_sent_pulse);

        let network_latency = if matches!(config.mode, ExperimentMode::Offload) {
            // For offload: total - (pi_overhead + jetson_overhead)
            // jetson_overhead already includes inference time
            total_latency.saturating_sub(pi_overhead + jetson_overhead)
        } else {
            // For local: total - pi_processing_time
            total_latency.saturating_sub(i.processing_time_us)
        };

        let record = vec![
            t.sequence_id.to_string(),
            t.frame_number.to_string(),
            t.pi_hostname.clone(),
            opt(t.pi_capture_start),
            opt(t.pi_sent_to_jetson),
            opt(t.jetson_received),
            opt(t.jetson_inference_start),
            opt(t.jetson_inference_complete),
            opt(t.jetson_sent_result),
            opt(t.controller_sent_pulse),
            opt(t.controller_received),
            pi_overhead.to_string(),
            jetson_inference.to_string(),
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
        ];
        writer.write_record(&record)?;
    }

    writer.flush()?;
    info!("Analysis saved to {}", filename);
    Ok(())
}

fn opt<T: ToString>(opt: Option<T>) -> String {
    opt.map(|v| v.to_string()).unwrap_or_default()
}

fn opt_diff_val(a: Option<u64>, b: Option<u64>) -> u64 {
    match (a, b) {
        (Some(a), Some(b)) => a.saturating_sub(b),
        _ => 0,
    }
}