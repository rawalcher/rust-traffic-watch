use std::{env, error::Error};
use tokio::time::{sleep, Duration, Instant};
use tracing::{debug, info, warn};

use csv::Writer;
use shared::constants::*;
use shared::connection::{get_device_sender, start_controller_listener};
use shared::types::*;
use shared::current_timestamp_micros;

pub async fn run_controller(config: ExperimentConfig) -> Result<(), Box<dyn Error + Send + Sync>> {
    info!("Starting experiment: {}", config.experiment_id);

    tokio::spawn(async {
        if let Err(e) = start_controller_listener().await {
            warn!("Connection listener error: {}", e);
        }
    });

    sleep(Duration::from_secs(2)).await;

    let msg = Message::Control(ControlMessage::StartExperiment { config: config.clone() });
    for id in &[DeviceId::Pi, DeviceId::Jetson] {
        if matches!(config.mode, ExperimentMode::LocalOnly) && *id == DeviceId::Jetson {
            continue;
        }
        if let Some(sender) = get_device_sender(id).await {
            sender.send(msg.clone())?;
        }
    }

    for &id in &[DeviceId::Pi, DeviceId::Jetson] {
        if matches!(config.mode, ExperimentMode::LocalOnly) && id == DeviceId::Jetson {
            continue;
        }
        info!("Waiting for {:?} to be ready...", id);
        sleep(Duration::from_millis(100)).await;
    }

    let begin_msg = Message::Control(ControlMessage::BeginExperiment);
    for id in &[DeviceId::Pi, DeviceId::Jetson] {
        if matches!(config.mode, ExperimentMode::LocalOnly) && *id == DeviceId::Jetson {
            continue;
        }
        if let Some(sender) = get_device_sender(id).await {
            sender.send(begin_msg.clone())?;
        }
    }

    let mut results = Vec::new();
    let start = Instant::now();
    let mut sequence_id: u64 = 0;

    while start.elapsed().as_secs() < config.duration_seconds {
        if let Some(sender) = get_device_sender(&DeviceId::Pi).await {
            let mut timing = TimingMetadata::default();
            timing.sequence_id = sequence_id;
            timing.controller_sent_pulse = Some(current_timestamp_micros());
            let pulse = Message::Control(ControlMessage::Pulse { timing });
            sender.send(pulse)?;
            info!("Sent pulse {} to Pi", sequence_id);
            sequence_id += 1;
        }
        sleep(Duration::from_millis(1000 / config.fixed_fps as u64)).await;
    }

    let shutdown = Message::Control(ControlMessage::Shutdown);
    for id in &[DeviceId::Pi, DeviceId::Jetson] {
        if let Some(sender) = get_device_sender(id).await {
            let _ = sender.send(shutdown.clone());
        }
    }

    generate_analysis_csv(&results, &config.experiment_id)?;
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

    let modes = match args.iter().find(|a| a == &&"--local".to_string() || a == &&"--remote".to_string()) {
        Some(flag) if flag == "--local" => vec![ExperimentMode::LocalOnly],
        Some(flag) if flag == "--remote" => vec![ExperimentMode::Offload],
        _ => vec![ExperimentMode::LocalOnly, ExperimentMode::Offload],
    };

    for mode in modes {
        let experiment_id = format!("{:?}_{}", mode, model_name);
        let config = ExperimentConfig::new(experiment_id.clone(), mode.clone(), model_name.clone());
        run_controller(config).await?;
        sleep(Duration::from_secs(2)).await;
    }

    Ok(())
}

fn generate_analysis_csv(results: &Vec<InferenceMessage>, experiment_id: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
    if results.is_empty() {
        return Ok(());
    }

    std::fs::create_dir_all("logs")?;
    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let filename = format!("logs/experiment_{}_{}.csv", experiment_id, timestamp);
    let mut writer = Writer::from_path(&filename)?;

    let headers = [
        "sequence_id", "pi_hostname", "pi_capture_start", "pi_sent_to_jetson",
        "jetson_received", "jetson_inference_start", "jetson_inference_complete",
        "jetson_sent_result", "controller_received", "pi_overhead_us",
        "network_latency_us", "processing_time_us", "total_latency_us",
        "frame_size_bytes", "detection_count", "image_width", "image_height",
        "model_name", "experiment_mode",
    ];
    writer.write_record(&headers)?;

    for r in results {
        let t = &r.timing;
        let i = &r.inference;
        let record = vec![
            t.sequence_id.to_string(),
            t.pi_hostname.clone(),
            opt(t.pi_capture_start),
            opt(t.pi_sent_to_jetson),
            opt(t.jetson_received),
            opt(t.jetson_inference_start),
            opt(t.jetson_inference_complete),
            opt(t.jetson_sent_result),
            opt(t.controller_received),
            opt(t.pi_overhead_us()),
            opt(t.network_latency_us()),
            i.processing_time_us.to_string(),
            opt(t.total_latency_us()),
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
