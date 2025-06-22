use csv::Writer;
use log::{debug, error, info, warn};
use shared::constants::{
    controller_bind_address, jetson_full_address, pi_full_address, CONTROLLER_PORT, DEFAULT_MODEL,
};
use shared::{
    current_timestamp_micros, receive_message, send_message, ControlMessage, ExperimentConfig,
    ExperimentMode, NetworkMessage, ProcessingResult,
};
use std::env;
use std::error;
use tokio::net::{TcpListener, TcpStream};
use tokio::time::{sleep, timeout, Duration, Instant};

#[tokio::main]
async fn main() -> Result<(), Box<dyn error::Error + Send + Sync>> {
    if env::var("RUST_LOG").is_err() {
        unsafe {
            env::set_var("RUST_LOG", "info");
        }
    }
    env_logger::init();

    info!("Experiment Controller starting...");

    let args: Vec<String> = env::args().collect();

    let model_name = if let Some(model_arg) = args.iter().find(|arg| arg.starts_with("--model=")) {
        model_arg.strip_prefix("--model=").unwrap().to_string()
    } else {
        DEFAULT_MODEL.to_string()
    };

    info!("Using model: {}", model_name);

    let modes = if args.contains(&"--local".to_string()) {
        vec![ExperimentMode::LocalOnly]
    } else if args.contains(&"--remote".to_string()) {
        vec![ExperimentMode::Offload]
    } else {
        vec![ExperimentMode::LocalOnly, ExperimentMode::Offload]
    };

    for mode in &modes {
        let experiment_id = format!("{:?}_{}", mode, model_name);
        info!("Starting experiment: {}", experiment_id);

        let config = ExperimentConfig::new(experiment_id.clone(), mode.clone(), model_name.clone());

        let results = run_experiment(config).await?;
        generate_analysis_csv(&results, &experiment_id)?;

        info!(
            "Experiment {} complete. Processed {} frames",
            experiment_id,
            results.len()
        );

        sleep(Duration::from_secs(2)).await;
    }

    info!("All experiments complete!");
    Ok(())
}

async fn run_experiment(
    config: ExperimentConfig,
) -> Result<Vec<ProcessingResult>, Box<dyn error::Error + Send + Sync>> {
    info!(
        "Starting preheating phase for experiment: {}",
        config.experiment_id
    );

    let listener = TcpListener::bind(controller_bind_address()).await?;
    debug!("Listening on port {}", CONTROLLER_PORT);

    // Phase 1: Connect and send config for preheating
    let mut pi_stream = TcpStream::connect(pi_full_address()).await?;
    info!("Connected to Pi");

    let mut jetson_stream_opt = None;
    if matches!(config.mode, ExperimentMode::Offload) {
        let mut jetson_stream = TcpStream::connect(jetson_full_address()).await?;
        let control_message = NetworkMessage::Control(ControlMessage::StartExperiment {
            config: config.clone(),
        });
        send_message(&mut jetson_stream, &control_message).await?;
        info!("Connected to Jetson and sent config for preheating");
        jetson_stream_opt = Some(jetson_stream);
    }

    let control_message = NetworkMessage::Control(ControlMessage::StartExperiment {
        config: config.clone(),
    });
    send_message(&mut pi_stream, &control_message).await?;
    info!("Sent experiment config to Pi for preheating");

    // Phase 2: Wait for preheating completion
    let mut devices_ready = 0;
    let expected_devices = if matches!(config.mode, ExperimentMode::Offload) { 2 } else { 1 };

    info!("Waiting for {} device(s) to complete preheating...", expected_devices);

    while devices_ready < expected_devices {
        let (mut stream, addr) = listener.accept().await?;
        debug!("Received connection from {}", addr);

        match timeout(Duration::from_secs(60), receive_message::<ControlMessage>(&mut stream)).await {
            Ok(Ok(ControlMessage::PreheatingComplete)) => {
                devices_ready += 1;
                info!("Device {} preheating complete ({}/{})", addr, devices_ready, expected_devices);
            }
            Ok(Ok(msg)) => {
                warn!("Unexpected message during preheating: {:?}", msg);
            }
            Ok(Err(e)) => {
                error!("Error receiving preheating confirmation: {}", e);
            }
            Err(_) => {
                error!("Timeout waiting for preheating confirmation from {}", addr);
                return Err("Preheating timeout".into());
            }
        }
    }

    info!("All devices ready! Starting experiment in 2 seconds...");
    sleep(Duration::from_secs(2)).await;

    // Phase 3: Signal devices to begin actual experiment
    let begin_message = NetworkMessage::Control(ControlMessage::BeginExperiment);
    send_message(&mut pi_stream, &begin_message).await?;

    if let Some(ref mut jetson_stream) = jetson_stream_opt {
        send_message(jetson_stream, &begin_message).await?;
    }

    info!("Experiment started - collecting results...");

    // Phase 4: Collect experiment results
    let results = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let experiment_start = Instant::now();

    while experiment_start.elapsed().as_secs() < config.duration_seconds {
        tokio::select! {
            accept_result = listener.accept() => {
                if let Ok((stream, addr)) = accept_result {
                    debug!("New connection from {}", addr);

                    let results_clone = std::sync::Arc::clone(&results);
                    tokio::spawn(async move {
                        handle_result_connection(stream, addr, results_clone).await;
                    });
                }
            }
            _ = sleep(Duration::from_millis(50)) => {
                // Continue listening
            }
        }
    }

    // Phase 5: Shutdown
    let shutdown_message = NetworkMessage::Control(ControlMessage::Shutdown);
    let _ = send_message(&mut pi_stream, &shutdown_message).await;

    if let Some(ref mut jetson_stream) = jetson_stream_opt {
        let _ = send_message(jetson_stream, &shutdown_message).await;
    }

    info!("Sent shutdown signals to all devices");

    let final_results = {
        let results_guard = results.lock().unwrap();
        results_guard.clone()
    };

    info!("Collected {} total results", final_results.len());
    Ok(final_results)
}

async fn handle_result_connection(
    mut stream: TcpStream,
    addr: std::net::SocketAddr,
    results: std::sync::Arc<std::sync::Mutex<Vec<ProcessingResult>>>,
) {
    info!("Handling persistent result connection from {}", addr);
    let mut frame_count = 0;

    loop {
        match receive_message::<ControlMessage>(&mut stream).await {
            Ok(ControlMessage::ProcessingResult(mut result)) => {
                result.timing.controller_received = Some(current_timestamp_micros());

                {
                    let mut results_guard = results.lock().unwrap();
                    results_guard.push(result);
                    frame_count += 1;
                }

                debug!("Received result {} from {}", frame_count, addr);
            }
            Ok(other_msg) => {
                debug!("Unexpected message from {}: {:?}", addr, other_msg);
            }
            Err(e) => {
                info!(
                    "Result connection from {} ended after {} frames: {}",
                    addr, frame_count, e
                );
                break;
            }
        }
    }
}

fn generate_analysis_csv(
    results: &[ProcessingResult],
    experiment_id: &str,
) -> Result<(), Box<dyn error::Error + Send + Sync>> {
    if results.is_empty() {
        debug!("No results to save for experiment {}", experiment_id);
        return Ok(());
    }

    std::fs::create_dir_all("logs")?;

    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let filename = format!("logs/experiment_{}_{}.csv", experiment_id, timestamp);

    let mut writer = Writer::from_path(&filename).map_err(|e| {
        error!("Failed to create CSV file {}: {}", filename, e);
        e
    })?;

    writer.write_record(&[
        "sequence_id",
        "pi_hostname",
        "pi_capture_start",
        "pi_sent_to_jetson",
        "jetson_received",
        "jetson_inference_start",
        "jetson_inference_complete",
        "jetson_sent_result",
        "controller_received",
        "pi_overhead_us",
        "network_latency_us",
        "processing_time_us",
        "total_latency_us",
        "frame_size_bytes",
        "detection_count",
        "image_width",
        "image_height",
        "model_name",
        "experiment_mode",
    ])?;

    for result in results {
        let timing = &result.timing;
        let inference = &result.inference;

        writer.write_record(&[
            &timing.sequence_id.to_string(),
            &timing.pi_hostname, // NEW: Include Pi hostname
            &timing
                .pi_capture_start
                .map(|t| t.to_string())
                .unwrap_or_default(),
            &timing
                .pi_sent_to_jetson
                .map(|t| t.to_string())
                .unwrap_or_default(),
            &timing
                .jetson_received
                .map(|t| t.to_string())
                .unwrap_or_default(),
            &timing
                .jetson_inference_start
                .map(|t| t.to_string())
                .unwrap_or_default(),
            &timing
                .jetson_inference_complete
                .map(|t| t.to_string())
                .unwrap_or_default(),
            &timing
                .jetson_sent_result
                .map(|t| t.to_string())
                .unwrap_or_default(),
            &timing
                .controller_received
                .map(|t| t.to_string())
                .unwrap_or_default(),
            &timing
                .pi_overhead_us()
                .map(|t| t.to_string())
                .unwrap_or_default(),
            &timing
                .network_latency_us()
                .map(|t| t.to_string())
                .unwrap_or_default(),
            &inference.processing_time_us.to_string(),
            &timing
                .total_latency_us()
                .map(|t| t.to_string())
                .unwrap_or_default(),
            &inference.frame_size_bytes.to_string(),
            &inference.detection_count.to_string(),
            &inference.image_width.to_string(),
            &inference.image_height.to_string(),
            &inference.model_name,
            &inference.experiment_mode,
        ])?;
    }

    writer.flush()?;
    info!("Analysis saved to {}", filename);
    Ok(())
}
