use std::error;
use csv::Writer;
use log::{debug, error, info};
use shared::{
    current_timestamp_micros, receive_message, send_message, ControlMessage, ExperimentConfig,
    ExperimentMode, NetworkMessage, ProcessingResult,
};
use std::env;
use tokio::net::{TcpListener, TcpStream};
use tokio::time::{sleep, Duration, Instant};
use shared::constants::{controller_bind_address, jetson_full_address, pi_full_address, CONTROLLER_PORT, DEFAULT_MODEL};

#[tokio::main]
async fn main() -> Result<(), Box<dyn error::Error + Send + Sync>> {
    if env::var("RUST_LOG").is_err() {
        unsafe { env::set_var("RUST_LOG", "info"); }
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
    let listener = TcpListener::bind(controller_bind_address()).await?;
    debug!("Listening on port {}", CONTROLLER_PORT);

    let mut pi_stream = TcpStream::connect(pi_full_address()).await?;
    info!("Connected to Pi");

    if matches!(config.mode, ExperimentMode::Offload) {
  
        let mut jetson_stream = TcpStream::connect(jetson_full_address()).await?;
        let control_message = NetworkMessage::Control(ControlMessage::StartExperiment {
            config: config.clone(),
        });
        send_message(&mut jetson_stream, &control_message).await?;
        info!("Connected to Jetson and sent config");
    }

    let control_message = NetworkMessage::Control(ControlMessage::StartExperiment {
        config: config.clone(),
    });
    send_message(&mut pi_stream, &control_message).await?;
    debug!("Sent experiment config to Pi");

    let mut results = Vec::new();
    let experiment_start = Instant::now();

    while experiment_start.elapsed().as_secs() < config.duration_seconds {
        tokio::select! {
            accept_result = listener.accept() => {
                if let Ok((mut stream, addr)) = accept_result {
                    debug!("Received connection from {}", addr);

                    if let Ok(message) = receive_message::<ControlMessage>(&mut stream).await {
                        if let ControlMessage::ProcessingResult(mut result) = message {
                            result.timing.controller_received = Some(current_timestamp_micros());
                            results.push(result);

                            debug!("Received result for frame {}", results.len());
                        }
                    }
                }
            }
            _ = sleep(Duration::from_millis(100)) => {
                // Continue listening
            }
        }
    }

    let shutdown_message = NetworkMessage::Control(ControlMessage::Shutdown);
    let _ = send_message(&mut pi_stream, &shutdown_message).await;
    debug!("Sent shutdown to Pi");

    Ok(results)
}

fn generate_analysis_csv(
    results: &[ProcessingResult],
    experiment_id: &str,
) -> Result<(), Box<dyn error::Error + Send + Sync>> {
    if results.is_empty() {
        debug!("No results to save for experiment {}", experiment_id);
        return Ok(());
    }

    let filename = format!("experiment_{}_analysis.csv", experiment_id);
    let mut writer = Writer::from_path(&filename).map_err(|e| {
        error!("Failed to create CSV file {}: {}", filename, e);
        e
    })?;

    writer.write_record(&[
        "sequence_id",
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
        "detections_count",
    ])?;

    for result in results {
        let timing = &result.timing;

        writer.write_record(&[
            &timing.sequence_id.to_string(),
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
            &timing
                .jetson_processing_us()
                .map(|t| t.to_string())
                .unwrap_or_default(),
            &timing
                .total_latency_us()
                .map(|t| t.to_string())
                .unwrap_or_default(),
            &result.inference.detections.len().to_string(),
        ])?;
    }

    writer.flush()?;
    info!("Analysis saved to {}", filename);
    Ok(())
}
