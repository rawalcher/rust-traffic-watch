use csv::Writer;
use log::{debug, error, info, warn};
use shared::constants::{controller_address, DEFAULT_MODEL, JETSON_ADDRESS, PI_ADDRESS};
use shared::{
    current_timestamp_micros, receive_message, send_message, ControlMessage, ExperimentConfig,
    ExperimentMode, NetworkMessage, ProcessingResult,
};
use std::env;
use std::error;
use tokio::net::{TcpListener, TcpStream};
use tokio::time::{sleep, Duration, Instant};

struct Connections {
    pi_stream: TcpStream,
    jetson_stream: Option<TcpStream>,
}

impl Connections {
    async fn wait_for_clients(
        config: &ExperimentConfig,
    ) -> Result<Self, Box<dyn error::Error + Send + Sync>> {
        let listener = TcpListener::bind(controller_address()).await?;
        info!("Controller listening on {}", controller_address());

        let mut pi_stream: Option<TcpStream> = None;
        let mut jetson_stream: Option<TcpStream> = None;

        let expected_connections = if matches!(config.mode, ExperimentMode::Offload) {
            2
        } else {
            1
        };
        let mut connections_received = 0;

        info!(
            "Waiting for {} client(s) to connect...",
            expected_connections
        );

        while connections_received < expected_connections {
            let (stream, addr) = listener.accept().await?;
            info!("Client connected from: {}", addr);

            let addr_str = addr.ip().to_string();
            if addr_str.eq(PI_ADDRESS) {
                pi_stream = Some(stream);
                info!("Pi connected");
                connections_received += 1;
            } else if addr_str.eq(JETSON_ADDRESS) && matches!(config.mode, ExperimentMode::Offload)
            {
                jetson_stream = Some(stream);
                info!("Jetson connected");
                connections_received += 1;
            } else {
                warn!("Unexpected connection from {}", addr);
            }
        }

        let pi_stream = pi_stream.ok_or("Pi never connected")?;

        info!("All expected clients connected!");

        Ok(Self {
            pi_stream,
            jetson_stream,
        })
    }

    async fn send_start_experiment(
        &mut self,
        config: &ExperimentConfig,
    ) -> Result<(), Box<dyn error::Error + Send + Sync>> {
        let message = NetworkMessage::Control(ControlMessage::StartExperiment {
            config: config.clone(),
        });

        send_message(&mut self.pi_stream, &message).await?;
        info!("Sent StartExperiment to Pi");

        if let Some(ref mut jetson_stream) = self.jetson_stream {
            send_message(jetson_stream, &message).await?;
            info!("Sent StartExperiment to Jetson");
        }

        Ok(())
    }

    async fn wait_for_preheating(
        &mut self,
        expected_devices: usize,
    ) -> Result<(), Box<dyn error::Error + Send + Sync>> {
        info!(
            "Waiting for {} device(s) to complete preheating...",
            expected_devices
        );
        let mut devices_ready = 0;

        let pi_response = receive_message::<ControlMessage>(&mut self.pi_stream).await?;
        match pi_response {
            ControlMessage::PreheatingComplete => {
                devices_ready += 1;
                info!(
                    "Pi preheating complete ({}/{})",
                    devices_ready, expected_devices
                );
            }
            msg => {
                return Err(format!("Unexpected message from Pi: {:?}", msg).into());
            }
        }

        if let Some(ref mut jetson_stream) = self.jetson_stream {
            let jetson_response = receive_message::<ControlMessage>(jetson_stream).await?;
            match jetson_response {
                ControlMessage::PreheatingComplete => {
                    devices_ready += 1;
                    info!(
                        "Jetson preheating complete ({}/{})",
                        devices_ready, expected_devices
                    );
                }
                msg => {
                    return Err(format!("Unexpected message from Jetson: {:?}", msg).into());
                }
            }
        }

        info!("All {} devices ready!", devices_ready);
        Ok(())
    }

    async fn begin_experiment(&mut self) -> Result<(), Box<dyn error::Error + Send + Sync>> {
        let begin_message = NetworkMessage::Control(ControlMessage::BeginExperiment);

        send_message(&mut self.pi_stream, &begin_message).await?;
        info!("Sent BeginExperiment to Pi");

        if let Some(ref mut jetson_stream) = self.jetson_stream {
            send_message(jetson_stream, &begin_message).await?;
            info!("Sent BeginExperiment to Jetson");
        }

        Ok(())
    }

    async fn collect_results(
        &mut self,
        config: &ExperimentConfig,
    ) -> Result<Vec<ProcessingResult>, Box<dyn error::Error + Send + Sync>> {
        let mut results = Vec::new();
        let experiment_start = Instant::now();

        info!(
            "Collecting results for {} seconds...",
            config.duration_seconds
        );

        while experiment_start.elapsed().as_secs() < config.duration_seconds {
            tokio::select! {
                result = self.receive_result_from_active_device(config) => {
                    match result {
                        Ok(mut processing_result) => {
                            processing_result.timing.controller_received = Some(current_timestamp_micros());
                            debug!("Received result {} (total: {})", processing_result.timing.sequence_id, results.len());
                            results.push(processing_result);
                        }
                        Err(e) => {
                            debug!("Result receive error (may be normal at experiment end): {}", e);
                        }
                    }
                }
                _ = sleep(Duration::from_millis(100)) => {
                    // Continue collecting
                }
            }
        }

        info!("Collected {} results", results.len());
        Ok(results)
    }

    async fn receive_result_from_active_device(
        &mut self,
        config: &ExperimentConfig,
    ) -> Result<ProcessingResult, Box<dyn error::Error + Send + Sync>> {
        match config.mode {
            ExperimentMode::LocalOnly => {
                let message = receive_message::<ControlMessage>(&mut self.pi_stream).await?;
                match message {
                    ControlMessage::ProcessingResult(result) => Ok(result),
                    msg => Err(format!("Expected ProcessingResult, got: {:?}", msg).into()),
                }
            }
            ExperimentMode::Offload => {
                if let Some(ref mut jetson_stream) = self.jetson_stream {
                    let message = receive_message::<ControlMessage>(jetson_stream).await?;
                    match message {
                        ControlMessage::ProcessingResult(result) => Ok(result),
                        msg => Err(format!("Expected ProcessingResult, got: {:?}", msg).into()),
                    }
                } else {
                    Err("No Jetson connection in offload mode".into())
                }
            }
        }
    }

    async fn shutdown(&mut self) -> Result<(), Box<dyn error::Error + Send + Sync>> {
        let shutdown_message = NetworkMessage::Control(ControlMessage::Shutdown);

        let _ = send_message(&mut self.pi_stream, &shutdown_message).await;
        info!("Sent shutdown to Pi");

        if let Some(ref mut jetson_stream) = self.jetson_stream {
            let _ = send_message(jetson_stream, &shutdown_message).await;
            info!("Sent shutdown to Jetson");
        }

        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn error::Error + Send + Sync>> {
    if env::var("RUST_LOG").is_err() {
        unsafe {
            // main() is not multithreaded so why is rust complaining??
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
    info!("Starting experiment: {}", config.experiment_id);

    let mut connections = Connections::wait_for_clients(&config).await?;
    connections.send_start_experiment(&config).await?;
    let expected_devices = if matches!(config.mode, ExperimentMode::Offload) {
        2
    } else {
        1
    };
    connections.wait_for_preheating(expected_devices).await?;

    info!("All devices ready! Starting experiment in 2 seconds...");
    sleep(Duration::from_secs(2)).await;

    connections.begin_experiment().await?;
    let results = connections.collect_results(&config).await?;
    connections.shutdown().await?;

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
            &timing.pi_hostname,
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
