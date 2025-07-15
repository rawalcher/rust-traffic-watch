use csv::Writer;
use log::{info, debug};
use shared::constants::*;
use shared::{
    current_timestamp_micros, ControlMessage, ExperimentConfig, ExperimentMode,
    NetworkMessage, ProcessingResult, ConnectionManager, ConnectionListener,
    PersistentConnection, ConnectionRole, ConnectionType
};
use std::env;
use std::error;
use tokio::time::{sleep, Duration, Instant};

pub struct ControllerService {
    control_connections: ConnectionManager,
    data_connections: ConnectionManager,
}

impl ControllerService {
    pub async fn new() -> Result<Self, Box<dyn error::Error + Send + Sync>> {
        let control_connections = ConnectionManager::new();
        let data_connections = ConnectionManager::new();

        Ok(Self {
            control_connections,
            data_connections,
        })
    }

    pub async fn run_experiment(&mut self, config: ExperimentConfig) -> Result<Vec<ProcessingResult>, Box<dyn error::Error + Send + Sync>> {
        info!("Starting experiment: {}", config.experiment_id);

        self.setup_connections(&config).await?;
        self.send_experiment_config(&config).await?;
        self.wait_for_ready_confirmations(&config).await?;
        
        sleep(Duration::from_secs(2)).await;
        
        self.start_experiment().await?;
        let results = self.collect_results(&config).await?;
        
        self.shutdown().await?;

        Ok(results)
    }

    async fn setup_connections(&mut self, config: &ExperimentConfig) -> Result<(), Box<dyn error::Error + Send + Sync>> {
        let expected_devices = match config.mode {
            ExperimentMode::LocalOnly => vec![
                (ConnectionRole::Pi, PI_ADDRESS.to_string())
            ],
            ExperimentMode::Offload => vec![
                (ConnectionRole::Pi, PI_ADDRESS.to_string()),
                (ConnectionRole::Jetson, JETSON_ADDRESS.to_string())
            ]
        };

        info!("Waiting for {} device(s) to connect...", expected_devices.len());

        let control_listener = ConnectionListener::new(
            &controller_control_bind_address(),
            expected_devices.clone(),
        ).await?;

        let control_streams = control_listener.accept_expected_connections().await?;
        for (role, stream) in control_streams {
            let connection = PersistentConnection::from_stream(
                stream,
                format!("{:?}_control", role),
                role.clone(),
                ConnectionType::Control,
            );
            self.control_connections.add_connection(connection);
        }

        let data_listener = ConnectionListener::new(
            &controller_data_bind_address(),
            expected_devices,
        ).await?;

        let data_streams = data_listener.accept_expected_connections().await?;
        for (role, stream) in data_streams {
            let connection = PersistentConnection::from_stream(
                stream,
                format!("{:?}_data", role),
                role.clone(),
                ConnectionType::Data,
            );
            self.data_connections.add_connection(connection);
        }

        info!("All connections established!");
        Ok(())
    }

    async fn send_experiment_config(&self, config: &ExperimentConfig) -> Result<(), Box<dyn error::Error + Send + Sync>> {
        let message = NetworkMessage::Control(ControlMessage::StartExperiment {
            config: config.clone()
        });

        if let Some(pi_control) = self.control_connections.get_connection(ConnectionRole::Pi, ConnectionType::Control) {
            pi_control.send(&message).await?;
            info!("Sent config to Pi");
        }

        if matches!(config.mode, ExperimentMode::Offload) {
            if let Some(jetson_control) = self.control_connections.get_connection(ConnectionRole::Jetson, ConnectionType::Control) {
                jetson_control.send(&message).await?;
                info!("Sent config to Jetson");
            }
        }

        Ok(())
    }

    async fn wait_for_ready_confirmations(&self, config: &ExperimentConfig) -> Result<(), Box<dyn error::Error + Send + Sync>> {
        info!("Waiting for ready confirmations...");

        if let Some(pi_control) = self.control_connections.get_connection(ConnectionRole::Pi, ConnectionType::Control) {
            let response: NetworkMessage = pi_control.receive().await?;
            match response {
                NetworkMessage::Control(ControlMessage::DataConnectionReady) => {
                    info!("Pi ready");
                }
                msg => return Err(format!("Expected ready from Pi, got: {:?}", msg).into()),
            }
        }

        if matches!(config.mode, ExperimentMode::Offload) {
            if let Some(jetson_control) = self.control_connections.get_connection(ConnectionRole::Jetson, ConnectionType::Control) {
                let response: NetworkMessage = jetson_control.receive().await?;
                match response {
                    NetworkMessage::Control(ControlMessage::DataConnectionReady) => {
                        info!("Jetson ready");
                    }
                    msg => return Err(format!("Expected ready from Jetson, got: {:?}", msg).into()),
                }
            }
        }

        info!("All devices ready!");
        Ok(())
    }

    async fn start_experiment(&self) -> Result<(), Box<dyn error::Error + Send + Sync>> {
        let message = NetworkMessage::Control(ControlMessage::BeginExperiment);

        for connection in &self.control_connections.connections {
            connection.send(&message).await?;
        }

        info!("Experiment started!");
        Ok(())
    }

    async fn collect_results(&self, config: &ExperimentConfig) -> Result<Vec<ProcessingResult>, Box<dyn error::Error + Send + Sync>> {
        let mut results = Vec::new();
        let experiment_start = Instant::now();

        info!("Collecting results for {} seconds...", config.duration_seconds);

        let result_source = match config.mode {
            ExperimentMode::LocalOnly => ConnectionRole::Pi,
            ExperimentMode::Offload => ConnectionRole::Jetson,
        };

        if let Some(data_conn) = self.data_connections.get_connection(result_source, ConnectionType::Data) {
            while experiment_start.elapsed().as_secs() < config.duration_seconds {
                tokio::select! {
                    result = data_conn.receive::<ProcessingResult>() => {
                        match result {
                            Ok(mut processing_result) => {
                                processing_result.timing.controller_received = Some(current_timestamp_micros());
                                debug!("Received result {} (total: {})",
                                       processing_result.timing.sequence_id, results.len() + 1);
                                results.push(processing_result);
                            }
                            Err(e) => {
                                debug!("Result receive error: {}", e);
                            }
                        }
                    }
                    _ = sleep(Duration::from_millis(100)) => {
                        // Continue collecting
                    }
                }
            }
        }

        info!("Collected {} results", results.len());
        Ok(results)
    }

    async fn shutdown(&self) -> Result<(), Box<dyn error::Error + Send + Sync>> {
        let message = NetworkMessage::Control(ControlMessage::Shutdown);

        for connection in &self.control_connections.connections {
            let _ = connection.send(&message).await; // Ignore errors during shutdown
        }

        info!("Shutdown signals sent");

        self.control_connections.disconnect_all().await;
        self.data_connections.disconnect_all().await;

        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn error::Error + Send + Sync>> {
    if env::var("RUST_LOG").is_err() {
        unsafe { env::set_var("RUST_LOG", "info"); }
    }
    env_logger::init();

    info!("Controller starting...");

    let args: Vec<String> = env::args().collect();
    let model_name = args.iter()
        .find(|arg| arg.starts_with("--model="))
        .map(|arg| arg.strip_prefix("--model=").unwrap().to_string())
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());

    let modes = if args.contains(&"--local".to_string()) {
        vec![ExperimentMode::LocalOnly]
    } else if args.contains(&"--remote".to_string()) {
        vec![ExperimentMode::Offload]
    } else {
        vec![ExperimentMode::LocalOnly, ExperimentMode::Offload]
    };

    for mode in &modes {
        let experiment_id = format!("{:?}_{}", mode, model_name);
        let config = ExperimentConfig::new(experiment_id.clone(), mode.clone(), model_name.clone());

        let mut controller = ControllerService::new().await?;
        let results = controller.run_experiment(config).await?;
        generate_analysis_csv(&results, &experiment_id)?;

        sleep(Duration::from_secs(2)).await;
    }

    Ok(())
}

fn generate_analysis_csv(
    results: &[ProcessingResult],
    experiment_id: &str,
) -> Result<(), Box<dyn error::Error + Send + Sync>> {
    if results.is_empty() {
        return Ok(());
    }

    std::fs::create_dir_all("logs")?;
    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let filename = format!("logs/experiment_{}_{}.csv", experiment_id, timestamp);

    let mut writer = Writer::from_path(&filename)?;

    writer.write_record(&[
        "sequence_id", "pi_hostname", "pi_capture_start", "pi_sent_to_jetson",
        "jetson_received", "jetson_inference_start", "jetson_inference_complete",
        "jetson_sent_result", "controller_received", "pi_overhead_us",
        "network_latency_us", "processing_time_us", "total_latency_us",
        "frame_size_bytes", "detection_count", "image_width", "image_height",
        "model_name", "experiment_mode",
    ])?;

    for result in results {
        let timing = &result.timing;
        let inference = &result.inference;

        writer.write_record(&[
            &timing.sequence_id.to_string(),
            &timing.pi_hostname,
            &timing.pi_capture_start.map(|t| t.to_string()).unwrap_or_default(),
            &timing.pi_sent_to_jetson.map(|t| t.to_string()).unwrap_or_default(),
            &timing.jetson_received.map(|t| t.to_string()).unwrap_or_default(),
            &timing.jetson_inference_start.map(|t| t.to_string()).unwrap_or_default(),
            &timing.jetson_inference_complete.map(|t| t.to_string()).unwrap_or_default(),
            &timing.jetson_sent_result.map(|t| t.to_string()).unwrap_or_default(),
            &timing.controller_received.map(|t| t.to_string()).unwrap_or_default(),
            &timing.pi_overhead_us().map(|t| t.to_string()).unwrap_or_default(),
            &timing.network_latency_us().map(|t| t.to_string()).unwrap_or_default(),
            &inference.processing_time_us.to_string(),
            &timing.total_latency_us().map(|t| t.to_string()).unwrap_or_default(),
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