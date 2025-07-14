use log::{debug, error, info};
use shared::constants::*;
use shared::{
    current_timestamp_micros, perform_python_inference_with_counts,
    ControlMessage, ExperimentConfig, NetworkMessage, PersistentPythonDetector,
    ProcessingResult, TimingPayload, ConnectionManager,
    PersistentConnection, ConnectionRole, ConnectionType
};
use std::error;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::{sleep, Duration};

pub struct JetsonService {
    connection_manager: Arc<ConnectionManager>,
    detector: Arc<Mutex<Option<PersistentPythonDetector>>>,
    should_shutdown: Arc<AtomicBool>,
    experiment_started: Arc<AtomicBool>,
    current_config: Arc<Mutex<Option<ExperimentConfig>>>,
}

impl JetsonService {
    pub async fn new() -> Result<Self, Box<dyn error::Error + Send + Sync>> {
        let mut connection_manager = ConnectionManager::new();

        connection_manager.add_connection(PersistentConnection::new(
            controller_control_address(),
            ConnectionRole::Controller,
            ConnectionType::Control,
        ));

        connection_manager.add_connection(PersistentConnection::new(
            controller_data_address(),
            ConnectionRole::Controller,
            ConnectionType::Data,
        ));

        Ok(Self {
            connection_manager: Arc::new(connection_manager),
            detector: Arc::new(Mutex::new(None)),
            should_shutdown: Arc::new(AtomicBool::new(false)),
            experiment_started: Arc::new(AtomicBool::new(false)),
            current_config: Arc::new(Mutex::new(None)),
        })
    }

    pub async fn start(&mut self) -> Result<(), Box<dyn error::Error + Send + Sync>> {
        info!("Jetson Receiver starting...");

        while !self.should_shutdown.load(Ordering::Relaxed) {
            if let Err(e) = self.connection_manager.connect_all().await {
                error!("Failed to connect to controller: {}", e);
                sleep(Duration::from_secs(5)).await;
                continue;
            }

            let pi_listener_task = self.start_pi_listener().await?;

            if let Err(e) = self.handle_controller_messages().await {
                error!("Controller message handling failed: {}", e);

                pi_listener_task.abort();

                if e.to_string().contains("Shutdown") {
                    self.should_shutdown.store(true, Ordering::Relaxed);
                    break;
                }

                self.connection_manager.disconnect_all().await;
                sleep(Duration::from_secs(2)).await;
            }
        }

        self.cleanup().await;
        Ok(())
    }

    async fn start_pi_listener(&self) -> Result<tokio::task::JoinHandle<()>, Box<dyn error::Error + Send + Sync>> {
        let should_shutdown = Arc::clone(&self.should_shutdown);
        let experiment_started = Arc::clone(&self.experiment_started);
        let connection_manager = Arc::clone(&self.connection_manager);
        let detector = Arc::clone(&self.detector);
        let current_config = Arc::clone(&self.current_config);

        let task = tokio::spawn(async move {
            Self::handle_pi_listener(
                should_shutdown,
                experiment_started,
                connection_manager,
                detector,
                current_config,
            ).await;
        });

        Ok(task)
    }

    async fn handle_controller_messages(&mut self) -> Result<(), Box<dyn error::Error + Send + Sync>> {
        while !self.should_shutdown.load(Ordering::Relaxed) {
            let control_conn = self.connection_manager
                .get_connection(ConnectionRole::Controller, ConnectionType::Control)
                .ok_or("No controller control connection")?;

            let message: NetworkMessage = control_conn.receive().await?;

            match message {
                NetworkMessage::Control(ControlMessage::StartExperiment { config }) => {
                    self.handle_start_experiment(config).await?;
                }
                NetworkMessage::Control(ControlMessage::BeginExperiment) => {
                    info!("Experiment started!");
                    self.experiment_started.store(true, Ordering::Relaxed);
                }
                NetworkMessage::Control(ControlMessage::Shutdown) => {
                    info!("Shutdown requested");
                    self.should_shutdown.store(true, Ordering::Relaxed);
                    return Err("Shutdown received".into());
                }
                _ => {
                    debug!("Unexpected control message");
                }
            }
        }

        Ok(())
    }

    async fn handle_start_experiment(
        &mut self,
        config: ExperimentConfig,
    ) -> Result<(), Box<dyn error::Error + Send + Sync>> {
        info!("Starting preheating for experiment: {} with model {}", config.experiment_id, config.model_name);

        let detector = PersistentPythonDetector::new(
            config.model_name.clone(),
            INFERENCE_TENSORRT_PATH.to_string(),
        )?;

        {
            let mut detector_guard = self.detector.lock().await;
            *detector_guard = Some(detector);
        }

        {
            let mut config_guard = self.current_config.lock().await;
            *config_guard = Some(config);
        }

        info!("Detector initialized and preheated");

        let control_conn = self.connection_manager
            .get_connection(ConnectionRole::Controller, ConnectionType::Control)
            .ok_or("No controller control connection")?;

        let message = NetworkMessage::Control(ControlMessage::DataConnectionReady);
        control_conn.send(&message).await?;
        info!("Sent ready confirmation to controller");

        Ok(())
    }

    async fn handle_pi_listener(
        should_shutdown: Arc<AtomicBool>,
        experiment_started: Arc<AtomicBool>,
        connection_manager: Arc<ConnectionManager>,
        detector: Arc<Mutex<Option<PersistentPythonDetector>>>,
        current_config: Arc<Mutex<Option<ExperimentConfig>>>,
    ) {
        let listener = match tokio::net::TcpListener::bind(&jetson_data_bind_address()).await {
            Ok(listener) => listener,
            Err(e) => {
                error!("Failed to bind Jetson listener: {}", e);
                return;
            }
        };
        info!("Jetson listening for Pi connections on port {}", JETSON_DATA_PORT);

        while !should_shutdown.load(Ordering::Relaxed) {
            tokio::select! {
                accept_result = listener.accept() => {
                    match accept_result {
                        Ok((stream, addr)) => {
                            info!("Pi connected from: {}", addr);
                            
                            let shutdown_flag = Arc::clone(&should_shutdown);
                            let experiment_flag = Arc::clone(&experiment_started);
                            let conn_manager = Arc::clone(&connection_manager);
                            let det = Arc::clone(&detector);
                            let config = Arc::clone(&current_config);

                            tokio::spawn(async move {
                                if let Err(e) = Self::handle_pi_connection(
                                    stream,
                                    shutdown_flag,
                                    experiment_flag,
                                    conn_manager,
                                    det,
                                    config,
                                ).await {
                                    error!("Pi connection error: {}", e);
                                }
                            });
                        }
                        Err(e) => {
                            if !should_shutdown.load(Ordering::Relaxed) {
                                error!("Failed to accept Pi connection: {}", e);
                            }
                        }
                    }
                }
                _ = sleep(Duration::from_millis(100)) => {
                    // Continue listening
                }
            }
        }
    }

    async fn handle_pi_connection(
        stream: tokio::net::TcpStream,
        should_shutdown: Arc<AtomicBool>,
        experiment_started: Arc<AtomicBool>,
        connection_manager: Arc<ConnectionManager>,
        detector: Arc<Mutex<Option<PersistentPythonDetector>>>,
        config: Arc<Mutex<Option<ExperimentConfig>>>,
    ) -> Result<(), String> {
        let pi_connection = PersistentConnection::from_stream(
            stream,
            "pi_frame_stream".to_string(),
            ConnectionRole::Pi,
            ConnectionType::Data,
        );

        let mut frames_processed = 0u64;

        while !should_shutdown.load(Ordering::Relaxed) {
            let message_result: Result<NetworkMessage, _> = pi_connection.receive().await;

            match message_result {
                Ok(NetworkMessage::Frame(timing)) => {
                    if !experiment_started.load(Ordering::Relaxed) {
                        debug!("Frame received but experiment not started yet, ignoring");
                        continue;
                    }

                    let config_guard = config.lock().await;
                    let mut detector_guard = detector.lock().await;

                    if let (Some(ref mut det), Some(ref cfg)) = (detector_guard.as_mut(), config_guard.as_ref()) {
                        let data_connection = connection_manager
                            .get_connection(ConnectionRole::Controller, ConnectionType::Data)
                            .ok_or("No controller data connection".to_string())?;

                        match Self::process_frame_and_send_result(
                            timing,
                            &cfg.model_name,
                            det,
                            data_connection,
                        ).await {
                            Ok(()) => {
                                frames_processed += 1;
                                debug!("Successfully processed frame (total: {})", frames_processed);
                            }
                            Err(e) => {
                                error!("Error processing frame: {}", e);
                            }
                        }
                    } else {
                        error!("Frame received but no detector or config available");
                    }
                }
                Ok(_) => {
                    debug!("Received unexpected message type from Pi");
                }
                Err(e) => {
                    info!("Pi connection ended: {}", e);
                    break;
                }
            }
        }

        info!("Pi connection handler finished. Processed {} frames", frames_processed);
        Ok(())
    }

    async fn process_frame_and_send_result(
        mut timing: TimingPayload,
        model_name: &str,
        detector: &mut PersistentPythonDetector,
        data_connection: &PersistentConnection,
    ) -> Result<(), String> {
        timing.jetson_received = Some(current_timestamp_micros());
        timing.jetson_inference_start = Some(current_timestamp_micros());

        let (inference_result, counts) = perform_python_inference_with_counts(
            &timing, detector, model_name, "offload"
        ).await?;

        timing.jetson_inference_complete = Some(current_timestamp_micros());
        timing.jetson_sent_result = Some(current_timestamp_micros());

        let result = ProcessingResult {
            timing: timing.clone(),
            inference: inference_result,
        };

        data_connection.send(&result).await.map_err(|e| {
            error!("Failed to send result to controller for frame {}: {}", timing.sequence_id, e);
            e.to_string()
        })?;

        debug!(
            "Frame {} inference complete: {} vehicles, {} pedestrians, {:.1}ms",
            timing.sequence_id,
            counts.total_vehicles,
            counts.pedestrians,
            result.inference.processing_time_us as f64 / 1000.0
        );

        Ok(())
    }

    async fn cleanup(&mut self) {
        {
            let mut detector_guard = self.detector.lock().await;
            if let Some(mut det) = detector_guard.take() {
                if let Err(e) = det.shutdown() {
                    error!("Detector shutdown failed: {}", e);
                }
            }
        }

        self.connection_manager.disconnect_all().await;
        info!("Jetson Receiver stopped");
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn error::Error>> {
    if std::env::var("RUST_LOG").is_err() {
        unsafe { std::env::set_var("RUST_LOG", "info"); }
    }
    env_logger::init();

    let mut service = JetsonService::new().await.unwrap();
    service.start().await.unwrap();

    Ok(())
}