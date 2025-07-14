use log::{debug, error, info, warn};
use shared::constants::*;
use shared::{
    current_timestamp_micros, perform_python_inference_with_counts,
    ControlMessage, ExperimentConfig, ExperimentMode, FrameThroughputController,
    NetworkMessage, PersistentPythonDetector, ProcessingResult, ThroughputMode,
    TimingPayload, ConnectionManager, PersistentConnection,
    ConnectionRole, ConnectionType
};
use std::error;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::time::{sleep, Duration, Instant};

pub struct PiService {
    connection_manager: ConnectionManager,
    should_shutdown: Arc<AtomicBool>,
    current_config: Option<ExperimentConfig>,
}

impl PiService {
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
            connection_manager,
            should_shutdown: Arc::new(AtomicBool::new(false)),
            current_config: None,
        })
    }

    pub async fn start(&mut self) -> Result<(), Box<dyn error::Error + Send + Sync>> {
        info!("Pi Sender starting...");

        while !self.should_shutdown.load(Ordering::Relaxed) {
            if let Err(e) = self.connection_manager.connect_all().await {
                error!("Failed to connect to controller: {}", e);
                sleep(Duration::from_secs(5)).await;
                continue;
            }

            match self.wait_for_experiment_config().await {
                Ok(config) => {
                    self.current_config = Some(config.clone());

                    let throughput_mode = if let Some(fps_arg) = std::env::args().find(|arg| arg.starts_with("--fps=")) {
                        let fps_str = fps_arg.strip_prefix("--fps=").unwrap();
                        let fps: f32 = fps_str.parse().unwrap_or(config.fixed_fps);
                        ThroughputMode::Custom(fps)
                    } else if std::env::args().any(|arg| arg == "--fps-mode") {
                        ThroughputMode::Fps
                    } else {
                        ThroughputMode::High
                    };

                    info!("Starting experiment: {} in mode {:?} with model {} (throughput: {:?})",
                          config.experiment_id, config.mode, config.model_name, throughput_mode);

                    match config.mode {
                        ExperimentMode::LocalOnly => {
                            if let Err(e) = self.run_local_experiment(config, throughput_mode).await {
                                error!("Local experiment failed: {}", e);
                            }
                        }
                        ExperimentMode::Offload => {
                            if let Err(e) = self.run_offload_experiment(config, throughput_mode).await {
                                error!("Offload experiment failed: {}", e);
                            }
                        }
                    }
                }
                Err(e) => {
                    if e.to_string().contains("Shutdown") {
                        self.should_shutdown.store(true, Ordering::Relaxed);
                        break;
                    }
                    error!("Failed to get experiment config: {}", e);
                    sleep(Duration::from_secs(2)).await;
                }
            }
        }

        self.cleanup().await;
        Ok(())
    }

    async fn wait_for_experiment_config(&self) -> Result<ExperimentConfig, Box<dyn error::Error + Send + Sync>> {
        let control_conn = self.connection_manager
            .get_connection(ConnectionRole::Controller, ConnectionType::Control)
            .ok_or("No controller control connection")?;

        let message: NetworkMessage = control_conn.receive().await?;
        match message {
            NetworkMessage::Control(ControlMessage::StartExperiment { config }) => {
                info!("Received experiment config");
                Ok(config)
            }
            NetworkMessage::Control(ControlMessage::Shutdown) => {
                Err("Shutdown received".into())
            }
            msg => Err(format!("Expected StartExperiment, got: {:?}", msg).into()),
        }
    }

    async fn run_local_experiment(
        &self,
        config: ExperimentConfig,
        throughput_mode: ThroughputMode,
    ) -> Result<(), Box<dyn error::Error + Send + Sync>> {
        info!("LOCAL MODE: Starting preheating phase...");

        let detector = PersistentPythonDetector::new(
            config.model_name.clone(),
            INFERENCE_PYTORCH_PATH.to_string(),
        )?;

        info!("Detector initialized and preheated");

        self.send_ready_confirmation().await?;
        self.wait_for_experiment_start().await?;
        self.local_processing(config, detector, throughput_mode).await?;

        sleep(Duration::from_secs(5)).await;
        self.should_shutdown.store(true, Ordering::Relaxed);
        info!("Local experiment completed, shutting down");

        Ok(())
    }

    async fn run_offload_experiment(
        &mut self,
        config: ExperimentConfig,
        throughput_mode: ThroughputMode,
    ) -> Result<(), Box<dyn error::Error + Send + Sync>> {
        info!("OFFLOAD MODE: Starting preheating phase...");

        self.connection_manager.add_connection(PersistentConnection::new(
            jetson_data_address(),
            ConnectionRole::Jetson,
            ConnectionType::Data,
        ));
        
        if let Some(jetson_conn) = self.connection_manager.get_connection(ConnectionRole::Jetson, ConnectionType::Data) {
            jetson_conn.connect().await?;
            info!("Connected to Jetson");
        }

        self.send_ready_confirmation().await?;
        self.wait_for_experiment_start().await?;
        self.offloading(config, throughput_mode).await?;
        
        sleep(Duration::from_secs(5)).await;
        self.should_shutdown.store(true, Ordering::Relaxed);
        info!("Local experiment completed, shutting down");

        Ok(())
    }

    async fn send_ready_confirmation(&self) -> Result<(), Box<dyn error::Error + Send + Sync>> {
        let control_conn = self.connection_manager
            .get_connection(ConnectionRole::Controller, ConnectionType::Control)
            .ok_or("No controller control connection")?;

        let message = NetworkMessage::Control(ControlMessage::DataConnectionReady);
        control_conn.send(&message).await?;
        info!("Sent preheating complete to controller");
        Ok(())
    }

    async fn wait_for_experiment_start(&self) -> Result<(), Box<dyn error::Error + Send + Sync>> {
        let control_conn = self.connection_manager
            .get_connection(ConnectionRole::Controller, ConnectionType::Control)
            .ok_or("No controller control connection")?;

        loop {
            if self.should_shutdown.load(Ordering::Relaxed) {
                return Err("Shutdown requested".into());
            }

            let message: NetworkMessage = control_conn.receive().await?;
            match message {
                NetworkMessage::Control(ControlMessage::BeginExperiment) => {
                    info!("Received experiment start signal!");
                    break;
                }
                NetworkMessage::Control(ControlMessage::Shutdown) => {
                    self.should_shutdown.store(true, Ordering::Relaxed);
                    return Err("Shutdown received during wait".into());
                }
                msg => {
                    warn!("Unexpected message while waiting for start: {:?}", msg);
                }
            }
        }
        Ok(())
    }

    async fn local_processing(
        &self,
        config: ExperimentConfig,
        mut detector: PersistentPythonDetector,
        throughput_mode: ThroughputMode,
    ) -> Result<(), Box<dyn error::Error + Send + Sync>> {
        let throughput_controller = FrameThroughputController::new(throughput_mode);
        let frame_skip = throughput_controller.get_frame_skip(config.fixed_fps);
        let actual_fps = 30.0 / frame_skip as f32;
        let frame_interval = Duration::from_secs_f32(1.0 / actual_fps);

        info!("Local processing: skipping every {} frames for {:.1} FPS", frame_skip, actual_fps);

        let mut dataset_frame_id = 1u64;
        let mut logical_sequence_id = 1u64;
        let mut frames_processed = 0u64;
        let experiment_start = Instant::now();

        let data_conn = self.connection_manager
            .get_connection(ConnectionRole::Controller, ConnectionType::Data)
            .ok_or("No controller data connection")?;

        while experiment_start.elapsed().as_secs() < config.duration_seconds && !self.should_shutdown.load(Ordering::Relaxed) {
            let mut timing = TimingPayload::new(logical_sequence_id);

            let frame_data = match self.load_frame_from_image(dataset_frame_id) {
                Ok(data) => data,
                Err(e) => {
                    error!("Failed to load frame {}: {}", dataset_frame_id, e);
                    self.advance_frame_ids(&mut dataset_frame_id, &mut logical_sequence_id, frame_skip);
                    sleep(frame_interval).await;
                    continue;
                }
            };

            timing.add_frame_data(frame_data, FRAME_WIDTH, FRAME_HEIGHT);

            let (inference_result, counts) = match perform_python_inference_with_counts(
                &timing, &mut detector, &config.model_name, "local"
            ).await {
                Ok(result) => result,
                Err(e) => {
                    error!("Python inference failed for frame {}: {}", dataset_frame_id, e);
                    self.advance_frame_ids(&mut dataset_frame_id, &mut logical_sequence_id, frame_skip);
                    sleep(frame_interval).await;
                    continue;
                }
            };

            let result = ProcessingResult {
                timing: timing.clone(),
                inference: inference_result.clone(),
            };

            if let Err(e) = data_conn.send(&result).await {
                error!("Failed to send result to controller for frame {}: {}", dataset_frame_id, e);
                if self.should_shutdown.load(Ordering::Relaxed) {
                    break;
                }
            } else {
                frames_processed += 1;
                debug!("Frame {} (dataset frame {}) processed: {} vehicles, {} pedestrians, {:.1}ms",
                       logical_sequence_id, dataset_frame_id, counts.total_vehicles, 
                       counts.pedestrians, inference_result.processing_time_us as f64 / 1000.0);
            }

            self.advance_frame_ids(&mut dataset_frame_id, &mut logical_sequence_id, frame_skip);
            sleep(frame_interval).await;
        }

        detector.shutdown()?;
        info!("Local processing completed. Processed: {} frames", frames_processed);
        Ok(())
    }

    async fn offloading(
        &self,
        config: ExperimentConfig,
        throughput_mode: ThroughputMode,
    ) -> Result<(), Box<dyn error::Error + Send + Sync>> {
        let throughput_controller = FrameThroughputController::new(throughput_mode);
        let frame_skip = throughput_controller.get_frame_skip(config.fixed_fps);
        let actual_fps = 30.0 / frame_skip as f32;
        let frame_interval = Duration::from_secs_f32(1.0 / actual_fps);

        info!("Offloading: skipping every {} frames for {:.1} FPS", frame_skip, actual_fps);

        let mut dataset_frame_id = 1u64;
        let mut logical_sequence_id = 1u64;
        let mut frames_sent = 0u64;
        let experiment_start = Instant::now();

        let jetson_conn = self.connection_manager
            .get_connection(ConnectionRole::Jetson, ConnectionType::Data)
            .ok_or("No Jetson data connection")?;

        while experiment_start.elapsed().as_secs() < config.duration_seconds && !self.should_shutdown.load(Ordering::Relaxed) {
            let mut timing = TimingPayload::new(logical_sequence_id);

            let frame_data = match self.load_frame_from_image(dataset_frame_id) {
                Ok(data) => data,
                Err(e) => {
                    error!("Failed to load frame {}: {}", dataset_frame_id, e);
                    self.advance_frame_ids(&mut dataset_frame_id, &mut logical_sequence_id, frame_skip);
                    sleep(frame_interval).await;
                    continue;
                }
            };

            timing.add_frame_data(frame_data, FRAME_WIDTH, FRAME_HEIGHT);
            timing.pi_sent_to_jetson = Some(current_timestamp_micros());

            let frame_message = NetworkMessage::Frame(timing);
            match jetson_conn.send(&frame_message).await {
                Ok(()) => {
                    frames_sent += 1;
                    debug!("Sent frame {} (dataset frame {}) to Jetson (total sent: {})",
                           logical_sequence_id, dataset_frame_id, frames_sent);
                }
                Err(e) => {
                    error!("Failed to send frame {} to Jetson: {}", dataset_frame_id, e);
                    if self.should_shutdown.load(Ordering::Relaxed) {
                        break;
                    }
                    if let Err(reconnect_err) = jetson_conn.connect().await {
                        error!("Failed to reconnect to Jetson: {}", reconnect_err);
                        break;
                    } else {
                        warn!("Reconnected to Jetson, continuing...");
                    }
                }
            }

            self.advance_frame_ids(&mut dataset_frame_id, &mut logical_sequence_id, frame_skip);
            sleep(frame_interval).await;
        }

        info!("Offloading completed. Sent: {} frames", frames_sent);
        Ok(())
    }

    fn advance_frame_ids(&self, dataset_frame_id: &mut u64, logical_sequence_id: &mut u64, frame_skip: u64) {
        *dataset_frame_id += frame_skip;
        *logical_sequence_id += 1;

        if *dataset_frame_id > MAX_FRAME_SEQUENCE {
            *dataset_frame_id = 1;
        }
    }

    fn load_frame_from_image(&self, frame_number: u64) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
        let filename = format!("pi-sender/sample/seq3-drone_{:07}.jpg", frame_number);
        debug!("Attempting to load frame from: {}", filename);
        let data = std::fs::read(&filename)?;
        debug!("Loaded frame {} ({} bytes)", frame_number, data.len());
        Ok(data)
    }

    async fn cleanup(&self) {
        self.connection_manager.disconnect_all().await;
        info!("Pi Sender stopped");
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn error::Error + Send + Sync>> {
    if std::env::var("RUST_LOG").is_err() {
        unsafe { std::env::set_var("RUST_LOG", "info"); }
    }
    env_logger::init();

    let mut service = PiService::new().await?;
    service.start().await?;

    Ok(())
}