use std::error;
use shared::{
    current_timestamp_micros, receive_message, send_message,
    ControlMessage, ExperimentConfig, ExperimentMode, InferenceResult, NetworkMessage,
    TimingPayload, PersistentPythonDetector, perform_python_inference_with_counts,
    ThroughputMode, FrameThroughputController, ProcessingResult,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::net::{TcpListener, TcpStream};
use tokio::time::{sleep, Duration, Instant};
use log::{info, debug, error, warn};
use shared::constants::{jetson_address, pi_address, FRAME_HEIGHT, FRAME_WIDTH, MAX_FRAME_SEQUENCE, PI_PORT, CONTROLLER_PORT, INFERENCE_PYTORCH_PATH, controller_address};

struct PersistentConnections {
    controller_stream: TcpStream,
    jetson_stream: Option<TcpStream>,
    result_stream: Option<TcpStream>,
}

impl PersistentConnections {
    fn new(controller_stream: TcpStream) -> Self {
        Self {
            controller_stream,
            jetson_stream: None,
            result_stream: None,
        }
    }

    async fn connect_to_controller_for_results(&mut self) -> Result<(), Box<dyn error::Error + Send + Sync>> {
        let stream = TcpStream::connect(controller_address()).await?;
        self.result_stream = Some(stream);
        info!("Connected persistent result stream to controller");
        Ok(())
    }

    async fn connect_to_jetson(&mut self) -> Result<(), Box<dyn error::Error + Send + Sync>> {
        let stream = TcpStream::connect(jetson_address()).await?;
        self.jetson_stream = Some(stream);
        info!("Connected persistent stream to Jetson");
        Ok(())
    }

    async fn send_to_jetson(&mut self, message: &NetworkMessage) -> Result<(), Box<dyn error::Error + Send + Sync>> {
        if let Some(ref mut stream) = self.jetson_stream {
            send_message(stream, message).await?;
            Ok(())
        } else {
            Err("No Jetson connection available".into())
        }
    }

    async fn send_result_to_controller(&mut self, timing: &TimingPayload, inference: InferenceResult) -> Result<(), Box<dyn error::Error + Send + Sync>> {
        let result = ProcessingResult {
            timing: timing.clone(),
            inference,
        };
        let message = ControlMessage::ProcessingResult(result);

        if let Some(ref mut stream) = self.result_stream {
            send_message(stream, &message).await?;
            Ok(())
        } else {
            Err("No controller result connection available".into())
        }
    }

    async fn wait_for_control_message(&mut self) -> Result<ControlMessage, Box<dyn error::Error + Send + Sync>> {
        let message = receive_message::<NetworkMessage>(&mut self.controller_stream).await?;
        match message {
            NetworkMessage::Control(control_msg) => Ok(control_msg),
            _ => Err("Expected control message".into()),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn error::Error + Send + Sync>> {
    if std::env::var("RUST_LOG").is_err() {
        unsafe { std::env::set_var("RUST_LOG", "info"); }
    }
    env_logger::init();

    info!("Pi Sender starting...");

    let listener = TcpListener::bind(&pi_address()).await?;
    info!("Listening on port {}", PI_PORT);

    let should_shutdown = Arc::new(AtomicBool::new(false));

    while !should_shutdown.load(Ordering::Relaxed) {
        let (mut controller_stream, addr) = listener.accept().await?;
        info!("Controller connected: {}", addr);

        let message = receive_message::<NetworkMessage>(&mut controller_stream).await?;

        match message {
            NetworkMessage::Control(ControlMessage::StartExperiment { config }) => {
                let throughput_mode = if std::env::args().any(|arg| arg == "--fps-mode") {
                    ThroughputMode::Fps
                } else {
                    ThroughputMode::High
                };

                info!(
                    "Starting experiment: {} in mode {:?} with model {} (throughput: {:?})",
                    config.experiment_id, config.mode, config.model_name, throughput_mode
                );

                let connections = PersistentConnections::new(controller_stream);

                match config.mode {
                    ExperimentMode::LocalOnly => {
                        run_local_experiment(config, connections, throughput_mode).await?;
                    }
                    ExperimentMode::Offload => {
                        run_offload_experiment(config, connections, throughput_mode).await?;
                    }
                }
            }
            NetworkMessage::Control(ControlMessage::Shutdown) => {
                info!("Shutdown requested");
                should_shutdown.store(true, Ordering::Relaxed);
                break;
            }
            _ => {
                debug!("Received unexpected message");
            }
        }
    }

    info!("Pi Sender stopped");
    Ok(())
}

async fn run_local_experiment(
    config: ExperimentConfig,
    mut connections: PersistentConnections,
    throughput_mode: ThroughputMode,
) -> Result<(), Box<dyn error::Error + Send + Sync>> {
    info!("LOCAL MODE: Starting preheating phase...");

    let detector = match PersistentPythonDetector::new(
        config.model_name.clone(),
        INFERENCE_PYTORCH_PATH.to_string()
    ) {
        Ok(detector) => {
            info!("Detector initialized and preheated");
            detector
        }
        Err(e) => {
            error!("Failed to initialize detector: {}", e);
            return Err(format!("Failed to initialize detector: {}", e).into());
        }
    };

    connections.connect_to_controller_for_results().await?;
    send_preheating_complete().await?;

    loop {
        match connections.wait_for_control_message().await? {
            ControlMessage::BeginExperiment => {
                info!("Received experiment start signal!");
                break;
            }
            ControlMessage::Shutdown => {
                info!("Received shutdown during wait");
                return Ok(());
            }
            msg => {
                warn!("Unexpected message while waiting for start: {:?}", msg);
            }
        }
    }

    info!("Starting LOCAL experiment with {:?} mode", throughput_mode);
    local_processing(config, detector, &mut connections, throughput_mode).await?;

    Ok(())
}

async fn run_offload_experiment(
    config: ExperimentConfig,
    mut connections: PersistentConnections,
    throughput_mode: ThroughputMode,
) -> Result<(), Box<dyn error::Error + Send + Sync>> {
    info!("OFFLOAD MODE: Starting preheating phase...");

    connections.connect_to_jetson().await?;
    info!("Pi ready for offload mode");
    send_preheating_complete().await?;

    loop {
        match connections.wait_for_control_message().await? {
            ControlMessage::BeginExperiment => {
                info!("Received experiment start signal!");
                break;
            }
            ControlMessage::Shutdown => {
                info!("Received shutdown during wait");
                return Ok(());
            }
            msg => {
                warn!("Unexpected message while waiting for start: {:?}", msg);
            }
        }
    }

    info!("Starting OFFLOAD experiment with {:?} mode", throughput_mode);
    offloading(config, connections, throughput_mode).await?;

    Ok(())
}

async fn send_preheating_complete() -> Result<(), Box<dyn error::Error + Send + Sync>> {
    let mut controller_stream = TcpStream::connect(controller_address()).await?;
    let message = ControlMessage::PreheatingComplete;
    send_message(&mut controller_stream, &message).await?;

    info!("Sent preheating complete signal to controller");
    Ok(())
}

async fn local_processing(
    config: ExperimentConfig,
    mut detector: PersistentPythonDetector,
    connections: &mut PersistentConnections,
    throughput_mode: ThroughputMode,
) -> Result<(), Box<dyn error::Error + Send + Sync>> {
    let throughput_controller = FrameThroughputController::new(throughput_mode);
    let frame_skip = throughput_controller.get_frame_skip();

    let actual_fps = 30.0 / frame_skip as f32;
    let frame_interval = Duration::from_secs_f32(1.0 / actual_fps);

    info!("Local processing: skipping every {} frames for {:.1} FPS", frame_skip, actual_fps);

    let mut dataset_frame_id = 1u64;
    let mut logical_sequence_id = 1u64;
    let mut frames_processed = 0u64;
    let experiment_start = Instant::now();

    while experiment_start.elapsed().as_secs() < config.duration_seconds {
        let mut timing = TimingPayload::new(logical_sequence_id);

        let frame_data = match load_frame_from_image(dataset_frame_id) {
            Ok(data) => data,
            Err(e) => {
                error!("Failed to load frame {}: {}", dataset_frame_id, e);
                advance_frame_ids(&mut dataset_frame_id, &mut logical_sequence_id, frame_skip);
                sleep(frame_interval).await;
                continue;
            }
        };

        timing.add_frame_data(frame_data, FRAME_WIDTH, FRAME_HEIGHT);

        let (inference_result, counts) = match process_locally(&timing, &mut detector, &config.model_name).await {
            Ok(result) => result,
            Err(e) => {
                error!("Python inference failed for frame {}: {}", dataset_frame_id, e);
                advance_frame_ids(&mut dataset_frame_id, &mut logical_sequence_id, frame_skip);
                sleep(frame_interval).await;
                continue;
            }
        };

        if let Err(e) = connections.send_result_to_controller(&timing, inference_result.clone()).await {
            error!("Failed to send result to controller for frame {}: {}", dataset_frame_id, e);
        }

        frames_processed += 1;
        debug!(
            "Frame {} (dataset frame {}) processed: {} vehicles, {} pedestrians, {:.1}ms",
            logical_sequence_id,
            dataset_frame_id,
            counts.total_vehicles,
            counts.pedestrians,
            inference_result.processing_time_us as f64 / 1000.0
        );

        advance_frame_ids(&mut dataset_frame_id, &mut logical_sequence_id, frame_skip);
        sleep(frame_interval).await;
    }

    detector.shutdown().map_err(|e| format!("Failed to shutdown detector: {}", e))?;
    info!("Local processing completed. Processed: {} frames", frames_processed);
    Ok(())
}

async fn offloading(
    config: ExperimentConfig,
    mut connections: PersistentConnections,
    throughput_mode: ThroughputMode,
) -> Result<(), Box<dyn error::Error + Send + Sync>> {
    let throughput_controller = FrameThroughputController::new(throughput_mode);
    let frame_skip = throughput_controller.get_frame_skip();

    let actual_fps = 30.0 / frame_skip as f32;
    let frame_interval = Duration::from_secs_f32(1.0 / actual_fps);

    info!("Offloading: skipping every {} frames for {:.1} FPS", frame_skip, actual_fps);

    let mut dataset_frame_id = 1u64;
    let mut logical_sequence_id = 1u64;
    let mut frames_sent = 0u64;
    let experiment_start = Instant::now();

    while experiment_start.elapsed().as_secs() < config.duration_seconds {
        let mut timing = TimingPayload::new(logical_sequence_id);

        let frame_data = match load_frame_from_image(dataset_frame_id) {
            Ok(data) => data,
            Err(e) => {
                error!("Failed to load frame {}: {}", dataset_frame_id, e);
                advance_frame_ids(&mut dataset_frame_id, &mut logical_sequence_id, frame_skip);
                sleep(frame_interval).await;
                continue;
            }
        };

        timing.add_frame_data(frame_data, FRAME_WIDTH, FRAME_HEIGHT);
        timing.pi_sent_to_jetson = Some(current_timestamp_micros());

        let frame_message = NetworkMessage::Frame(timing);

        match connections.send_to_jetson(&frame_message).await {
            Ok(()) => {
                frames_sent += 1;
                debug!("Sent frame {} (dataset frame {}) to Jetson (total sent: {})", 
                      logical_sequence_id, dataset_frame_id, frames_sent);
            },
            Err(e) => {
                error!("Failed to send frame {} to Jetson: {}", dataset_frame_id, e);
                // Try to reconnect once
                if let Err(reconnect_err) = connections.connect_to_jetson().await {
                    error!("Failed to reconnect to Jetson: {}", reconnect_err);
                    break;
                } else {
                    warn!("Reconnected to Jetson, continuing...");
                }
            }
        }

        advance_frame_ids(&mut dataset_frame_id, &mut logical_sequence_id, frame_skip);
        sleep(frame_interval).await;
    }

    info!("Offloading completed. Sent: {} frames", frames_sent);
    Ok(())
}

fn advance_frame_ids(dataset_frame_id: &mut u64, logical_sequence_id: &mut u64, frame_skip: u64) {
    *dataset_frame_id += frame_skip;
    *logical_sequence_id += 1;

    if *dataset_frame_id > MAX_FRAME_SEQUENCE {
        *dataset_frame_id = 1;
    }
}

async fn process_locally(
    timing: &TimingPayload,
    detector: &mut PersistentPythonDetector,
    model_name: &str,
) -> Result<(InferenceResult, shared::ObjectCounts), String> {
    perform_python_inference_with_counts(timing, detector, model_name, "local").await
}

fn load_frame_from_image(
    frame_number: u64,
) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    let filename = format!("pi-sender/sample/seq3-drone_{:07}.jpg", frame_number);
    debug!("Attempting to load frame from: {}", filename);

    let data = std::fs::read(&filename)?;
    debug!("Loaded frame {} ({} bytes)", frame_number, data.len());

    Ok(data)
}