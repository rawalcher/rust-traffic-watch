use std::error;
use shared::{
    current_timestamp_micros, receive_message, send_message, send_result_to_controller,
    ControlMessage, ExperimentConfig, ExperimentMode, InferenceResult, NetworkMessage,
    TimingPayload, PersistentPythonDetector, perform_python_inference_with_counts,
    ThroughputMode, FrameThroughputController, // Add these imports
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::net::{TcpListener, TcpStream};
use tokio::time::{sleep, Duration, Instant};
use log::{info, debug, error, warn};
use shared::constants::{jetson_full_address, pi_bind_address, FRAME_HEIGHT, FRAME_WIDTH, MAX_FRAME_SEQUENCE, PI_PORT, CONTROLLER_PORT, INFERENCE_PYTORCH_PATH};

#[tokio::main]
async fn main() -> Result<(), Box<dyn error::Error + Send + Sync>> {
    if std::env::var("RUST_LOG").is_err() {
        unsafe { std::env::set_var("RUST_LOG", "info"); }
    }
    env_logger::init();

    info!("Pi Sender starting...");

    let listener = TcpListener::bind(&pi_bind_address()).await?;
    info!("Listening on port {}", PI_PORT);

    let should_shutdown = Arc::new(AtomicBool::new(false));

    while !should_shutdown.load(Ordering::Relaxed) {
        let (mut stream, addr) = listener.accept().await?;
        info!("Controller connected: {}", addr);

        let message = receive_message::<NetworkMessage>(&mut stream).await?;

        match message {
            NetworkMessage::Control(ControlMessage::StartExperiment { config }) => {
                // Determine throughput mode (you can add command line args)
                let throughput_mode = if std::env::args().any(|arg| arg == "--fps-mode") {
                    ThroughputMode::Fps
                } else {
                    ThroughputMode::High
                };

                info!(
                    "Starting experiment: {} in mode {:?} with model {} (throughput: {:?})",
                    config.experiment_id, config.mode, config.model_name, throughput_mode
                );

                match config.mode {
                    ExperimentMode::LocalOnly => {
                        run_local_experiment_with_preheating(config, stream, throughput_mode).await?;
                    }
                    ExperimentMode::Offload => {
                        run_offload_experiment_with_preheating(config, stream, throughput_mode).await?;
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

async fn run_local_experiment_with_preheating(
    config: ExperimentConfig,
    mut controller_stream: TcpStream,
    throughput_mode: ThroughputMode,
) -> Result<(), Box<dyn error::Error + Send + Sync>> {
    info!("LOCAL MODE: Starting preheating phase...");

    // Phase 1: Preheating - Initialize detector
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

    // Phase 2: Signal preheating complete
    send_preheating_complete().await?;

    // Phase 3: Wait for experiment start signal on the same stream
    wait_for_experiment_start_on_stream(&mut controller_stream).await?;

    // Phase 4: Run actual experiment
    info!("Starting LOCAL experiment with {:?} mode", throughput_mode);
    local_processing(config, detector, throughput_mode).await?;

    Ok(())
}

async fn run_offload_experiment_with_preheating(
    config: ExperimentConfig,
    mut controller_stream: TcpStream,
    throughput_mode: ThroughputMode,
) -> Result<(), Box<dyn error::Error + Send + Sync>> {
    info!("OFFLOAD MODE: Starting preheating phase...");

    // Phase 1: Signal preheating complete (Pi doesn't need to load model in offload mode)
    info!("Pi ready for offload mode");
    send_preheating_complete().await?;

    // Phase 2: Wait for experiment start signal on the same stream
    wait_for_experiment_start_on_stream(&mut controller_stream).await?;

    // Phase 3: Run actual experiment
    info!("Starting OFFLOAD experiment with {:?} mode", throughput_mode);
    offloading(config, throughput_mode).await?;

    Ok(())
}

async fn wait_for_experiment_start_on_stream(stream: &mut TcpStream) -> Result<(), Box<dyn error::Error + Send + Sync>> {
    info!("Waiting for experiment start signal...");

    let message = receive_message::<NetworkMessage>(stream).await?;

    match message {
        NetworkMessage::Control(ControlMessage::BeginExperiment) => {
            info!("Received experiment start signal!");
            Ok(())
        }
        NetworkMessage::Control(ControlMessage::Shutdown) => {
            info!("Received shutdown during wait");
            Err("Shutdown received".into())
        }
        _ => {
            warn!("Unexpected message while waiting for start signal");
            Err("Unexpected message".into())
        }
    }
}

async fn send_preheating_complete() -> Result<(), Box<dyn error::Error + Send + Sync>> {
    let controller_addr = format!("{}:{}",
                                  shared::constants::CONTROLLER_ADDRESS, CONTROLLER_PORT
    );

    let mut controller_stream = TcpStream::connect(&controller_addr).await?;
    let message = ControlMessage::PreheatingComplete;
    send_message(&mut controller_stream, &message).await?;

    info!("Sent preheating complete signal to controller");
    Ok(())
}

async fn local_processing(
    config: ExperimentConfig,
    mut detector: PersistentPythonDetector,
    throughput_mode: ThroughputMode,
) -> Result<(), Box<dyn error::Error + Send + Sync>> {
    let mut throughput_controller = FrameThroughputController::new(throughput_mode);

    let frame_interval = Duration::from_secs_f32(1.0 / config.fixed_fps);
    let mut sequence_id = 1u64;
    let mut frames_processed = 0u64;
    let mut frames_dropped = 0u64;
    let mut experiment_start: Option<Instant> = None;

    loop {
        let mut timing = TimingPayload::new(sequence_id);

        let frame_data = match load_frame_from_image(sequence_id) {
            Ok(data) => data,
            Err(e) => {
                error!("Failed to load frame {}: {}", sequence_id, e);
                sequence_id += 1;
                if sequence_id > MAX_FRAME_SEQUENCE {
                    sequence_id = 1;
                }
                sleep(frame_interval).await;
                continue;
            }
        };

        timing.add_frame_data(frame_data, FRAME_WIDTH, FRAME_HEIGHT);

        if throughput_controller.should_send_frame() {
            // Start timer on first frame process
            if experiment_start.is_none() {
                experiment_start = Some(Instant::now());
                info!("Starting experiment timer on first frame process");
            }

            let (inference_result, counts) = match process_locally_with_python(&timing, &mut detector, &config.model_name).await {
                Ok(result) => result,
                Err(e) => {
                    error!("Python inference failed for frame {}: {}", sequence_id, e);
                    sequence_id += 1;
                    if sequence_id > MAX_FRAME_SEQUENCE {
                        sequence_id = 1;
                    }
                    sleep(frame_interval).await;
                    continue;
                }
            };

            if let Err(e) = send_result_to_controller(&timing, inference_result.clone()).await {
                error!("Failed to send result to controller for frame {}: {}", sequence_id, e);
            }

            frames_processed += 1;
            debug!(
                "Frame {} processed: {} vehicles, {} pedestrians, {:.1}ms",
                sequence_id,
                counts.total_vehicles,
                counts.pedestrians,
                inference_result.processing_time_us as f64 / 1000.0
            );
        } else {
            frames_dropped += 1;
            debug!("Dropped frame {} (FPS mode)", sequence_id);
        }

        // Check if experiment duration is complete (only after first frame)
        if let Some(start_time) = experiment_start {
            if start_time.elapsed().as_secs() >= config.duration_seconds {
                break;
            }
        }

        sequence_id += 1;
        if sequence_id > MAX_FRAME_SEQUENCE {
            sequence_id = 1;
        }

        sleep(frame_interval).await;
    }

    detector.shutdown().map_err(|e| format!("Failed to shutdown detector: {}", e))?;
    info!("Local processing completed. Processed: {}, dropped: {}", frames_processed, frames_dropped);
    Ok(())
}

async fn offloading(
    config: ExperimentConfig,
    throughput_mode: ThroughputMode,
) -> Result<(), Box<dyn error::Error + Send + Sync>> {
    let mut jetson_stream = TcpStream::connect(jetson_full_address()).await?;
    info!("Connected to Jetson at {}", jetson_full_address());

    let mut throughput_controller = FrameThroughputController::new(throughput_mode);

    let frame_interval = Duration::from_secs_f32(1.0 / config.fixed_fps);
    let mut sequence_id = 1u64;
    let mut frames_sent = 0u64;
    let mut frames_dropped = 0u64;
    let mut experiment_start: Option<Instant> = None;

    loop {
        let mut timing = TimingPayload::new(sequence_id);

        let frame_data = match load_frame_from_image(sequence_id) {
            Ok(data) => data,
            Err(e) => {
                error!("Failed to load frame {}: {}", sequence_id, e);
                sequence_id += 1;
                if sequence_id > MAX_FRAME_SEQUENCE {
                    sequence_id = 1;
                }
                sleep(frame_interval).await;
                continue;
            }
        };

        timing.add_frame_data(frame_data, FRAME_WIDTH, FRAME_HEIGHT);

        // Check if we should send this frame
        if throughput_controller.should_send_frame() {
            // Start timer on first frame send
            if experiment_start.is_none() {
                experiment_start = Some(Instant::now());
                info!("Starting experiment timer on first frame send");
            }

            timing.pi_sent_to_jetson = Some(current_timestamp_micros());
            let frame_message = NetworkMessage::Frame(timing);

            match send_message(&mut jetson_stream, &frame_message).await {
                Ok(()) => {
                    frames_sent += 1;
                    debug!("Sent frame {} to Jetson (total sent: {})", sequence_id, frames_sent);
                },
                Err(e) => {
                    error!("Failed to send frame {} to Jetson: {}", sequence_id, e);
                    match TcpStream::connect(jetson_full_address()).await {
                        Ok(new_stream) => {
                            warn!("Reconnected to Jetson");
                            jetson_stream = new_stream;
                        },
                        Err(reconnect_err) => {
                            error!("Failed to reconnect to Jetson: {}", reconnect_err);
                            break;
                        }
                    }
                }
            }
        } else {
            frames_dropped += 1;
            debug!("Dropped frame {} (FPS mode)", sequence_id);
        }

        // Check if experiment duration is complete (only after first frame)
        if let Some(start_time) = experiment_start {
            if start_time.elapsed().as_secs() >= config.duration_seconds {
                break;
            }
        }

        sequence_id += 1;
        if sequence_id > MAX_FRAME_SEQUENCE {
            sequence_id = 1;
        }

        sleep(frame_interval).await;
    }

    info!("Offloading completed. Sent: {}, dropped: {}", frames_sent, frames_dropped);
    Ok(())
}

async fn process_locally_with_python(
    timing: &TimingPayload,
    detector: &mut PersistentPythonDetector,
    model_name: &str,
) -> Result<(InferenceResult, shared::ObjectCounts), String> {
    perform_python_inference_with_counts(timing, detector, model_name, "local").await
}

fn load_frame_from_image(
    frame_number: u64,
) -> Result<Vec<u8>, Box<dyn error::Error + Send + Sync>> {
    let filename = format!("pi-sender/sample/seq3-drone_{:07}.jpg", frame_number);
    debug!("Attempting to load frame from: {}", filename);

    match std::fs::read(&filename) {
        Ok(data) => {
            debug!("Successfully loaded frame {} ({} bytes)", frame_number, data.len());
            Ok(data)
        },
        Err(e) => {
            error!("Failed to read file {}: {}", filename, e);
            Err(Box::new(e))
        }
    }
}