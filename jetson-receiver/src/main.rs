use log::{debug, error, info};
use shared::constants::{jetson_bind_address, CONTROLLER_ADDRESS, CONTROLLER_PORT, INFERENCE_TENSORRT_PATH, JETSON_PORT};
use shared::{
    current_timestamp_micros, perform_python_inference_with_counts, receive_message, send_message,
    send_result_to_controller, ControlMessage, ExperimentConfig, InferenceResult, NetworkMessage,
    PersistentPythonDetector, TimingPayload,
};
use std::error;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;

#[derive(Clone)]
struct SharedState {
    config: Arc<Mutex<Option<ExperimentConfig>>>,
    detector: Arc<Mutex<Option<PersistentPythonDetector>>>,
    experiment_started: Arc<AtomicBool>,
    frames_processed: Arc<Mutex<u64>>,
}

impl SharedState {
    fn new() -> Self {
        Self {
            config: Arc::new(Mutex::new(None)),
            detector: Arc::new(Mutex::new(None)),
            experiment_started: Arc::new(AtomicBool::new(false)),
            frames_processed: Arc::new(Mutex::new(0)),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn error::Error>> {
    if std::env::var("RUST_LOG").is_err() {
        unsafe {
            std::env::set_var("RUST_LOG", "info");
        }
    }
    env_logger::init();

    info!("Jetson Coordinator starting...");

    let controller_listener = match TcpListener::bind(&jetson_bind_address()).await {
        Ok(listener) => {
            info!("Listening on Port {}", JETSON_PORT);
            listener
        }
        Err(_) => {
            info!("Fallback to 127.0.0.1:9092");
            TcpListener::bind("127.0.0.1:9092").await?
        }
    };

    let should_shutdown = Arc::new(AtomicBool::new(false));
    let shared_state = SharedState::new();

    while !should_shutdown.load(Ordering::Relaxed) {
        let (stream, addr) = controller_listener.accept().await?;
        info!("New connection from: {}", addr);

        let shutdown_flag = Arc::clone(&should_shutdown);
        let state = shared_state.clone();

        tokio::spawn(async move {
            if let Err(e) = handle_connection(stream, shutdown_flag, addr, state).await {
                error!("Connection error from {}: {}", addr, e);
            }
        });
    }

    info!("Jetson Coordinator stopped");
    Ok(())
}

async fn handle_connection(
    mut stream: TcpStream,
    should_shutdown: Arc<AtomicBool>,
    addr: std::net::SocketAddr,
    state: SharedState,
) -> Result<(), String> {
    loop {
        if should_shutdown.load(Ordering::Relaxed) {
            break;
        }

        let message_result = receive_message::<NetworkMessage>(&mut stream).await;

        match message_result {
            Ok(NetworkMessage::Control(ControlMessage::StartExperiment { config })) => {
                info!(
                    "Starting preheating for experiment: {} with model {} (from {})",
                    config.experiment_id, config.model_name, addr
                );

                // Phase 1: Initialize detector
                let new_detector = match PersistentPythonDetector::new(config.model_name.clone(), INFERENCE_TENSORRT_PATH.to_string()) {
                    Ok(detector) => {
                        info!("Detector initialized and preheated");
                        detector
                    }
                    Err(e) => {
                        error!("Failed to initialize detector: {}", e);
                        return Err(format!("Failed to initialize detector: {}", e));
                    }
                };

                {
                    let mut detector_guard = state.detector.lock().await;
                    *detector_guard = Some(new_detector);
                }

                {
                    let mut config_guard = state.config.lock().await;
                    *config_guard = Some(config);
                }

                // Phase 2: Signal preheating complete
                if let Err(e) = send_preheating_complete().await {
                    error!("Failed to send preheating complete: {}", e);
                    return Err(format!("Failed to send preheating complete: {}", e));
                }
            }
            Ok(NetworkMessage::Control(ControlMessage::BeginExperiment)) => {
                info!("Received experiment start signal from {}!", addr);
                state.experiment_started.store(true, Ordering::Relaxed);
            }
            Ok(NetworkMessage::Control(ControlMessage::Shutdown)) => {
                info!("Shutdown requested from {}", addr);

                let mut detector_guard = state.detector.lock().await;
                if let Some(mut det) = detector_guard.take() {
                    let _ = det.shutdown();
                    debug!("Detector shut down");
                }

                should_shutdown.store(true, Ordering::Relaxed);
                break;
            }
            Ok(NetworkMessage::Frame(timing)) => {
                let experiment_started = state.experiment_started.load(Ordering::Relaxed);
                debug!(
                    "Received frame {} from {} (experiment_started: {})",
                    timing.sequence_id, addr, experiment_started
                );

                let config_opt = {
                    let config_guard = state.config.lock().await;
                    config_guard.clone()
                };

                if let Some(config) = config_opt {
                    let model_name = config.model_name.clone();

                    // Process the frame
                    match process_frame_and_send_result(timing, &model_name, &state).await {
                        Ok(()) => {
                            let mut frames_guard = state.frames_processed.lock().await;
                            *frames_guard += 1;
                            debug!("Successfully processed frame (total: {})", *frames_guard);
                        }
                        Err(e) => {
                            error!("Error processing frame: {}", e);
                        }
                    }
                } else {
                    error!("Frame received but no experiment config set in shared state");
                }
            }
            Ok(other_msg) => {
                debug!(
                    "Received unexpected message type from {}: {:?}",
                    addr, other_msg
                );
            }
            Err(e) => {
                info!("Connection from {} ended: {}", addr, e);
                break;
            }
        }
    }

    let frames_processed = {
        let frames_guard = state.frames_processed.lock().await;
        *frames_guard
    };

    debug!(
        "Connection handler for {} finished. Total frames processed: {}",
        addr, frames_processed
    );
    Ok(())
}

async fn send_preheating_complete() -> Result<(), Box<dyn error::Error + Send + Sync>> {
    let controller_addr = format!("{}:{}", CONTROLLER_ADDRESS, CONTROLLER_PORT);
    let mut controller_stream = TcpStream::connect(&controller_addr).await?;
    let message = ControlMessage::PreheatingComplete;
    send_message(&mut controller_stream, &message).await?;

    info!("Sent preheating complete signal to controller");
    Ok(())
}

async fn process_frame_and_send_result(
    mut timing: TimingPayload,
    model_name: &str,
    state: &SharedState,
) -> Result<(), String> {
    timing.jetson_received = Some(current_timestamp_micros());
    timing.jetson_inference_start = Some(current_timestamp_micros());

    debug!("Starting inference for frame {}", timing.sequence_id);

    let inference_result = {
        let mut detector_guard = state.detector.lock().await;
        if let Some(ref mut detector) = *detector_guard {
            perform_inference(&timing, model_name, detector).await?
        } else {
            return Err("No detector available".to_string());
        }
    };

    timing.jetson_inference_complete = Some(current_timestamp_micros());
    timing.jetson_sent_result = Some(current_timestamp_micros());

    debug!(
        "Sending result for frame {} to controller",
        timing.sequence_id
    );

    send_result_to_controller(&timing, inference_result)
        .await
        .map_err(|e| {
            error!(
                "Failed to send result to controller for frame {}: {}",
                timing.sequence_id, e
            );
            e.to_string()
        })?;

    debug!(
        "Successfully processed and sent result for frame {}",
        timing.sequence_id
    );

    Ok(())
}

async fn perform_inference(
    timing: &TimingPayload,
    model_name: &str,
    detector: &mut PersistentPythonDetector,
) -> Result<InferenceResult, String> {
    debug!(
        "Performing inference for frame {} with model {}",
        timing.sequence_id, model_name
    );

    let (inference_result, counts) =
        perform_python_inference_with_counts(timing, detector, model_name, "offload").await?;

    debug!(
        "Frame {} inference complete: {} vehicles, {} pedestrians, {:.1}ms",
        timing.sequence_id,
        counts.total_vehicles,
        counts.pedestrians,
        inference_result.processing_time_us as f64 / 1000.0
    );

    Ok(inference_result)
}
