use log::{debug, error, info, warn};
use shared::constants::{jetson_bind_address, JETSON_PORT, CONTROLLER_ADDRESS, CONTROLLER_PORT};
use shared::{
    current_timestamp_micros, perform_python_inference_with_counts, receive_message,
    send_result_to_controller, send_message, ControlMessage, ExperimentConfig, InferenceResult, NetworkMessage,
    PersistentPythonDetector, TimingPayload,
};
use std::error;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;

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

    while !should_shutdown.load(Ordering::Relaxed) {
        let (stream, addr) = controller_listener.accept().await?;
        info!("New connection from: {}", addr);

        let shutdown_flag = Arc::clone(&should_shutdown);

        tokio::spawn(async move {
            if let Err(e) = handle_connection(stream, shutdown_flag, addr).await {
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
) -> Result<(), String> {
    let mut detector: Option<PersistentPythonDetector> = None;
    let mut current_config: Option<ExperimentConfig> = None;
    let mut experiment_started = false;
    let mut frames_processed = 0u64;

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

                // Phase 1: Initialize detector (YOLOv5 handles warmup automatically)
                match PersistentPythonDetector::new(config.model_name.clone()) {
                    Ok(new_detector) => {
                        detector = Some(new_detector);
                        info!("Detector initialized and preheated");
                    }
                    Err(e) => {
                        error!("Failed to initialize detector: {}", e);
                        return Err(format!("Failed to initialize detector: {}", e));
                    }
                }

                current_config = Some(config);

                // Phase 2: Signal preheating complete
                if let Err(e) = send_preheating_complete().await {
                    error!("Failed to send preheating complete: {}", e);
                    return Err(format!("Failed to send preheating complete: {}", e));
                }
            }
            Ok(NetworkMessage::Control(ControlMessage::BeginExperiment)) => {
                info!("Received experiment start signal from {}!", addr);
                experiment_started = true;
            }
            Ok(NetworkMessage::Control(ControlMessage::Shutdown)) => {
                info!("Shutdown requested from {}", addr);

                if let Some(mut det) = detector.take() {
                    let _ = det.shutdown();
                    debug!("Detector shut down");
                }
                should_shutdown.store(true, Ordering::Relaxed);
                break;
            }
            Ok(NetworkMessage::Frame(timing)) => {
                info!("Received frame {} from {} (experiment_started: {})", 
                      timing.sequence_id, addr, experiment_started);

                if let Some(ref config) = current_config {
                    let model_name = config.model_name.clone();

                    if let Some(ref mut det) = detector {
                        match process_frame_and_send_result(timing, &model_name, det).await {
                            Ok(()) => {
                                frames_processed += 1;
                                info!("Successfully processed frame (total: {})", frames_processed);
                            },
                            Err(e) => {
                                error!("Error processing frame: {}", e);
                            }
                        }
                    } else {
                        error!("Frame received but detector not initialized");
                    }
                } else {
                    error!("Frame received but no experiment config set");
                }
            }
            Ok(other_msg) => {
                debug!("Received unexpected message type from {}: {:?}", addr, other_msg);
            }
            Err(e) => {
                info!("Connection from {} ended: {}", addr, e);
                break;
            }
        }
    }

    if let Some(mut det) = detector {
        let _ = det.shutdown();
    }

    info!("Connection handler for {} finished. Processed {} frames total.", addr, frames_processed);
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
    detector: &mut PersistentPythonDetector,
) -> Result<(), String> {
    timing.jetson_received = Some(current_timestamp_micros());
    timing.jetson_inference_start = Some(current_timestamp_micros());

    info!("Starting inference for frame {}", timing.sequence_id);

    let inference_result = perform_inference(&timing, model_name, detector)
        .await
        .map_err(|e| {
            error!("Inference failed for frame {}: {}", timing.sequence_id, e);
            e.to_string()
        })?;

    timing.jetson_inference_complete = Some(current_timestamp_micros());
    timing.jetson_sent_result = Some(current_timestamp_micros());

    info!("Sending result for frame {} to controller", timing.sequence_id);

    send_result_to_controller(&timing, inference_result)
        .await
        .map_err(|e| {
            error!("Failed to send result to controller for frame {}: {}", timing.sequence_id, e);
            e.to_string()
        })?;

    info!("Successfully processed and sent result for frame {}", timing.sequence_id);

    Ok(())
}

async fn perform_inference(
    timing: &TimingPayload,
    model_name: &str,
    detector: &mut PersistentPythonDetector,
) -> Result<InferenceResult, String> {
    info!("Performing inference for frame {} with model {}", timing.sequence_id, model_name);

    let (inference_result, counts) = perform_python_inference_with_counts(timing, detector, model_name, "offload").await?;

    info!(
        "Frame {} inference complete: {} vehicles, {} pedestrians, {:.1}ms",
        timing.sequence_id,
        counts.total_vehicles,
        counts.pedestrians,
        inference_result.processing_time_us as f64 / 1000.0
    );

    Ok(inference_result)
}