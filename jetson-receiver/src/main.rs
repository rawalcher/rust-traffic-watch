use log::{debug, error, info};
use shared::constants::{jetson_bind_address, JETSON_PORT};
use shared::{
    current_timestamp_micros, perform_python_inference_with_counts, receive_message,
    send_result_to_controller, ControlMessage, ExperimentConfig, InferenceResult, NetworkMessage,
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
    let current_config = Arc::new(Mutex::new(None::<ExperimentConfig>));
    let detector = Arc::new(Mutex::new(None::<PersistentPythonDetector>));

    while !should_shutdown.load(Ordering::Relaxed) {
        let (stream, addr) = controller_listener.accept().await?;
        info!("Connected: {}", addr);

        let shutdown_flag = Arc::clone(&should_shutdown);
        let config_ref = Arc::clone(&current_config);
        let detector_ref = Arc::clone(&detector);
        let stream = Arc::new(Mutex::new(stream));

        tokio::spawn(async move {
            if let Err(e) = handle_connection(stream, shutdown_flag, config_ref, detector_ref).await
            {
                error!("Connection error: {}", e);
            }
        });
    }

    let mut detector_guard = detector.lock().await;
    if let Some(ref mut det) = detector_guard.as_mut() {
        let _ = det.shutdown();
    }

    info!("Jetson Coordinator stopped");
    Ok(())
}

async fn handle_connection(
    stream: Arc<Mutex<TcpStream>>,
    should_shutdown: Arc<AtomicBool>,
    current_config: Arc<Mutex<Option<ExperimentConfig>>>,
    detector: Arc<Mutex<Option<PersistentPythonDetector>>>,
) -> Result<(), String> {
    loop {
        if should_shutdown.load(Ordering::Relaxed) {
            break;
        }

        let message_result = {
            let mut stream_guard = stream.lock().await;
            receive_message::<NetworkMessage>(&mut *stream_guard).await
        };

        match message_result {
            Ok(NetworkMessage::Control(ControlMessage::StartExperiment { config })) => {
                info!(
                    "Starting experiment: {} with model {}",
                    config.experiment_id, config.model_name
                );

                match PersistentPythonDetector::new(config.model_name.clone()) {
                    Ok(new_detector) => {
                        *detector.lock().await = Some(new_detector);
                        debug!("Detector initialized");
                    }
                    Err(e) => {
                        error!("Failed to initialize detector: {}", e);
                        return Err(format!("Failed to initialize detector: {}", e));
                    }
                }
                *current_config.lock().await = Some(config);
            }
            Ok(NetworkMessage::Control(ControlMessage::Shutdown)) => {
                info!("Shutdown requested");

                let mut detector_guard = detector.lock().await;
                if let Some(mut det) = detector_guard.take() {
                    let _ = det.shutdown();
                    debug!("Detector shut down");
                }
                should_shutdown.store(true, Ordering::Relaxed);
                break;
            }
            Ok(NetworkMessage::Frame(timing)) => {
                let config_guard = current_config.lock().await;
                if let Some(ref config) = *config_guard {
                    let model_name = config.model_name.clone();
                    drop(config_guard);

                    let detector_ref = Arc::clone(&detector);
                    process_frame_and_send_result(timing, &model_name, detector_ref).await?;
                } else {
                    debug!("Frame received but no experiment config set");
                }
            }
            Ok(_) => {
                debug!("Received unexpected message type");
            }
            Err(e) => {
                debug!("Connection ended: {}", e);
                break;
            }
        }
    }

    Ok(())
}

async fn process_frame_and_send_result(
    mut timing: TimingPayload,
    model_name: &str,
    detector: Arc<Mutex<Option<PersistentPythonDetector>>>,
) -> Result<(), String> {
    timing.jetson_received = Some(current_timestamp_micros());
    timing.jetson_inference_start = Some(current_timestamp_micros());

    let inference_result = perform_inference(&timing, model_name, detector)
        .await
        .map_err(|e| e.to_string())?;

    timing.jetson_inference_complete = Some(current_timestamp_micros());
    timing.jetson_sent_result = Some(current_timestamp_micros());

    send_result_to_controller(&timing, inference_result)
        .await
        .map_err(|e| e.to_string())?;

    debug!("Processed frame {}", timing.sequence_id);

    Ok(())
}

async fn perform_inference(
    timing: &TimingPayload,
    _model_name: &str,
    detector: Arc<Mutex<Option<PersistentPythonDetector>>>,
) -> Result<InferenceResult, String> {
    let mut detector_guard = detector.lock().await;

    if let Some(ref mut det) = detector_guard.as_mut() {
        let (inference_result, counts) = perform_python_inference_with_counts(timing, det).await?;

        debug!(
            "Frame {} inference: {} vehicles, {} pedestrians, {:.1}ms",
            timing.sequence_id,
            counts.total_vehicles,
            counts.pedestrians,
            inference_result.processing_time_us as f64 / 1000.0
        );

        Ok(inference_result)
    } else {
        Err("Python detector not initialized".to_string())
    }
}
