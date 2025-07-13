use log::{debug, error, info};
use shared::constants::{INFERENCE_TENSORRT_PATH, JETSON_PORT};
use shared::{
    current_timestamp_micros, perform_python_inference_with_counts, receive_message, send_message,
    ControlMessage, ExperimentConfig, InferenceResult, NetworkMessage,
    PersistentPythonDetector, TimingPayload, ProcessingResult,
};
use std::error;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;
use tokio::time::{sleep, Duration};

#[derive(Clone)]
struct SharedState {
    config: Arc<Mutex<Option<ExperimentConfig>>>,
    detector: Arc<Mutex<Option<PersistentPythonDetector>>>,
    experiment_started: Arc<AtomicBool>,
    frames_processed: Arc<Mutex<u64>>,
    controller_stream: Arc<Mutex<Option<TcpStream>>>,
}

impl SharedState {
    fn new() -> Self {
        Self {
            config: Arc::new(Mutex::new(None)),
            detector: Arc::new(Mutex::new(None)),
            experiment_started: Arc::new(AtomicBool::new(false)),
            frames_processed: Arc::new(Mutex::new(0)),
            controller_stream: Arc::new(Mutex::new(None)),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn error::Error>> {
    if std::env::var("RUST_LOG").is_err() {
        unsafe { std::env::set_var("RUST_LOG", "info"); }
    }
    env_logger::init();

    info!("Jetson Receiver starting...");

    let should_shutdown = Arc::new(AtomicBool::new(false));
    let shared_state = SharedState::new();

    // do we need all this data on controller connection??
    let controller_shutdown = Arc::clone(&should_shutdown);
    let controller_state = shared_state.clone();

    let pi_shutdown = Arc::clone(&should_shutdown);
    let pi_state = shared_state.clone();

    tokio::select! {
        result = handle_controller_connections(controller_shutdown, controller_state) => {
            if let Err(e) = result {
                error!("Controller connection handler failed: {}", e);
            }
        }
        result = handle_pi_connections(pi_shutdown, pi_state) => {
            if let Err(e) = result {
                error!("Pi connection handler failed: {}", e);
            }
        }
    }

    info!("Jetson Receiver stopped");
    Ok(())
}

async fn handle_controller_connections(
    should_shutdown: Arc<AtomicBool>,
    state: SharedState,
) -> Result<(), Box<dyn error::Error>> {
    while !should_shutdown.load(Ordering::Relaxed) {
        info!("Connecting to controller...");
        let stream = match TcpStream::connect(shared::constants::controller_address()).await {
            Ok(stream) => {
                info!("Connected to controller");
                stream
            }
            Err(e) => {
                error!("Failed to connect to controller: {}", e);
                sleep(Duration::from_secs(5)).await;
                continue;
            }
        };
        let mut controller_guard = state.controller_stream.lock().await;
        *controller_guard = Some(stream);
        
        let shutdown_flag = Arc::clone(&should_shutdown);
        let local_state = state.clone();

        if let Err(e) = handle_controller_connection(shutdown_flag, local_state).await {
            error!("Controller connection error: {}", e);
            let mut controller_guard = state.controller_stream.lock().await;
            *controller_guard = None;
        }
    }
    Ok(())
}

async fn handle_controller_connection(
    should_shutdown: Arc<AtomicBool>,
    state: SharedState,
) -> Result<(), String> {
    loop {
        if should_shutdown.load(Ordering::Relaxed) {
            break;
        }

        let mut stream_opt = {
            let mut controller_guard = state.controller_stream.lock().await;
            controller_guard.take()
        };

        if let Some(ref mut stream) = stream_opt {
            let message_result = receive_message::<NetworkMessage>(stream).await;

            {
                let mut controller_guard = state.controller_stream.lock().await;
                *controller_guard = stream_opt;
            }

            match message_result {
                Ok(NetworkMessage::Control(ControlMessage::StartExperiment { config })) => {
                    info!(
                        "Starting preheating for experiment: {} with model {}",
                        config.experiment_id, config.model_name
                    );

                    let new_detector = match PersistentPythonDetector::new(
                        config.model_name.clone(),
                        INFERENCE_TENSORRT_PATH.to_string()
                    ) {
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

                    let mut controller_guard = state.controller_stream.lock().await;
                    if let Some(ref mut stream) = *controller_guard {
                        if let Err(e) = send_message(stream, &ControlMessage::PreheatingComplete).await {
                            error!("Failed to send preheating complete: {}", e);
                            return Err(format!("Failed to send preheating complete: {}", e));
                        }
                        info!("Sent preheating complete to controller");
                    }
                }
                Ok(NetworkMessage::Control(ControlMessage::BeginExperiment)) => {
                    info!("Received experiment start signal!");
                    state.experiment_started.store(true, Ordering::Relaxed);
                }
                Ok(NetworkMessage::Control(ControlMessage::Shutdown)) => {
                    info!("Shutdown requested");

                    let mut detector_guard = state.detector.lock().await;
                    if let Some(mut det) = detector_guard.take() {
                        let _ = det.shutdown();
                        debug!("Detector shut down");
                    }

                    should_shutdown.store(true, Ordering::Relaxed);
                    break;
                }
                Ok(_) => {
                    debug!("Received unexpected message type from controller");
                }
                Err(e) => {
                    info!("Controller connection ended: {}", e);
                    break;
                }
            }
        } else {
            sleep(Duration::from_millis(100)).await;
        }
    }

    Ok(())
}

async fn handle_pi_connections(
    should_shutdown: Arc<AtomicBool>,
    state: SharedState,
) -> Result<(), Box<dyn error::Error>> {
    let listener = TcpListener::bind(format!("0.0.0.0:{}", JETSON_PORT)).await?;
    info!("Jetson listening for Pi connections on port {}", JETSON_PORT);

    while !should_shutdown.load(Ordering::Relaxed) {
        tokio::select! {
            accept_result = listener.accept() => {
                match accept_result {
                    Ok((stream, addr)) => {
                        info!("Pi connected from: {}", addr);
                        
                        let shutdown_flag = Arc::clone(&should_shutdown);
                        let local_state = state.clone();
                        
                        tokio::spawn(async move {
                            if let Err(e) = handle_pi_connection(stream, shutdown_flag, local_state).await {
                                error!("Pi connection error: {}", e);
                            }
                        });
                    }
                    Err(e) => {
                        error!("Failed to accept Pi connection: {}", e);
                    }
                }
            }
            _ = sleep(Duration::from_millis(100)) => {
                // Continue listening
            }
        }
    }

    Ok(())
}

async fn handle_pi_connection(
    mut pi_stream: TcpStream,
    should_shutdown: Arc<AtomicBool>,
    state: SharedState,
) -> Result<(), String> {
    while !should_shutdown.load(Ordering::Relaxed) {
        let message_result = receive_message::<NetworkMessage>(&mut pi_stream).await;

        match message_result {
            Ok(NetworkMessage::Frame(timing)) => {
                if !state.experiment_started.load(Ordering::Relaxed) {
                    debug!("Frame received but experiment not started yet, ignoring");
                    continue;
                }

                let config_opt = {
                    let config_guard = state.config.lock().await;
                    config_guard.clone()
                };

                if let Some(config) = config_opt {
                    let model_name = config.model_name.clone();

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
                    error!("Frame received but no experiment config set");
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

    Ok(())
}

async fn process_frame_and_send_result(
    mut timing: TimingPayload,
    model_name: &str,
    state: &SharedState,
) -> Result<(), String> {
    timing.jetson_received = Some(current_timestamp_micros());
    timing.jetson_inference_start = Some(current_timestamp_micros());

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

    let result = ProcessingResult {
        timing: timing.clone(),
        inference: inference_result,
    };
    let result_message = ControlMessage::ProcessingResult(result);

    let mut controller_guard = state.controller_stream.lock().await;
    if let Some(ref mut controller_stream) = *controller_guard {
        send_message(controller_stream, &result_message).await
            .map_err(|e| {
                error!("Failed to send result to controller for frame {}: {}", timing.sequence_id, e);
                e.to_string()
            })?;
    } else {
        return Err("No controller connection available".to_string());
    }

    Ok(())
}

async fn perform_inference(
    timing: &TimingPayload,
    model_name: &str,
    detector: &mut PersistentPythonDetector,
) -> Result<InferenceResult, String> {
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