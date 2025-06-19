use std::error;
use shared::{
    current_timestamp_micros, receive_message, send_message, send_result_to_controller,
    ControlMessage, ExperimentConfig, ExperimentMode, InferenceResult, NetworkMessage,
    TimingPayload, PersistentPythonDetector, perform_python_inference_with_counts,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::net::{TcpListener, TcpStream};
use tokio::time::{sleep, Duration, Instant};
use log::{info, debug, error, warn};
use shared::constants::{jetson_full_address, pi_bind_address, FRAME_HEIGHT, FRAME_WIDTH, MAX_FRAME_SEQUENCE, PI_PORT, CONTROLLER_PORT};

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
                info!(
                    "Starting preheating for experiment: {} in mode {:?} with model {}",
                    config.experiment_id, config.mode, config.model_name
                );

                match config.mode {
                    ExperimentMode::LocalOnly => {
                        run_local_experiment_with_preheating(config, &listener).await?;
                    }
                    ExperimentMode::Offload => {
                        run_offload_experiment_with_preheating(config, &listener).await?;
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
    listener: &TcpListener,
) -> Result<(), Box<dyn error::Error + Send + Sync>> {
    info!("LOCAL MODE: Starting preheating phase...");

    // Phase 1: Preheating - Initialize detector (YOLOv5 will handle warmup automatically)
    let detector = PersistentPythonDetector::new(config.model_name.clone())
        .map_err(|e| {
            error!("Failed to initialize detector: {}", e);
            format!("Failed to initialize Python detector: {}", e)
        })?;

    info!("Detector initialized and preheated");

    // Phase 2: Signal preheating complete
    send_preheating_complete().await?;

    // Phase 3: Wait for experiment start signal (reuse existing listener)
    wait_for_experiment_start(listener).await?;

    // Phase 4: Run actual experiment
    info!("Starting LOCAL experiment - Pi processes frames locally");
    run_local_experiment_loop(config, detector).await?;

    Ok(())
}

async fn run_offload_experiment_with_preheating(
    config: ExperimentConfig,
    listener: &TcpListener,
) -> Result<(), Box<dyn error::Error + Send + Sync>> {
    info!("OFFLOAD MODE: Starting preheating phase...");

    // Phase 1: Signal preheating complete (Pi doesn't need to load model in offload mode)
    info!("Pi ready for offload mode");
    send_preheating_complete().await?;

    // Phase 2: Wait for experiment start signal (reuse existing listener)
    wait_for_experiment_start(listener).await?;

    // Phase 3: Run actual experiment
    info!("Starting OFFLOAD experiment - Pi sends frames to Jetson");
    run_offload_experiment_loop(config).await?;

    Ok(())
}

async fn wait_for_experiment_start(listener: &TcpListener) -> Result<(), Box<dyn error::Error + Send + Sync>> {
    info!("Waiting for experiment start signal...");

    let (mut stream, _addr) = listener.accept().await?;
    let message = receive_message::<NetworkMessage>(&mut stream).await?;

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

async fn run_local_experiment_loop(
    config: ExperimentConfig,
    mut detector: PersistentPythonDetector,
) -> Result<(), Box<dyn error::Error + Send + Sync>> {
    let experiment_start = Instant::now();
    let frame_interval = Duration::from_secs_f32(1.0 / config.fixed_fps);
    let mut sequence_id = 1u64;

    while experiment_start.elapsed().as_secs() < config.duration_seconds {
        let mut timing = TimingPayload::new(sequence_id);

        let frame_data = load_frame_from_image(sequence_id)?;
        timing.add_frame_data(frame_data, FRAME_WIDTH, FRAME_HEIGHT);

        let (inference_result, counts) = process_locally_with_python(&timing, &mut detector).await
            .map_err(|e| {
                error!("Python inference failed: {}", e);
                format!("Python inference failed: {}", e)
            })?;

        let processing_time = inference_result.processing_time_us;
        send_result_to_controller(&timing, inference_result).await?;

        debug!(
            "Frame {} processed: {} vehicles, {} pedestrians, {:.1}ms",
            sequence_id,
            counts.total_vehicles,
            counts.pedestrians,
            processing_time as f64 / 1000.0
        );

        sequence_id += 1;

        if sequence_id > MAX_FRAME_SEQUENCE {
            sequence_id = 1;
        }

        sleep(frame_interval).await;
    }

    detector.shutdown().map_err(|e| {
        error!("Failed to shutdown detector: {}", e);
        format!("Failed to shutdown detector: {}", e)
    })?;

    debug!("Detector shut down");
    Ok(())
}

async fn run_offload_experiment_loop(
    config: ExperimentConfig,
) -> Result<(), Box<dyn error::Error + Send + Sync>> {
    let mut jetson_stream = TcpStream::connect(jetson_full_address()).await?;
    info!("Connected to Jetson at {}", jetson_full_address());

    let experiment_start = Instant::now();
    let frame_interval = Duration::from_secs_f32(1.0 / config.fixed_fps);
    let mut sequence_id = 1u64;

    while experiment_start.elapsed().as_secs() < config.duration_seconds {
        let mut timing = TimingPayload::new(sequence_id);

        let frame_data = load_frame_from_image(sequence_id)?;
        timing.add_frame_data(frame_data, 1920, 1080);

        timing.pi_sent_to_jetson = Some(current_timestamp_micros());

        let frame_message = NetworkMessage::Frame(timing);
        send_message(&mut jetson_stream, &frame_message).await?;

        debug!("Sent frame {} to Jetson", sequence_id);
        sequence_id += 1;

        if sequence_id > MAX_FRAME_SEQUENCE {
            sequence_id = 1;
        }

        sleep(frame_interval).await;
    }

    Ok(())
}

async fn process_locally_with_python(
    timing: &TimingPayload,
    detector: &mut PersistentPythonDetector,
) -> Result<(InferenceResult, shared::ObjectCounts), String> {
    perform_python_inference_with_counts(timing, detector).await
}

fn load_frame_from_image(
    frame_number: u64,
) -> Result<Vec<u8>, Box<dyn error::Error + Send + Sync>> {
    let filename = format!("pi-sender/sample/seq3-drone_{:07}.jpg", frame_number);
    Ok(std::fs::read(&filename)?)
}