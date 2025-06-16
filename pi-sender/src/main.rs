use shared::{
    current_timestamp_micros, receive_message, send_message, send_result_to_controller,
    ControlMessage, ExperimentConfig, ExperimentMode, InferenceResult, NetworkMessage,
    TimingPayload, PersistentPythonDetector, perform_python_inference_with_counts,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::net::{TcpListener, TcpStream};
use tokio::time::{sleep, Duration, Instant};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("Pi Sender starting...");

    let listener = TcpListener::bind("0.0.0.0:8080").await?;
    println!("Listening for controller on port 8080");

    let should_shutdown = Arc::new(AtomicBool::new(false));

    while !should_shutdown.load(Ordering::Relaxed) {
        let (mut stream, addr) = listener.accept().await?;
        println!("Controller connected from: {}", addr);

        let message = receive_message::<NetworkMessage>(&mut stream).await?;

        match message {
            NetworkMessage::Control(ControlMessage::StartExperiment { config }) => {
                println!(
                    "Starting experiment: {} in mode {:?}",
                    config.experiment_id, config.mode
                );

                match config.mode {
                    ExperimentMode::LocalOnly => {
                        run_local_experiment(config).await?;
                    }
                    ExperimentMode::Offload => {
                        run_offload_experiment(config).await?;
                    }
                }
            }
            NetworkMessage::Control(ControlMessage::Shutdown) => {
                should_shutdown.store(true, Ordering::Relaxed);
                break;
            }
            _ => {}
        }
    }

    println!("Pi Sender stopped");
    Ok(())
}

async fn run_local_experiment(
    config: ExperimentConfig,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("Running LOCAL experiment - Pi processes frames and sends results to controller");

    // Initialize Python detector once for the experiment
    let mut detector = PersistentPythonDetector::new(config.model_name.clone())
        .map_err(|e| format!("Failed to initialize Python detector: {}", e))?;

    println!("Python detector initialized with model: {}", config.model_name);

    let experiment_start = Instant::now();
    let frame_interval = Duration::from_secs_f32(1.0 / config.fixed_fps);
    let mut sequence_id = 1u64;

    while experiment_start.elapsed().as_secs() < config.duration_seconds {
        let mut timing = TimingPayload::new(sequence_id);

        let frame_data = load_frame_from_image(sequence_id)?;
        timing.add_frame_data(frame_data, 1920, 1080);

        // Use actual Python inference instead of mock
        let (inference_result, counts) = process_locally_with_python(&timing, &mut detector).await
            .map_err(|e| format!("Python inference failed: {}", e))?;

        let processing_time = inference_result.processing_time_us;
        send_result_to_controller(&timing, inference_result).await?;

        println!(
            "Processed frame {} locally: {} vehicles, {} pedestrians, {:.1}ms",
            sequence_id,
            counts.total_vehicles,
            counts.pedestrians,
            processing_time as f64 / 1000.0
        );

        sequence_id += 1;

        if sequence_id > 900 {
            sequence_id = 1;
        }

        sleep(frame_interval).await;
    }

    // Cleanup detector
    detector.shutdown().map_err(|e| format!("Failed to shutdown detector: {}", e))?;
    println!("Python detector shut down");

    Ok(())
}

async fn run_offload_experiment(
    config: ExperimentConfig,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("Running OFFLOAD experiment - Pi sends frames to Jetson");

    let jetson_addr = "localhost:9092";
    let mut jetson_stream = TcpStream::connect(jetson_addr).await?;
    println!("Connected to Jetson at {}", jetson_addr);

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

        println!("Sent frame {} to Jetson", sequence_id);
        sequence_id += 1;

        if sequence_id > 900 {
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
    // This function now uses the actual Python detector
    perform_python_inference_with_counts(timing, detector).await
}

fn load_frame_from_image(
    frame_number: u64,
) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    let filename = format!("pi-sender/sample/seq3-drone_{:07}.jpg", frame_number);
    Ok(std::fs::read(&filename)?)
}