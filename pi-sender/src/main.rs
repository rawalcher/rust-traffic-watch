use shared::{
    current_timestamp_micros, receive_message, send_message, send_result_to_controller,
    ControlMessage, Detection, ExperimentConfig, ExperimentMode, InferenceResult, NetworkMessage,
    TimingPayload,
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

    let experiment_start = Instant::now();
    let frame_interval = Duration::from_secs_f32(1.0 / config.fixed_fps);
    let mut sequence_id = 1u64;

    while experiment_start.elapsed().as_secs() < config.duration_seconds {
        let mut timing = TimingPayload::new(sequence_id);

        let frame_data = load_frame_from_image(sequence_id)?;
        timing.add_frame_data(frame_data, 1920, 1080);

        let inference_result = process_locally(&timing, &config.model_name).await?;

        send_result_to_controller(&timing, inference_result).await?;

        println!("Processed frame {} locally", sequence_id);
        sequence_id += 1;

        if sequence_id > 900 {
            sequence_id = 1;
        }

        sleep(frame_interval).await;
    }

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

async fn process_locally(
    timing: &TimingPayload,
    model_name: &str,
) -> Result<InferenceResult, Box<dyn std::error::Error + Send + Sync>> {
    let (detections, actual_processing_time_us) = get_mock_pi_inference_data(model_name).await;

    Ok(InferenceResult {
        sequence_id: timing.sequence_id,
        detections,
        confidence: 0.89,
        processing_time_us: actual_processing_time_us,
    })
}

async fn get_mock_pi_inference_data(model_name: &str) -> (Vec<Detection>, u64) {
    let processing_start = std::time::Instant::now();

    let mock_time_us = match model_name {
        "yolov5n" => 50_000,
        "yolov5s" => 100_000,
        "yolov5m" => 200_000,
        "yolov5l" => 300_000,
        "yolov5x" => 500_000,
        _ => 150_000,
    };

    sleep(Duration::from_micros(mock_time_us)).await;

    let detections = vec![
        Detection {
            class: "person".to_string(),
            bbox: [150.0, 100.0, 30.0, 80.0],
            confidence: 0.92,
        },
        Detection {
            class: "car".to_string(),
            bbox: [100.0, 200.0, 50.0, 30.0],
            confidence: 0.85,
        },
    ];

    (detections, processing_start.elapsed().as_micros() as u64)
}

fn load_frame_from_image(
    frame_number: u64,
) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    let filename = format!("pi-sender/sample/seq3-drone_{:07}.jpg", frame_number);
    Ok(std::fs::read(&filename)?)
}
