use shared::{
    current_timestamp_micros, receive_message, send_result_to_controller, ControlMessage,
    Detection, ExperimentConfig, InferenceResult, NetworkMessage, TimingPayload,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Jetson Coordinator starting...");

    let controller_listener = TcpListener::bind("0.0.0.0:9092").await?;
    println!("Listening for controller/Pi on port 9092");

    let should_shutdown = Arc::new(AtomicBool::new(false));
    let current_config = Arc::new(Mutex::new(None::<ExperimentConfig>));

    while !should_shutdown.load(Ordering::Relaxed) {
        let (stream, addr) = controller_listener.accept().await?;
        println!("Connection from: {}", addr);

        let shutdown_flag = Arc::clone(&should_shutdown);
        let config_ref = Arc::clone(&current_config);
        let stream = Arc::new(Mutex::new(stream));

        tokio::spawn(async move {
            let result = handle_connection(stream, shutdown_flag, config_ref).await;
            if let Err(e) = result {
                println!("Connection error: {}", e);
            }
        });
    }

    println!("Jetson Coordinator stopped");
    Ok(())
}

async fn handle_connection(
    stream: Arc<Mutex<TcpStream>>,
    should_shutdown: Arc<AtomicBool>,
    current_config: Arc<Mutex<Option<ExperimentConfig>>>,
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
                println!(
                    "Received experiment config: {} with model {}",
                    config.experiment_id, config.model_name
                );
                *current_config.lock().await = Some(config);
            }
            Ok(NetworkMessage::Control(ControlMessage::Shutdown)) => {
                should_shutdown.store(true, Ordering::Relaxed);
                break;
            }
            Ok(NetworkMessage::Frame(timing)) => {
                let config_guard = current_config.lock().await;
                if let Some(ref config) = *config_guard {
                    let model_name = config.model_name.clone();
                    drop(config_guard);

                    process_frame_and_send_result(timing, &model_name).await?;
                } else {
                    println!("Warning: Received frame but no experiment config set");
                }
            }
            Ok(_) => {
                println!("Received unexpected message type");
            }
            Err(e) => {
                println!("Connection ended: {}", e);
                break;
            }
        }
    }

    Ok(())
}

async fn process_frame_and_send_result(
    mut timing: TimingPayload,
    model_name: &str,
) -> Result<(), String> {
    timing.jetson_received = Some(current_timestamp_micros());
    timing.jetson_inference_start = Some(current_timestamp_micros());

    let inference_result = perform_inference(&timing, model_name)
        .await
        .map_err(|e| e.to_string())?;

    timing.jetson_inference_complete = Some(current_timestamp_micros());
    timing.jetson_sent_result = Some(current_timestamp_micros());

    send_result_to_controller(&timing, inference_result)
        .await
        .map_err(|e| e.to_string())?;

    println!(
        "Processed frame {} with {} and sent result to controller",
        timing.sequence_id, model_name
    );

    Ok(())
}

async fn perform_inference(
    timing: &TimingPayload,
    model_name: &str,
) -> Result<InferenceResult, String> {
    let (detections, actual_processing_time_us) = get_mock_inference_data(model_name).await;

    Ok(InferenceResult {
        sequence_id: timing.sequence_id,
        detections,
        confidence: 0.89,
        processing_time_us: actual_processing_time_us,
    })
}

async fn get_mock_inference_data(model_name: &str) -> (Vec<Detection>, u64) {
    let processing_start = std::time::Instant::now();

    let mock_time_us = match model_name {
        "yolov5n" => 20_000,
        "yolov5s" => 40_000,
        "yolov5m" => 80_000,
        "yolov5l" => 120_000,
        "yolov5x" => 200_000,
        _ => 50_000,
    };

    sleep(Duration::from_micros(mock_time_us)).await;

    let detections = vec![
        Detection {
            class: "vehicle".to_string(),
            bbox: [120.0, 180.0, 60.0, 40.0],
            confidence: 0.88,
        },
        Detection {
            class: "person".to_string(),
            bbox: [200.0, 150.0, 25.0, 70.0],
            confidence: 0.92,
        },
    ];

    (detections, processing_start.elapsed().as_micros() as u64)
}
