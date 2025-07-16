
use log::{debug, info, warn};
use shared::constants::*;
use shared::{current_timestamp_micros, perform_python_inference_with_counts, ControlMessage, ExperimentConfig, ExperimentMode, FrameMessage, InferenceMessage, Message, PersistentPythonDetector, TimingMetadata};
use std::error::Error;
use std::time::Duration;
use tokio::net::{TcpStream};
use tokio::net::tcp::{OwnedReadHalf};
use tokio::time::sleep;
use shared::network::{send_message, read_message};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    env_logger::init();
    info!("Connecting to controller at {}", controller_address());

    let controller_stream = TcpStream::connect(controller_address()).await?;
    let (mut ctrl_reader, mut ctrl_writer) = controller_stream.into_split();

    send_message(&mut ctrl_writer, &Message::Hello(shared::types::DeviceId::Pi)).await?;

    let config = wait_for_experiment_config(&mut ctrl_reader).await?;

    // Preheat phase
    match config.mode {
        ExperimentMode::LocalOnly => {
            let mut detector = PersistentPythonDetector::new(
                config.model_name.clone(),
                INFERENCE_PYTORCH_PATH.to_string(),
            )?;

            send_message(&mut ctrl_writer, &Message::Control(ControlMessage::ReadyToStart)).await?;
            wait_for_experiment_start(&mut ctrl_reader).await?;
            info!("Experiment started. Waiting for Pulse messages...");

            loop {
                match read_message(&mut ctrl_reader).await? {
                    Message::Pulse(mut timing) => {
                        timing.pi_capture_start = Some(current_timestamp_micros());
                        let result = handle_pulse_local(timing, &mut detector, &config).await?;
                        send_message(&mut ctrl_writer, &Message::Result(result)).await?;
                    }
                    Message::Control(ControlMessage::Shutdown) => {
                        info!("Shutdown received");
                        break;
                    }
                    msg => warn!("Unexpected message during experiment: {:?}", msg),
                }
            }
            detector.shutdown()?;
        }
        ExperimentMode::Offload => {
            info!("Connecting to Jetson on {}", jetson_address());
            sleep(Duration::from_millis(7500)).await;
            
            let jetson_stream = TcpStream::connect(jetson_address()).await?;
            let (_, mut jetson_writer) = jetson_stream.into_split();
            send_message(&mut jetson_writer, &Message::Hello(shared::types::DeviceId::Pi)).await?;
            
            send_message(&mut ctrl_writer, &Message::Control(ControlMessage::ReadyToStart)).await?;
            wait_for_experiment_start(&mut ctrl_reader).await?;

            info!("Experiment started. Waiting for Pulse messages...");

            loop {
                match read_message(&mut ctrl_reader).await? {
                    Message::Pulse(mut timing) => {
                        timing.pi_capture_start = Some(current_timestamp_micros());
                        let frame = handle_pulse_offload(timing).await?;
                        send_message(&mut jetson_writer, &Message::Frame(frame)).await?;
                    }
                    Message::Control(ControlMessage::Shutdown) => {
                        info!("Shutdown received");
                        break;
                    }
                    msg => warn!("Unexpected message during experiment: {:?}", msg),
                }
            }
        }
    }

    info!("Experiment done. Exiting.");
    Ok(())
}

async fn wait_for_experiment_config(reader: &mut OwnedReadHalf) -> Result<ExperimentConfig, Box<dyn Error + Send + Sync>> {
    loop {
        match read_message(reader).await? {
            Message::Control(ControlMessage::StartExperiment { config }) => {
                info!("Received experiment config");
                return Ok(config);
            }
            Message::Control(ControlMessage::Shutdown) => {
                return Err("Shutdown before experiment started".into());
            }
            msg => warn!("Waiting for config, got: {:?}", msg),
        }
    }
}

async fn wait_for_experiment_start(reader: &mut OwnedReadHalf) -> Result<(), Box<dyn Error + Send + Sync>> {
    loop {
        match read_message(reader).await? {
            Message::Control(ControlMessage::BeginExperiment) => return Ok(()),
            Message::Control(ControlMessage::Shutdown) => return Err("Shutdown during wait".into()),
            msg => warn!("Waiting for start, got: {:?}", msg),
        }
    }
}

async fn handle_pulse_local(
    mut timing: TimingMetadata,
    detector: &mut PersistentPythonDetector,
    config: &ExperimentConfig,
) -> Result<InferenceMessage, Box<dyn Error + Send + Sync>> {
    let frame_number = timing.frame_number;
    let path = format!("pi-sender/sample/seq3-drone_{:07}.jpg", frame_number);
    let frame_data = std::fs::read(&path)?;
    debug!("Loaded frame {} ({} bytes)", frame_number, frame_data.len());
    
    let temp_frame_message = FrameMessage {
        sequence_id: timing.sequence_id,
        frame_data,
        width: FRAME_WIDTH,
        height: FRAME_HEIGHT,
        timing: timing.clone(),
    };

    timing.pi_capture_start = Some(current_timestamp_micros());

    let inference = perform_python_inference_with_counts(&temp_frame_message,
        detector, &config.model_name, "local")?;

    Ok(InferenceMessage {
        sequence_id: timing.sequence_id,
        timing,
        inference,
    })
}

async fn handle_pulse_offload(
    mut timing: TimingMetadata,
) -> Result<FrameMessage, Box<dyn Error + Send + Sync>> {
    let frame_number = timing.frame_number;
    let path = format!("pi-sender/sample/seq3-drone_{:07}.jpg", frame_number);
    let frame_data = std::fs::read(&path)?;
    debug!("Loaded frame {} ({} bytes)", frame_number, frame_data.len());
    
    timing.pi_sent_to_jetson = Some(current_timestamp_micros());

    Ok(FrameMessage {
        sequence_id: timing.sequence_id,
        frame_data,
        width: FRAME_WIDTH,
        height: FRAME_HEIGHT,
        timing,
    })
}