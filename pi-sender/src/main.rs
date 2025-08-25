use log::{debug, error, info, warn};
use shared::constants::*;
use shared::{current_timestamp_micros, perform_python_inference_with_counts, ControlMessage, ExperimentConfig, ExperimentMode, FrameMessage, InferenceMessage, Message, PersistentPythonDetector, TimingMetadata};
use std::error::Error;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, Mutex};
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

    match config.mode {
        ExperimentMode::LocalOnly => {
            let pending_frame: Arc<Mutex<Option<FrameMessage>>> = Arc::new(Mutex::new(None));

            let mut detector = PersistentPythonDetector::new(
                config.model_name.clone(),
                INFERENCE_PYTORCH_PATH.to_string(),
            )?;

            send_message(&mut ctrl_writer, &Message::Control(ControlMessage::ReadyToStart)).await?;
            wait_for_experiment_start(&mut ctrl_reader).await?;
            info!("Experiment started. Processing frames locally...");

            let (result_tx, mut result_rx) = mpsc::unbounded_channel::<InferenceMessage>();

            let pending_frame_inference = Arc::clone(&pending_frame);
            let inference_config = config.clone();

            let inference_task = tokio::spawn(async move {
                loop {
                    let frame = {
                        let mut pending = pending_frame_inference.lock().await;
                        pending.take()
                    };

                    if let Some(frame_msg) = frame {
                        info!("Pi: Starting local inference for sequence_id={}", frame_msg.sequence_id);

                        match handle_frame_local(frame_msg, &mut detector, &inference_config).await {
                            Ok(result) => {
                                let seq_id = result.sequence_id;
                                if let Err(e) = result_tx.send(result) {
                                    error!("Failed to send result to channel: {}", e);
                                    break;
                                }
                                info!("Pi: Completed local inference for sequence_id={}", seq_id);
                            }
                            Err(e) => {
                                error!("Failed to process frame locally: {}", e);
                            }
                        }
                    } else {
                        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
                    }
                }
                detector.shutdown()
            });

            loop {
                tokio::select! {
                    Some(result) = result_rx.recv() => {
                        send_message(&mut ctrl_writer, &Message::Result(result)).await?;
                    }
                    msg = read_message(&mut ctrl_reader) => {
                        match msg? {
                            Message::Pulse(mut timing) => {
                                timing.pi_capture_start = Some(current_timestamp_micros());
                                let seq_id = timing.sequence_id;

                                let frame_number = timing.frame_number;
                                let path = format!("pi-sender/sample/seq3-drone_{:07}.jpg", frame_number);
                                match std::fs::read(&path) {
                                    Ok(frame_data) => {
                                        debug!("Loaded frame {} ({} bytes)", frame_number, frame_data.len());

                                        let frame_msg = FrameMessage {
                                            sequence_id: timing.sequence_id,
                                            frame_data,
                                            width: FRAME_WIDTH,
                                            height: FRAME_HEIGHT,
                                            timing,
                                        };

                                        let mut pending = pending_frame.lock().await;
                                        let old_seq = pending.as_ref().map(|f| f.sequence_id);
                                        *pending = Some(frame_msg);

                                        if let Some(old) = old_seq {
                                            debug!("Pi: Dropped frame {} for newer frame {}", old, seq_id);
                                        }
                                        info!("Pi: Updated pending frame to sequence_id={}", seq_id);
                                    }
                                    Err(e) => {
                                        error!("Failed to load frame {}: {}", frame_number, e);
                                    }
                                }
                            }
                            Message::Control(ControlMessage::Shutdown) => {
                                info!("Shutdown received");
                                inference_task.abort();
                                break;
                            }
                            msg => warn!("Unexpected message during experiment: {:?}", msg),
                        }
                    }
                }
            }
        }
        ExperimentMode::Offload => {
            info!("Connecting to Jetson on {}", jetson_address());
            sleep(Duration::from_millis(7500)).await;

            let jetson_stream = TcpStream::connect(jetson_address()).await?;
            let (_, mut jetson_writer) = jetson_stream.into_split();
            send_message(&mut jetson_writer, &Message::Hello(shared::types::DeviceId::Pi)).await?;

            send_message(&mut ctrl_writer, &Message::Control(ControlMessage::ReadyToStart)).await?;
            wait_for_experiment_start(&mut ctrl_reader).await?;

            info!("Experiment started. Forwarding frames to Jetson...");

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

async fn handle_frame_local(
    frame: FrameMessage,
    detector: &mut PersistentPythonDetector,
    config: &ExperimentConfig,
) -> Result<InferenceMessage, Box<dyn Error + Send + Sync>> {
    let inference = perform_python_inference_with_counts(&frame,
                                                         detector, &config.model_name, "local")?;

    Ok(InferenceMessage {
        sequence_id: frame.sequence_id,
        timing: frame.timing,
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