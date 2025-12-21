use log::{error, info, warn};
use std::error::Error;
use std::time::Duration;

use tokio::net::tcp::OwnedReadHalf;
use tokio::net::TcpStream;
use tokio::sync::mpsc;
use tokio::time::sleep;

use crate::frame_loader::handle_frame;
use crate::protocol_ext::ControllerReaderExt;

use inference::inference_manager::InferenceManager;
use inference::persistent::perform_onnx_inference_with_counts;

use network::framing::{read_message, spawn_writer};

use protocol::config::jetson_address;
use protocol::types::{ExperimentConfig, ExperimentMode, Frame};
use protocol::{
    current_timestamp_micros, ControlMessage, DeviceId, FrameMessage, InferenceMessage, Message,
    TimingMetadata,
};

pub async fn run_experiment_cycle(
    ctrl_reader: &mut OwnedReadHalf,
    ctrl_tx: mpsc::Sender<Message>,
) -> Result<bool, Box<dyn Error + Send + Sync>> {
    let config = match ctrl_reader.wait_for_config().await {
        Ok(c) => c,
        Err(e) => {
            if e.to_string().contains("Shutdown") {
                return Ok(false);
            }
            return Err(e);
        }
    };

    match config.mode {
        ExperimentMode::Local => {
            handle_local_experiment(ctrl_reader, ctrl_tx, config).await?;
        }
        ExperimentMode::Offload => {
            handle_offload_experiment(ctrl_reader, ctrl_tx, config).await?;
        }
    }

    Ok(true)
}

async fn handle_local_experiment(
    ctrl_reader: &mut OwnedReadHalf,
    ctrl_tx: mpsc::Sender<Message>,
    config: ExperimentConfig,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut manager =
        InferenceManager::new(config.model_name.clone(), std::path::PathBuf::from("models"))?;

    ctrl_tx.send(Message::Control(ControlMessage::ReadyToStart)).await.ok();

    ctrl_reader.wait_for_start().await?;
    info!("Experiment started (local Rust inference)");

    let (result_tx, mut result_rx) = mpsc::unbounded_channel();

    manager.start_inference(result_tx, config.clone(), |frame, detector, _cfg| {
        let inference = perform_onnx_inference_with_counts(&frame, detector)?;

        Ok(InferenceMessage { sequence_id: frame.sequence_id, timing: frame.timing, inference })
    });

    let result =
        run_local_experiment_loop(ctrl_reader, ctrl_tx.clone(), &mut result_rx, &manager, &config)
            .await;

    manager.shutdown().await;
    sleep(Duration::from_secs(2)).await;

    result
}

async fn run_local_experiment_loop(
    ctrl_reader: &mut OwnedReadHalf,
    ctrl_tx: mpsc::Sender<Message>,
    result_rx: &mut mpsc::UnboundedReceiver<InferenceMessage>,
    manager: &InferenceManager,
    config: &ExperimentConfig,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    loop {
        tokio::select! {
            Some(result) = result_rx.recv() => {
                if ctrl_tx.send(Message::Result(result)).await.is_err() {
                    return Err("Controller disconnected".into());
                }
            }

            msg = read_message(ctrl_reader) => {
                match msg? {
                    Message::Pulse(mut timing) => {
                        timing.pi_capture_start = Some(current_timestamp_micros());

                        let frame_number = timing.frame_number;
                        match handle_frame(frame_number, &config.encoding_spec) {
                            Ok((frame_data, width, height)) => {
                                let frame_msg = FrameMessage {
                                    sequence_id: timing.sequence_id,
                                    timing,
                                    frame: Frame {
                                        frame_data,
                                        width,
                                        height,
                                        encoding: config.encoding_spec.clone(),
                                    },
                                };
                                manager.update_pending_frame(frame_msg);
                            }
                            Err(e) => error!("Failed to load frame {frame_number}: {e}"),
                        }
                    }

                    Message::Control(ControlMessage::Shutdown) => {
                        info!("Shutdown received");
                        return Ok(());
                    }

                    msg => warn!("Unexpected message during experiment: {msg:?}"),
                }
            }
        }
    }
}

async fn handle_offload_experiment(
    ctrl_reader: &mut OwnedReadHalf,
    ctrl_tx: mpsc::Sender<Message>,
    config: ExperimentConfig,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    info!("Connecting to Jetson at {}", jetson_address());

    let jetson_tx = loop {
        match TcpStream::connect(jetson_address()).await {
            Ok(stream) => {
                let (_reader, writer) = stream.into_split();
                let tx = spawn_writer(writer, 10);
                tx.send(Message::Hello(DeviceId::RoadsideUnit(0))).await.ok();
                break tx;
            }
            Err(e) => {
                warn!("Jetson connection failed: {e}, retrying...");
                sleep(Duration::from_secs(2)).await;
            }
        }
    };

    ctrl_tx.send(Message::Control(ControlMessage::ReadyToStart)).await.ok();

    ctrl_reader.wait_for_start().await?;
    info!("Experiment started (offload mode)");

    loop {
        match read_message(ctrl_reader).await? {
            Message::Pulse(timing) => {
                if let Err(e) = run_offload_experiment_loop(timing, &config, &jetson_tx).await {
                    error!("Remote frame failed: {e}");
                }
            }

            Message::Control(ControlMessage::Shutdown) => {
                info!("Shutdown received");
                break;
            }

            msg => warn!("Unexpected message: {msg:?}"),
        }
    }

    Ok(())
}

async fn run_offload_experiment_loop(
    mut timing: TimingMetadata,
    config: &ExperimentConfig,
    jetson_tx: &mpsc::Sender<Message>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    timing.pi_capture_start = Some(current_timestamp_micros());

    let frame_number = timing.frame_number;
    let (frame_data, width, height) = handle_frame(frame_number, &config.encoding_spec)?;

    timing.pi_sent_to_jetson = Some(current_timestamp_micros());

    let frame_msg = FrameMessage {
        sequence_id: timing.sequence_id,
        timing,
        frame: Frame { frame_data, width, height, encoding: config.encoding_spec.clone() },
    };

    jetson_tx.send(Message::Frame(frame_msg)).await.map_err(|_| "Jetson channel closed")?;

    Ok(())
}
