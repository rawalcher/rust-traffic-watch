use std::error::Error;
use std::path::PathBuf;
use tokio::net::TcpStream;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use crate::frame_loader::handle_frame;
use inference::inference_manager::InferenceManager;
use inference::persistent::perform_onnx_inference_with_counts;
use network::connection::{signal_ready, wait_for_config, wait_for_start};
use network::framing::{read_message, spawn_writer};
use protocol::config::{controller_address, zone_processor_address};
use protocol::types::{ExperimentConfig, ExperimentMode, Frame};
use protocol::{
    current_timestamp_micros, ControlMessage, DeviceId, FrameMessage, InferenceMessage, Message,
};

/// # Errors
pub async fn run_single_experiment(
    device_id: DeviceId,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let stream = TcpStream::connect(controller_address()).await?;
    let (mut reader, writer) = stream.into_split();
    let tx = spawn_writer(writer, 64);

    tx.send(Message::Hello(device_id)).await?;
    info!("Connected to controller, sent Hello");

    let Some(config) = wait_for_config(&mut reader).await? else {
        info!("Received shutdown before config");
        return Ok(());
    };

    info!(
        "Received config: mode={:?}, model={}, fps={}",
        config.mode, config.model_name, config.fixed_fps
    );

    match config.mode {
        ExperimentMode::Local => run_local_experiment(&mut reader, tx, config, device_id).await,
        ExperimentMode::Offload => run_offload_experiment(&mut reader, tx, config, device_id).await,
    }
}

async fn run_local_experiment(
    ctrl_reader: &mut tokio::net::tcp::OwnedReadHalf,
    ctrl_tx: mpsc::Sender<Message>,
    config: ExperimentConfig,
    device_id: DeviceId,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    info!("Initializing local inference with model '{}'", config.model_name);

    let mut manager = InferenceManager::new(config.model_name.clone(), &PathBuf::from("models"))?;

    let (result_tx, mut result_rx) = mpsc::unbounded_channel::<InferenceMessage>();

    let source_device = device_id.to_string();
    manager.start_inference(result_tx, config.clone(), move |mut frame, detector, _cfg| {
        let inference = perform_onnx_inference_with_counts(&mut frame, detector)?;
        frame.timing.send_start = Some(current_timestamp_micros());
        Ok(InferenceMessage { sequence_id: frame.sequence_id, timing: frame.timing, inference })
    });

    signal_ready(&ctrl_tx).await?;
    info!("Signaled ready to controller");

    if !wait_for_start(ctrl_reader).await? {
        info!("Received shutdown before start");
        manager.shutdown().await;
        return Ok(());
    }

    info!("Experiment started (local mode)");

    loop {
        tokio::select! {
            Some(result) = result_rx.recv() => {
                if ctrl_tx.send(Message::Result(result)).await.is_err() {
                    error!("Controller connection lost");
                    break;
                }
            }

            msg = read_message(ctrl_reader) => {
                match msg {
                    Ok(Message::Pulse(mut timing)) => {
                        timing.source_device = source_device.clone();
                        timing.capture_start = Some(current_timestamp_micros());

                        let frame_number = timing.frame_number;
                        match handle_frame(frame_number, &config.encoding_spec) {
                            Ok((frame_data, width, height)) => {
                                timing.encode_complete = Some(current_timestamp_micros());

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
                                manager.update_pending_frame(device_id, frame_msg).await;
                            }
                            Err(e) => {
                                error!("Failed to load frame {}: {}", frame_number, e);
                            }
                        }
                    }

                    Ok(Message::Control(ControlMessage::Shutdown)) => {
                        info!("Received shutdown signal");
                        break;
                    }
                    Ok(other) => {
                        debug!("Ignored unexpected message: {:?}", other);
                    }
                    Err(e) => {
                        error!("Read error: {}", e);
                        break;
                    }
                }
            }
        }
    }
    manager.shutdown().await;
    info!("Local experiment completed");
    Ok(())
}

async fn run_offload_experiment(
    ctrl_reader: &mut tokio::net::tcp::OwnedReadHalf,
    ctrl_tx: mpsc::Sender<Message>,
    config: ExperimentConfig,
    device_id: DeviceId,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    info!("Connecting to Zone Processor at {}...", zone_processor_address());

    let zp_stream = loop {
        match TcpStream::connect(zone_processor_address()).await {
            Ok(stream) => {
                info!("Connected to Zone Processor");
                break stream;
            }
            Err(e) => {
                warn!("Zone Processor connection failed: {}. Retrying in 2s...", e);
                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            }
        }
    };

    let (_zp_reader, zp_writer) = zp_stream.into_split();
    let zp_tx = spawn_writer(zp_writer, 64);

    zp_tx.send(Message::Hello(device_id)).await?;
    info!("Sent Hello to Zone Processor");

    signal_ready(&ctrl_tx).await?;
    info!("Signaled ready to controller");

    if !wait_for_start(ctrl_reader).await? {
        info!("Received shutdown before start");
        return Ok(());
    }

    info!("Experiment started (offload mode)");

    let source_device = device_id.to_string();

    loop {
        match read_message(ctrl_reader).await {
            Ok(Message::Pulse(mut timing)) => {
                timing.source_device = source_device.clone();
                timing.capture_start = Some(current_timestamp_micros());

                let frame_number = timing.frame_number;
                match handle_frame(frame_number, &config.encoding_spec) {
                    Ok((frame_data, width, height)) => {
                        timing.encode_complete = Some(current_timestamp_micros());
                        timing.send_start = Some(current_timestamp_micros());

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

                        if zp_tx.send(Message::Frame(frame_msg)).await.is_err() {
                            error!("Zone Processor connection lost");
                            break;
                        }
                    }
                    Err(e) => {
                        error!("Failed to load frame {}: {}", frame_number, e);
                    }
                }
            }

            Ok(Message::Control(ControlMessage::Shutdown)) => {
                info!("Received shutdown signal");
                break;
            }
            Ok(other) => {
                debug!("Ignored unexpected message: {:?}", other);
            }
            Err(e) => {
                error!("Read error: {}", e);
                break;
            }
        }
    }

    info!("Offload experiment completed");
    Ok(())
}
