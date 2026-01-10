use log::{error, info, warn};
use std::error::Error;

use tokio::net::tcp::OwnedReadHalf;
use tokio::sync::mpsc;

use crate::frame_loader::handle_frame;

use inference::inference_manager::InferenceManager;
use inference::persistent::perform_onnx_inference_with_counts;
use network::connection::{wait_for_config, wait_for_start};
use network::framing::{read_message, spawn_writer};

use protocol::config::zone_processor_address;
use protocol::types::{ExperimentConfig, ExperimentMode, Frame};
use protocol::{
    current_timestamp_micros, get_hostname, ControlMessage, DeviceId, FrameMessage, InferenceMessage,
    Message, TimingMetadata,
};

pub async fn run_experiment_cycle(
    ctrl_reader: &mut OwnedReadHalf,
    ctrl_tx: mpsc::Sender<Message>,
) -> Result<bool, Box<dyn Error + Send + Sync>> {
    let Some(config) = wait_for_config(ctrl_reader).await? else {
        return Ok(false);
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
    wait_for_start(ctrl_reader).await?;

    info!("Experiment started (local mode)");

    let (result_tx, mut result_rx) = mpsc::unbounded_channel();
    let source_device = get_hostname();

    manager.start_inference(result_tx, config.clone(), move |mut frame, detector, _cfg| {
        let inference = perform_onnx_inference_with_counts(&mut frame, detector)?;
        frame.timing.send_start = Some(current_timestamp_micros());

        Ok(InferenceMessage { sequence_id: frame.sequence_id, timing: frame.timing, inference })
    });

    let result = run_local_loop(
        ctrl_reader,
        ctrl_tx.clone(),
        &mut result_rx,
        &manager,
        &config,
        &source_device,
    )
    .await;

    manager.shutdown().await;
    result
}

async fn run_local_loop(
    ctrl_reader: &mut OwnedReadHalf,
    ctrl_tx: mpsc::Sender<Message>,
    result_rx: &mut mpsc::UnboundedReceiver<InferenceMessage>,
    manager: &InferenceManager,
    config: &ExperimentConfig,
    source_device: &str,
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
                        timing.source_device = source_device.to_string();

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

                                manager.update_pending_frame(frame_msg);
                            }
                            Err(e) => error!("Failed to load frame {frame_number}: {e}"),
                        }
                    }

                    Message::Control(ControlMessage::Shutdown) => {
                        info!("Shutdown received");
                        return Ok(());
                    }

                    msg => warn!("Unexpected message: {msg:?}"),
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
    info!("Connecting to Zone Processor at {}", zone_processor_address());

    let zp_tx = loop {
        match tokio::net::TcpStream::connect(zone_processor_address()).await {
            Ok(stream) => {
                let (_reader, writer) = stream.into_split();
                let tx = spawn_writer(writer, 10);
                tx.send(Message::Hello(DeviceId::RoadsideUnit(0))).await.ok();
                break tx;
            }
            Err(e) => {
                warn!("Zone Processor connection failed: {e}, retrying...");
                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            }
        }
    };

    ctrl_tx.send(Message::Control(ControlMessage::ReadyToStart)).await.ok();
    wait_for_start(ctrl_reader).await?;

    info!("Experiment started (remote mode)");

    let source_device = get_hostname();

    loop {
        match read_message(ctrl_reader).await? {
            Message::Pulse(timing) => {
                if let Err(e) = process_remote_frame(timing, &config, &zp_tx, &source_device).await
                {
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

async fn process_remote_frame(
    mut timing: TimingMetadata,
    config: &ExperimentConfig,
    zp_tx: &mpsc::Sender<Message>,
    source_device: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    timing.source_device = source_device.to_string();

    // Start capturing frame
    timing.capture_start = Some(current_timestamp_micros());

    let frame_number = timing.frame_number;
    let (frame_data, width, height) = handle_frame(frame_number, &config.encoding_spec)?;

    // Frame encoding complete
    timing.encode_complete = Some(current_timestamp_micros());

    // Start sending to zone processor
    timing.send_start = Some(current_timestamp_micros());

    let frame_msg = FrameMessage {
        sequence_id: timing.sequence_id,
        timing,
        frame: Frame { frame_data, width, height, encoding: config.encoding_spec.clone() },
    };

    zp_tx.send(Message::Frame(frame_msg)).await.map_err(|_| "Zone Processor channel closed")?;

    Ok(())
}
