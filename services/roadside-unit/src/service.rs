use std::error::Error;
use std::path::PathBuf;
use tokio::net::tcp::OwnedReadHalf;
use tokio::net::TcpStream;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use crate::frame_loader::handle_frame;
use inference::inference_manager::InferenceManager;
use inference::persistent::perform_onnx_inference_with_counts;
use network::connection::{establish_controller_connection, signal_ready, wait_for_start};
use network::framing::{read_message, spawn_writer};
use protocol::config::zone_processor_address;
use protocol::types::{ExperimentConfig, ExperimentMode, Frame};
use protocol::{
    current_timestamp_micros, ControlMessage, DeviceId, FrameMessage, InferenceMessage, Message,
    TimingMetadata,
};

/// # Errors
pub async fn run(device_id: DeviceId) -> Result<(), Box<dyn Error + Send + Sync>> {
    let (ctrl_reader, ctrl_tx, config) = establish_controller_connection(device_id).await?;

    match config.mode {
        ExperimentMode::Local => run_local_experiment(ctrl_reader, ctrl_tx, config).await,
        ExperimentMode::Offload => {
            run_offload_experiment(ctrl_reader, ctrl_tx, config, device_id).await
        }
    }
}

async fn run_local_experiment(
    mut ctrl_reader: OwnedReadHalf,
    ctrl_tx: mpsc::Sender<Message>,
    config: ExperimentConfig,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    info!("Initializing local inference with model '{}'", config.model_name);

    let mut manager = InferenceManager::new(config.model_name.clone(), &PathBuf::from("models"))?;

    let (result_tx, result_rx) = mpsc::unbounded_channel::<InferenceMessage>();

    manager.start_inference(result_tx, config.clone(), move |mut frame, detector, _cfg| {
        let inference = perform_onnx_inference_with_counts(&mut frame, detector)?;
        frame.timing.send_start = Some(current_timestamp_micros());
        Ok(InferenceMessage { sequence_id: frame.sequence_id, timing: frame.timing, inference })
    });

    signal_ready(&ctrl_tx).await?;
    info!("Signaled ready to controller");

    if !wait_for_start(&mut ctrl_reader).await? {
        info!("Received shutdown before start");
        manager.shutdown().await;
        return Ok(());
    }

    info!("Experiment started (local mode)");

    process_local_loop(ctrl_reader, ctrl_tx, result_rx, config, manager).await
}

async fn process_local_loop(
    mut ctrl_reader: OwnedReadHalf,
    ctrl_tx: mpsc::Sender<Message>,
    mut result_rx: mpsc::UnboundedReceiver<InferenceMessage>,
    config: ExperimentConfig,
    mut manager: InferenceManager,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    loop {
        tokio::select! {
            Some(result) = result_rx.recv() => {
                if ctrl_tx.send(Message::Result(result)).await.is_err() {
                    error!("Controller connection lost");
                    break;
                }
            }

            msg = read_message(&mut ctrl_reader) => {
                match msg {
                    Ok(Message::Pulse(timing)) => {
                        if let Err(e) = handle_pulse_local(
                            timing,
                            &config,
                            &manager,
                        ).await {
                            error!("Failed to handle pulse: {}", e);
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

async fn handle_pulse_local(
    mut timing: TimingMetadata,
    config: &ExperimentConfig,
    manager: &InferenceManager,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    timing.capture_start = Some(current_timestamp_micros());

    let frame_number = timing.frame_number;
    let (frame_data, width, height) = handle_frame(frame_number, &config.encoding_spec)?;

    timing.encode_complete = Some(current_timestamp_micros());

    let frame_msg = FrameMessage {
        sequence_id: timing.sequence_id,
        timing,
        frame: Frame { frame_data, width, height, encoding: config.encoding_spec.clone() },
    };

    manager.update_pending_frame(frame_msg).await;
    Ok(())
}

async fn run_offload_experiment(
    mut ctrl_reader: OwnedReadHalf,
    ctrl_tx: mpsc::Sender<Message>,
    config: ExperimentConfig,
    device_id: DeviceId,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    info!("Connecting to Zone Processor at {}...", zone_processor_address());

    let zp_tx = connect_to_zone_processor(device_id).await?;

    signal_ready(&ctrl_tx).await?;
    info!("Signaled ready to controller");

    if !wait_for_start(&mut ctrl_reader).await? {
        info!("Received shutdown before start");
        return Ok(());
    }

    info!("Experiment started (offload mode)");

    process_offload_loop(ctrl_reader, zp_tx, config).await
}

async fn connect_to_zone_processor(
    device_id: DeviceId,
) -> Result<mpsc::Sender<Message>, Box<dyn Error + Send + Sync>> {
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

    Ok(zp_tx)
}

async fn process_offload_loop(
    mut ctrl_reader: OwnedReadHalf,
    zp_tx: mpsc::Sender<Message>,
    config: ExperimentConfig,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    loop {
        match read_message(&mut ctrl_reader).await {
            Ok(Message::Pulse(timing)) => {
                if let Err(e) = handle_pulse_offload(timing, &config, &zp_tx).await {
                    error!("Failed to handle pulse: {}", e);
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

async fn handle_pulse_offload(
    mut timing: TimingMetadata,
    config: &ExperimentConfig,
    zp_tx: &mpsc::Sender<Message>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    timing.capture_start = Some(current_timestamp_micros());

    let frame_number = timing.frame_number;
    let (frame_data, width, height) = handle_frame(frame_number, &config.encoding_spec)?;

    timing.encode_complete = Some(current_timestamp_micros());
    timing.send_start = Some(current_timestamp_micros());

    let frame_msg = FrameMessage {
        sequence_id: timing.sequence_id,
        timing,
        frame: Frame { frame_data, width, height, encoding: config.encoding_spec.clone() },
    };

    zp_tx.send(Message::Frame(frame_msg)).await.map_err(|_| "Zone Processor connection lost")?;

    Ok(())
}
