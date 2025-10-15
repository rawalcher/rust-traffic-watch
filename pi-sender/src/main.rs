use log::{debug, error, info, warn};
use shared::constants::*;
use shared::network::{read_message, spawn_writer};
use shared::{current_timestamp_micros, perform_python_inference_with_counts, ControlMessage, EncodingSpec, ExperimentConfig, ExperimentMode, Frame, FrameMessage, InferenceMessage, Message, PersistentPythonDetector};
use std::error::Error;
use std::path::PathBuf;
use std::time::Duration;
use tokio::net::tcp::OwnedReadHalf;
use tokio::net::TcpStream;
use tokio::sync::{mpsc, Mutex};
use tokio::time::sleep;
use std::sync::Arc;
use image::GenericImageView;

async fn run_experiment_cycle(
    ctrl_reader: &mut OwnedReadHalf,
    ctrl_tx: mpsc::Sender<Message>,
) -> Result<bool, Box<dyn Error + Send + Sync>> {
    let config = match wait_for_experiment_config(ctrl_reader).await {
        Ok(c) => c,
        Err(e) => {
            if e.to_string().contains("Shutdown before experiment") {
                return Ok(false);
            }
            return Err(e);
        }
    };

    match config.mode {
        ExperimentMode::LocalOnly => {
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
    let pending_frame: Arc<Mutex<Option<FrameMessage>>> = Arc::new(Mutex::new(None));

    let detector = PersistentPythonDetector::new(
        config.model_name.clone(),
        INFERENCE_PYTORCH_PATH.to_string(),
    )?;

    ctrl_tx
        .send(Message::Control(ControlMessage::ReadyToStart))
        .await
        .ok();
    wait_for_experiment_start(ctrl_reader).await?;
    info!("Experiment started. Processing frames locally...");

    let (result_tx, mut result_rx) = mpsc::unbounded_channel::<InferenceMessage>();
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

    let pending_frame_inference = Arc::clone(&pending_frame);
    let inference_config = config.clone();
    let ctrl_tx2 = ctrl_tx.clone();

    let detector_arc = Arc::new(Mutex::new(detector));
    let detector_clone = Arc::clone(&detector_arc);

    let inference_task = tokio::spawn(async move {
        loop {
            if *shutdown_rx.borrow() {
                info!("Inference task received shutdown signal");
                break;
            }

            let frame = {
                let mut pending = pending_frame_inference.lock().await;
                pending.take()
            };

            if let Some(frame_msg) = frame {
                if *shutdown_rx.borrow() {
                    info!("Inference task: Dropping frame {} due to shutdown", frame_msg.sequence_id);
                    break;
                }

                info!(
                    "Pi: Starting local inference for sequence_id={}",
                    frame_msg.sequence_id
                );

                let mut det = detector_clone.lock().await;
                match handle_frame_local(frame_msg, &mut det, &inference_config).await {
                    Ok(result) => {
                        let seq_id = result.sequence_id;
                        if result_tx.send(result).is_err() {
                            info!("Result channel closed, inference task exiting");
                            break;
                        }
                        info!("Pi: Completed local inference for sequence_id={}", seq_id);
                    }
                    Err(e) => {
                        error!("Failed to process frame locally: {}", e);
                    }
                }
            } else {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }
        info!("Inference task loop ended");
    });

    loop {
        tokio::select! {
            Some(result) = result_rx.recv() => {
                if ctrl_tx2.send(Message::Result(result)).await.is_err() {
                    error!("Controller writer channel closed");
                    break;
                }
            }
            msg = read_message(ctrl_reader) => {
                match msg? {
                    Message::Pulse(mut timing) => {
                        timing.pi_capture_start = Some(current_timestamp_micros());
                        let seq_id = timing.sequence_id;
                        let frame_number = timing.frame_number;

                        match handle_frame(frame_number, config.encoding_spec.clone()) {
                            Ok((frame_data, width, height)) => {
                                debug!(
                                    "Loaded frame {} ({:?}, tier={:?}, res={:?}) -> {} bytes",
                                    frame_number,
                                    config.encoding_spec.codec,
                                    config.encoding_spec.tier,
                                    config.encoding_spec.resolution,
                                    frame_data.len()
                                );

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
                        info!("Shutdown received, cleaning up...");
                        {
                            let mut pending = pending_frame.lock().await;
                            *pending = None;
                        }

                        let _ = shutdown_tx.send(true);

                        tokio::time::sleep(Duration::from_millis(50)).await;

                        info!("Shutting down Python detector (will wait for in-flight inference)...");
                        let mut det = detector_arc.lock().await;
                        if let Err(e) = det.shutdown() {
                            error!("Failed to shutdown detector: {}", e);
                        } else {
                            info!("Detector shutdown successful");
                        }

                        match tokio::time::timeout(Duration::from_secs(1), inference_task).await {
                            Ok(Ok(())) => {
                                info!("Inference task completed");
                            }
                            Ok(Err(e)) => {
                                error!("Inference task panicked: {:?}", e);
                            }
                            Err(_) => {
                                warn!("Inference task did not exit in 1s (already stopped)");
                            }
                        }

                        info!("Experiment cycle complete, ready for next experiment");
                        break;
                    }
                    msg => warn!("Unexpected message during experiment: {:?}", msg),
                }
            }
        }
    }

    Ok(())
}

async fn handle_offload_experiment(
    ctrl_reader: &mut OwnedReadHalf,
    ctrl_tx: mpsc::Sender<Message>,
    config: ExperimentConfig,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    info!("Connecting to Jetson on {}", jetson_address());
    sleep(Duration::from_secs(2)).await;

    let jetson_stream = TcpStream::connect(jetson_address()).await?;
    let (_jetson_reader, jetson_writer) = jetson_stream.into_split();
    let jetson_tx = spawn_writer(jetson_writer);

    jetson_tx
        .send(Message::Hello(shared::types::DeviceId::Pi))
        .await
        .ok();

    ctrl_tx
        .send(Message::Control(ControlMessage::ReadyToStart))
        .await
        .ok();
    wait_for_experiment_start(ctrl_reader).await?;

    info!("Experiment started. Forwarding frames to Jetson...");

    loop {
        match read_message(ctrl_reader).await? {
            Message::Pulse(timing) => {
                match handle_frame_remote(timing, &config, &jetson_tx).await {
                    Ok(()) => {
                        // Frame loaded and sent successfully
                    }
                    Err(e) => {
                        error!("Failed to handle remote frame: {}", e);
                    }
                }
            }
            Message::Control(ControlMessage::Shutdown) => {
                info!("Experiment cycle complete, resetting for next experiment");
                break;
            }
            msg => warn!("Unexpected message during experiment: {:?}", msg),
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    env_logger::init();

    loop {
        info!("Connecting to controller at {}", controller_address());

        match TcpStream::connect(controller_address()).await {
            Ok(controller_stream) => {
                let (mut ctrl_reader, ctrl_writer) = controller_stream.into_split();
                let ctrl_tx = spawn_writer(ctrl_writer);

                ctrl_tx
                    .send(Message::Hello(shared::types::DeviceId::Pi))
                    .await
                    .ok();

                loop {
                    match run_experiment_cycle(&mut ctrl_reader, ctrl_tx.clone()).await {
                        Ok(true) => {
                            info!("Ready for next experiment");
                            continue;
                        }
                        Ok(false) => {
                            info!("Clean shutdown received");
                            break;
                        }
                        Err(e) => {
                            error!("Experiment cycle error: {}", e);
                            break;
                        }
                    }
                }
            }
            Err(e) => {
                error!(
                    "Failed to connect to controller: {}. Retrying in 10 seconds...",
                    e
                );
            }
        }

        sleep(Duration::from_secs(10)).await;
    }
}

async fn wait_for_experiment_config(
    reader: &mut OwnedReadHalf,
) -> Result<ExperimentConfig, Box<dyn Error + Send + Sync>> {
    loop {
        match read_message(reader).await? {
            Message::Control(ControlMessage::ConfigureExperiment { config }) => {
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

async fn wait_for_experiment_start(
    reader: &mut OwnedReadHalf,
) -> Result<(), Box<dyn Error + Send + Sync>> {
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
    let inference =
        perform_python_inference_with_counts(&frame, detector, &config.model_name, "local")?;

    Ok(InferenceMessage {
        sequence_id: frame.sequence_id,
        timing: frame.timing,
        inference,
    })
}

async fn handle_frame_remote(
    mut timing: shared::TimingMetadata,
    config: &ExperimentConfig,
    jetson_tx: &mpsc::Sender<Message>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    timing.pi_capture_start = Some(current_timestamp_micros());

    let frame_number = timing.frame_number;
    let (frame_data, width, height) =
        handle_frame(frame_number, config.encoding_spec.clone())?;

    debug!(
        "Loaded frame {} for offload ({:?}, tier={:?}, res={:?}) -> {} bytes",
        frame_number,
        config.encoding_spec.codec,
        config.encoding_spec.tier,
        config.encoding_spec.resolution,
        frame_data.len()
    );

    timing.pi_sent_to_jetson = Some(current_timestamp_micros());

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

    jetson_tx
        .send(Message::Frame(frame_msg))
        .await
        .map_err(|_| "Jetson writer channel closed")?;

    Ok(())
}

pub fn handle_frame(
    frame_number: u64,
    spec: EncodingSpec,
) -> Result<(Vec<u8>, u32, u32), Box<dyn Error + Send + Sync>> {
    let folder_res = res_folder(spec.resolution);
    let folder_codec = codec_folder(spec.codec);
    let ext = codec_ext(spec.codec);

    // pi-sender/sample/{resolution}/{codec}/seq3-drone_{:07}.{ext}
    let mut path = PathBuf::from("pi-sender/sample");
    path.push(folder_res);
    path.push(folder_codec);
    path.push(format!("seq3-drone_{:07}.{}", frame_number, ext));

    if !path.exists() {
        return Err(format!("Frame file not found: {:?}", path).into());
    }

    let frame_data = std::fs::read(&path)?;

    let img = image::load_from_memory(&frame_data)?;
    let (width, height) = img.dimensions();

    Ok((frame_data, width, height))
}