use image::GenericImageView;
use log::{debug, error, info, warn};
use shared::constants::*;
use shared::experiment_manager::ExperimentManager;
use shared::network::{read_message, spawn_writer};
use shared::{
    current_timestamp_micros, perform_python_inference_with_counts, ControlMessage, EncodingSpec,
    ExperimentConfig, ExperimentMode, Frame, FrameMessage, InferenceMessage, Message,
};
use std::error::Error;
use std::path::PathBuf;
use std::time::Duration;
use tokio::net::tcp::OwnedReadHalf;
use tokio::net::TcpStream;
use tokio::sync::mpsc;
use tokio::time::sleep;

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
    let mut manager = ExperimentManager::new(
        config.model_name.clone(),
        INFERENCE_PYTORCH_PATH.to_string(),
    )?;

    ctrl_tx
        .send(Message::Control(ControlMessage::ReadyToStart))
        .await
        .ok();
    wait_for_experiment_start(ctrl_reader).await?;
    info!("Experiment started. Processing frames locally...");

    let (result_tx, mut result_rx) = mpsc::unbounded_channel();

    manager.start_inference(
        result_tx,
        config.clone(),
        |frame, detector, cfg| async move {
            let mut det_opt = detector.lock().await;
            if let Some(ref mut det) = *det_opt {
                perform_python_inference_with_counts(&frame, det, &cfg.model_name, "local")
                    .map(|inference| InferenceMessage {
                        sequence_id: frame.sequence_id,
                        timing: frame.timing,
                        inference,
                    })
                    .map_err(|e| -> Box<dyn Error + Send + Sync> { e.into() })
            } else {
                Err("Detector unavailable".into())
            }
        },
    );

    let result = run_local_experiment_loop(
        ctrl_reader,
        ctrl_tx.clone(),
        &mut result_rx,
        &manager,
        &config,
    )
    .await;

    if let Err(e) = manager.shutdown().await {
        error!("Manager shutdown error: {}", e);
    }
    sleep(Duration::from_secs(2)).await;

    result
}

async fn run_local_experiment_loop(
    ctrl_reader: &mut OwnedReadHalf,
    ctrl_tx: mpsc::Sender<Message>,
    result_rx: &mut mpsc::UnboundedReceiver<InferenceMessage>,
    manager: &ExperimentManager,
    config: &ExperimentConfig,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    loop {
        tokio::select! {
            Some(result) = result_rx.recv() => {
                if ctrl_tx.send(Message::Result(result)).await.is_err() {
                    error!("Controller writer channel closed");
                    return Err("Controller disconnected".into());
                }
            }
            msg = read_message(ctrl_reader) => {
                match msg? {
                    Message::Pulse(mut timing) => {
                        timing.pi_capture_start = Some(current_timestamp_micros());
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

                                manager.update_pending_frame(frame_msg).await;
                            }
                            Err(e) => error!("Failed to load frame {}: {}", frame_number, e),
                        }
                    }
                    Message::Control(ControlMessage::Shutdown) => {
                        info!("Shutdown received");
                        return Ok(());
                    }

                    msg => warn!("Unexpected message during experiment: {:?}", msg),
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
    info!("Attempting to connect to Jetson on {}", jetson_address());

    let jetson_tx = loop {
        match TcpStream::connect(jetson_address()).await {
            Ok(stream) => {
                info!("Successfully connected to Jetson");
                let (_jetson_reader, jetson_writer) = stream.into_split();
                let tx = spawn_writer(jetson_writer);

                tx.send(Message::Hello(shared::types::DeviceId::Pi))
                    .await
                    .ok();

                break tx;
            }
            Err(e) => {
                warn!(
                    "Failed to connect to Jetson: {}. Retrying in 2 seconds...",
                    e
                );
                sleep(Duration::from_secs(2)).await;
            }
        }
    };

    info!("Jetson connection established");

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
    tracing_subscriber::fmt::init();

    loop {
        info!("Connecting to tmc-coordinator at {}", controller_address());

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
                    "Failed to connect to tmc-coordinator: {}. Retrying in 10 seconds...",
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

async fn handle_frame_remote(
    mut timing: shared::TimingMetadata,
    config: &ExperimentConfig,
    jetson_tx: &mpsc::Sender<Message>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    timing.pi_capture_start = Some(current_timestamp_micros());

    let frame_number = timing.frame_number;
    let (frame_data, width, height) = handle_frame(frame_number, config.encoding_spec.clone())?;

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
    let tier = image_tier(spec.tier);

    // roadside-unit/sample/{resolution}/{codec}/seq3-drone_{:07}_{tier}.{ext}
    let mut path = PathBuf::from("roadside-unit/sample");
    path.push(folder_res);
    path.push(folder_codec);
    path.push(format!("seq3-drone_{:07}_{}.{}", frame_number, tier, ext));

    if !path.exists() {
        return Err(format!("Frame file not found: {:?}", path).into());
    }

    let frame_data = std::fs::read(&path)?;

    let img = image::load_from_memory(&frame_data)?;
    let (width, height) = img.dimensions();

    Ok((frame_data, width, height))
}
