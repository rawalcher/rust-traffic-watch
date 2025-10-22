mod service;

use common::constants::{controller_address, jetson_bind_address, INFERENCE_TENSORRT_PATH};
use common::time::current_timestamp_micros;
use inference::experiment_manager::ExperimentManager;
use log::{debug, error, info, warn};
use network::framing::{read_message, read_message_stream, spawn_writer};
use protocol::{
    ControlMessage, DeviceId, ExperimentConfig, FrameMessage, InferenceMessage, Message,
};
use shared::perform_python_inference_with_counts;
use std::error::Error;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::tcp::OwnedReadHalf;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, Mutex};
use tokio::time::sleep;

// TODO: handle pi connections more gracefully, for future implementation we want to always listen to new pi connections since
// if one disconnects we just want to continue when reconnected. the jetson acts as an server that "exposes" the inference

async fn run_experiment_cycle(
    ctrl_reader: &mut OwnedReadHalf,
    ctrl_tx: mpsc::Sender<Message>,
) -> Result<bool, Box<dyn Error + Send + Sync>> {
    let config = match wait_for_experiment_config(ctrl_reader).await {
        Ok(c) => c,
        Err(e) => {
            if e.to_string().contains("Shutdown during") {
                return Ok(false);
            }
            return Err(e);
        }
    };

    info!("Initializing experiment manager...");
    let manager = Arc::new(Mutex::new(ExperimentManager::new(
        config.model_name.clone(),
        INFERENCE_TENSORRT_PATH.to_string(),
    )?));

    info!("Waiting for Pi to connect on {}...", jetson_bind_address());
    let listener = TcpListener::bind(jetson_bind_address()).await?;

    let (mut pi_stream, pi_addr) = listener.accept().await?;
    info!("Pi connected from {}", pi_addr);

    drop(listener);
    info!("Listener dropped, port freed");

    let hello = read_message_stream(&mut pi_stream).await?;
    match hello {
        Message::Hello(DeviceId::ZoneProcessor(0)) => info!("Pi hello received"),
        other => warn!("Unexpected hello from Pi: {:?}", other),
    }

    let (mut pi_reader, _pi_writer) = pi_stream.into_split();
    let (frame_tx, mut frame_rx) = mpsc::unbounded_channel::<FrameMessage>();
    let frame_tx_clone = frame_tx.clone();

    let pi_handler = tokio::spawn(async move {
        loop {
            match read_message(&mut pi_reader).await {
                Ok(Message::Frame(frame)) => {
                    let seq_id = frame.sequence_id;
                    info!("Jetson received frame from Pi: sequence_id={}", seq_id);

                    if frame_tx_clone.send(frame).is_err() {
                        error!("Failed to forward frame to processing pipeline");
                        break;
                    }
                }
                Ok(Message::Control(ControlMessage::Shutdown)) => {
                    info!("Pi sent shutdown");
                    break;
                }
                Ok(other) => debug!("Pi sent: {:?}", other),
                Err(_) => {
                    info!("Pi disconnected");
                    break;
                }
            }
        }
    });

    info!("Experiment manager ready, sending ReadyToStart");
    ctrl_tx
        .send(Message::Control(ControlMessage::ReadyToStart))
        .await
        .ok();

    wait_for_experiment_start(ctrl_reader).await?;
    info!("Experiment started. Processing frames...");

    let (result_tx, mut result_rx) = mpsc::unbounded_channel::<InferenceMessage>();

    {
        let manager_clone = Arc::clone(&manager);
        let mut locked = manager_clone.lock().await;
        locked.start_inference(
            result_tx,
            config.clone(),
            |frame, detector, cfg| async move {
                let mut det_opt = detector.lock().await;
                if let Some(ref mut det) = *det_opt {
                    let mut frame_mut = frame;
                    frame_mut.timing.jetson_received = Some(current_timestamp_micros());

                    let inference = perform_python_inference_with_counts(
                        &frame_mut,
                        det,
                        &cfg.model_name,
                        "offload",
                    )?;

                    frame_mut.timing.jetson_sent_result = Some(current_timestamp_micros());

                    Ok(InferenceMessage {
                        sequence_id: frame_mut.sequence_id,
                        timing: frame_mut.timing,
                        inference,
                    })
                } else {
                    Err("Detector unavailable".into())
                }
            },
        );
    }

    let manager_clone = Arc::clone(&manager);
    let frame_forwarder = tokio::spawn(async move {
        while let Some(frame) = frame_rx.recv().await {
            let mgr = manager_clone.lock().await;
            mgr.update_pending_frame(frame).await;
        }
        debug!("Frame forwarder task ended");
    });

    loop {
        tokio::select! {
            Some(result) = result_rx.recv() => {
                info!("Sending result for sequence_id {} back to controller", result.sequence_id);
                if ctrl_tx.send(Message::Result(result)).await.is_err() {
                    error!("Controller writer channel closed");
                    break;
                }
            }
            msg = read_message(ctrl_reader) => {
                match msg? {
                    Message::Frame(frame) => {
                        warn!("Unexpected frame on controller channel, forwarding to pipeline");
                        let _ = frame_tx.send(frame);
                    }
                    Message::Control(ControlMessage::Shutdown) => {
                        info!("Shutdown received from controller, beginning teardown");

                        drop(frame_tx);
                        pi_handler.abort();
                        let _ = frame_forwarder.await;

                        let mut mgr = manager.lock().await;
                        if let Err(e) = mgr.shutdown().await {
                            error!("Manager shutdown error: {}", e);
                        }

                        break;
                    }
                    unexpected => warn!("Unexpected controller message: {:?}", unexpected),
                }
            }
        }
    }

    info!("Experiment cycle complete");
    Ok(true)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    tracing_subscriber::fmt::init();

    loop {
        info!(
            "Jetson connecting to controller at {}",
            controller_address()
        );

        match TcpStream::connect(controller_address()).await {
            Ok(controller_stream) => {
                let (mut ctrl_reader, ctrl_writer) = controller_stream.into_split();
                // TODO: decide on what capacity each message should operate
                let ctrl_tx = spawn_writer(ctrl_writer, 10);

                ctrl_tx
                    .send(Message::Hello(DeviceId::ZoneProcessor(0)))
                    .await
                    .ok();

                loop {
                    match run_experiment_cycle(&mut ctrl_reader, ctrl_tx.clone()).await {
                        Ok(true) => {
                            info!("Ready for next experiment");
                            sleep(Duration::from_secs(2)).await;
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
            Message::Control(ControlMessage::ConfigureExperiment { config }) => return Ok(config),
            Message::Control(ControlMessage::Shutdown) => {
                return Err("Shutdown during config wait".into());
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
            Message::Control(ControlMessage::Shutdown) => {
                return Err("Shutdown during start wait".into())
            }
            msg => warn!("Waiting for start, got: {:?}", msg),
        }
    }
}
