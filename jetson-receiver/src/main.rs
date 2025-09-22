use log::{debug, error, info, warn};
use shared::constants::*;
use shared::{
    current_timestamp_micros, perform_python_inference_with_counts, ControlMessage, DeviceId,
    ExperimentConfig, FrameMessage, InferenceMessage, Message, PersistentPythonDetector,
};
use std::error::Error;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::tcp::OwnedReadHalf;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tokio::sync::Mutex;
use tokio::time::sleep;

use shared::network::{read_message, read_message_stream, spawn_writer};

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

    let pending_frame: Arc<Mutex<Option<FrameMessage>>> = Arc::new(Mutex::new(None));
    let pending_frame_network = Arc::clone(&pending_frame);
    let pending_frame_inference = Arc::clone(&pending_frame);

    let (frame_tx, mut frame_rx) = mpsc::unbounded_channel();

    info!("Waiting for Pi to connect on {}...", jetson_bind_address());
    let listener = TcpListener::bind(jetson_bind_address()).await?;

    let (mut pi_stream, pi_addr) = listener.accept().await?;
    info!("Pi connected from {}", pi_addr);

    drop(listener);
    info!("Listener dropped, port freed");

    let hello = read_message_stream(&mut pi_stream).await?;
    match hello {
        Message::Hello(DeviceId::Pi) => info!("Pi hello received"),
        other => warn!("Unexpected hello from Pi: {:?}", other),
    }

    let (mut pi_reader, _pi_writer) = pi_stream.into_split();

    let pi_handler = tokio::spawn(async move {
        loop {
            match read_message(&mut pi_reader).await {
                Ok(Message::Frame(frame)) => {
                    let seq_id = frame.sequence_id;
                    info!("Jetson received frame from Pi: sequence_id={}", seq_id);
                    if frame_tx.send(frame).is_err() {
                        break;
                    }
                }
                Ok(other) => debug!("Pi sent: {:?}", other),
                Err(_) => {
                    info!("Pi disconnected");
                    break;
                }
            }
        }
    });

    tokio::spawn(async move {
        while let Some(frame) = frame_rx.recv().await {
            let seq_id = frame.sequence_id;
            let mut pending = pending_frame_network.lock().await;
            let old_seq = pending.as_ref().map(|f| f.sequence_id);
            *pending = Some(frame);
            if let Some(old) = old_seq {
                debug!("Jetson: Dropped frame {} for newer frame {}", old, seq_id);
            }
            info!("Jetson: Updated pending frame to sequence_id={}", seq_id);
        }
    });

    info!("Initializing detector...");
    let model_name = config.model_name.clone();
    let mut detector = tokio::task::spawn_blocking(move || {
        PersistentPythonDetector::new(model_name, INFERENCE_TENSORRT_PATH.to_string())
    })
    .await??;

    info!("Python detector ready, sending ReadyToStart");
    ctrl_tx
        .send(Message::Control(ControlMessage::ReadyToStart))
        .await
        .ok();

    wait_for_experiment_start(ctrl_reader).await?;
    info!("Experiment started. Processing frames...");

    let (result_tx, mut result_rx) = mpsc::unbounded_channel::<InferenceMessage>();
    let inference_config = config.clone();

    let inference_task = tokio::spawn(async move {
        loop {
            let frame = {
                let mut pending = pending_frame_inference.lock().await;
                pending.take()
            };
            if let Some(frame_msg) = frame {
                info!(
                    "Jetson: Starting inference for sequence_id={}",
                    frame_msg.sequence_id
                );
                match handle_frame(frame_msg, &mut detector, &inference_config).await {
                    Ok(result) => {
                        let seq_id = result.sequence_id;
                        if result_tx.send(result).is_err() {
                            break;
                        }
                        info!("Jetson: Completed inference for sequence_id={}", seq_id);
                    }
                    Err(e) => error!("Failed to process frame: {}", e),
                }
            } else {
                sleep(Duration::from_millis(1)).await;
            }
        }
        let _ = detector.shutdown();
    });

    loop {
        tokio::select! {
            Some(result) = result_rx.recv() => {
                if ctrl_tx.send(Message::Result(result)).await.is_err() {
                    error!("Controller writer channel closed");
                    break;
                }
            }
            msg = read_message(ctrl_reader) => {
                match msg? {
                    Message::Control(ControlMessage::Shutdown) => {
                        info!("Experiment cycle complete, shutting down tasks");
                        inference_task.abort();
                        pi_handler.abort();
                        break;
                    }
                    unexpected => warn!("Unexpected controller message: {:?}", unexpected),
                }
            }
        }
    }

    info!("Experiment complete");
    Ok(true)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    env_logger::init();

    loop {
        info!(
            "Jetson connecting to controller at {}",
            controller_address()
        );

        match TcpStream::connect(controller_address()).await {
            Ok(controller_stream) => {
                let (mut ctrl_reader, ctrl_writer) = controller_stream.into_split();
                let ctrl_tx = spawn_writer(ctrl_writer);

                ctrl_tx.send(Message::Hello(DeviceId::Jetson)).await.ok();

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

async fn handle_frame(
    mut frame: FrameMessage,
    detector: &mut PersistentPythonDetector,
    config: &ExperimentConfig,
) -> Result<InferenceMessage, Box<dyn Error + Send + Sync>> {
    frame.timing.jetson_received = Some(current_timestamp_micros());

    let inference =
        perform_python_inference_with_counts(&frame, detector, &config.model_name, "offload")?;

    frame.timing.jetson_sent_result = Some(current_timestamp_micros());

    Ok(InferenceMessage {
        sequence_id: frame.sequence_id,
        timing: frame.timing,
        inference,
    })
}
