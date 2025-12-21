mod service;

use inference::inference_manager::InferenceManager;
use inference::persistent::perform_onnx_inference_with_counts;

use network::framing::{read_message, read_message_stream, spawn_writer};
use protocol::config::{controller_address, jetson_bind_address};
use protocol::types::ExperimentConfig;
use protocol::{
    current_timestamp_micros, ControlMessage, DeviceId, FrameMessage, InferenceMessage, Message,
};

use std::error::Error;
use std::time::Duration;

use tokio::net::tcp::OwnedReadHalf;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tokio::time::sleep;

use tracing::{debug, error, info, warn};

// TODO: handle Pi reconnects more gracefully (future work)
// TODO: extract logic from pi and jetson that are same
// TODO: refactor timestamping to remove jetson and pi references

#[allow(clippy::too_many_lines)]
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

    info!("Initializing ONNX inference manager...");
    let mut manager =
        InferenceManager::new(config.model_name.clone(), std::path::PathBuf::from("models"))?;

    info!("Waiting for Pi to connect on {}...", jetson_bind_address());
    let listener = TcpListener::bind(jetson_bind_address()).await?;

    let (mut pi_stream, pi_addr) = listener.accept().await?;
    info!("Pi connected from {pi_addr}");

    drop(listener);
    info!("Listener dropped, port freed");

    let hello = read_message_stream(&mut pi_stream).await?;
    match hello {
        Message::Hello(DeviceId::RoadsideUnit(_)) => {
            info!("Pi hello received");
        }
        other => warn!("Unexpected hello from Pi: {other:?}"),
    }

    let (mut pi_reader, _pi_writer) = pi_stream.into_split();

    let (frame_tx, mut frame_rx) = mpsc::unbounded_channel::<FrameMessage>();
    let frame_tx_clone = frame_tx.clone();

    let pi_handler = tokio::spawn(async move {
        loop {
            match read_message(&mut pi_reader).await {
                Ok(Message::Frame(frame)) => {
                    info!("Jetson received frame from Pi: sequence_id={}", frame.sequence_id);

                    if frame_tx_clone.send(frame).is_err() {
                        error!("Failed to forward frame to inference pipeline");
                        break;
                    }
                }
                Ok(Message::Control(ControlMessage::Shutdown)) => {
                    info!("Pi sent shutdown");
                    break;
                }
                Ok(other) => debug!("Pi sent: {other:?}"),
                Err(_) => {
                    info!("Pi disconnected");
                    break;
                }
            }
        }
    });

    info!("Sending ReadyToStart to controller");
    ctrl_tx.send(Message::Control(ControlMessage::ReadyToStart)).await.ok();

    wait_for_experiment_start(ctrl_reader).await?;
    info!("Experiment started (Jetson ONNX inference)");

    let (result_tx, mut result_rx) = mpsc::unbounded_channel::<InferenceMessage>();

    manager.start_inference(result_tx, config.clone(), |mut frame, detector, _cfg| {
        frame.timing.jetson_received = Some(current_timestamp_micros());

        let inference = perform_onnx_inference_with_counts(&frame, detector)?;

        frame.timing.jetson_sent_result = Some(current_timestamp_micros());

        Ok(InferenceMessage { sequence_id: frame.sequence_id, timing: frame.timing, inference })
    });

    loop {
        tokio::select! {
            Some(result) = result_rx.recv() => {
                info!(
                    "Sending inference result for sequence_id {} to controller",
                    result.sequence_id
                );

                if ctrl_tx.send(Message::Result(result)).await.is_err() {
                    error!("Controller writer channel closed");
                    break;
                }
            }

            Some(frame) = frame_rx.recv() => {
                manager.update_pending_frame(frame);
            }

            msg = read_message(ctrl_reader) => {
                match msg? {
                    Message::Frame(frame) => {
                        warn!("Unexpected frame from controller, forwarding to pipeline");
                        let _ = frame_tx.send(frame);
                    }

                    Message::Control(ControlMessage::Shutdown) => {
                        info!("Shutdown received from controller, tearing down");

                        drop(frame_tx);
                        pi_handler.abort();

                        manager.shutdown().await;
                        break;
                    }

                    other => warn!("Unexpected controller message: {other:?}"),
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
        info!("Jetson connecting to controller at {}", controller_address());

        match TcpStream::connect(controller_address()).await {
            Ok(controller_stream) => {
                let (mut ctrl_reader, ctrl_writer) = controller_stream.into_split();
                let ctrl_tx = spawn_writer(ctrl_writer, 10);

                ctrl_tx.send(Message::Hello(DeviceId::ZoneProcessor(0))).await.ok();

                loop {
                    match run_experiment_cycle(&mut ctrl_reader, ctrl_tx.clone()).await {
                        Ok(true) => {
                            info!("Ready for next experiment");
                            sleep(Duration::from_secs(2)).await;
                        }
                        Ok(false) => {
                            info!("Clean shutdown received");
                            break;
                        }
                        Err(e) => {
                            error!("Experiment cycle error: {e}");
                            break;
                        }
                    }
                }
            }
            Err(e) => {
                error!("Failed to connect to controller: {e}. Retrying in 10 seconds...");
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
            msg => warn!("Waiting for config, got: {msg:?}"),
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
                return Err("Shutdown during start wait".into());
            }
            msg => warn!("Waiting for start, got: {msg:?}"),
        }
    }
}
