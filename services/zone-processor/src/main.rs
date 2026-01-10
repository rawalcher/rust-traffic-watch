mod service;

use inference::inference_manager::InferenceManager;
use inference::persistent::perform_onnx_inference_with_counts;

use network::framing::{read_message, read_message_stream, spawn_writer};
use protocol::config::{controller_address, zone_processor_bind_address};
use protocol::{
    current_timestamp_micros, ControlMessage, DeviceId, FrameMessage, InferenceMessage, Message,
};

use std::error::Error;
use std::time::Duration;

use tokio::net::tcp::OwnedReadHalf;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tokio::time::sleep;

use network::connection::{wait_for_config, wait_for_start};
use tracing::{debug, error, info, warn};

// TODO: handle Pi reconnects more gracefully (future work)
// TODO: extract logic from pi and jetson that are same

#[allow(clippy::too_many_lines)]
async fn run_experiment_cycle(
    ctrl_reader: &mut OwnedReadHalf,
    ctrl_tx: mpsc::Sender<Message>,
) -> Result<bool, Box<dyn Error + Send + Sync>> {
    let Some(config) = wait_for_config(ctrl_reader).await? else {
        return Ok(false);
    };

    info!(
        "Received experiment config: mode={:?}, model={}, codec={:?}, tier={:?}",
        config.mode, config.model_name, config.encoding_spec.codec, config.encoding_spec.tier
    );

    info!("Initializing ONNX inference manager with model '{}'", config.model_name);
    let mut manager =
        InferenceManager::new(config.model_name.clone(), std::path::PathBuf::from("models"))?;

    info!("Waiting for RSU to connect on {}...", zone_processor_bind_address());
    let listener = TcpListener::bind(zone_processor_bind_address()).await?;

    let (mut rsu_stream, rsu_addr) = listener.accept().await?;
    info!("RSU connected from {rsu_addr}");

    drop(listener);
    info!("Listener dropped, port freed for next experiment");

    let hello = read_message_stream(&mut rsu_stream).await?;
    match hello {
        Message::Hello(DeviceId::RoadsideUnit(id)) => {
            info!("RSU-{id:04} hello received");
        }
        other => {
            warn!("Unexpected hello from RSU: {other:?}");
        }
    }

    let (mut rsu_reader, _rsu_writer) = rsu_stream.into_split();

    let (frame_tx, mut frame_rx) = mpsc::unbounded_channel::<FrameMessage>();
    let frame_tx_clone = frame_tx.clone();

    let rsu_handler = tokio::spawn(async move {
        loop {
            match read_message(&mut rsu_reader).await {
                Ok(Message::Frame(frame)) => {
                    info!("ZP received frame from RSU: sequence_id={}", frame.sequence_id);

                    if frame_tx_clone.send(frame).is_err() {
                        error!("Failed to forward frame to inference pipeline");
                        break;
                    }
                }
                Ok(Message::Control(ControlMessage::Shutdown)) => {
                    info!("RSU sent shutdown signal");
                    break;
                }
                Ok(other) => debug!("RSU sent: {other:?}"),
                Err(e) => {
                    info!("RSU disconnected: {e:?}");
                    break;
                }
            }
        }
        info!("RSU handler task finished");
    });

    info!("Sending ReadyToStart to controller");
    ctrl_tx.send(Message::Control(ControlMessage::ReadyToStart)).await.ok();

    wait_for_start(ctrl_reader).await?;
    info!("Experiment started - ZP is now processing frames");

    let (result_tx, mut result_rx) = mpsc::unbounded_channel::<InferenceMessage>();

    manager.start_inference(result_tx, config.clone(), move |mut frame, detector, _cfg| {
        frame.timing.receive_start = Some(current_timestamp_micros());

        frame.timing.inference_start = Some(current_timestamp_micros());
        let inference = perform_onnx_inference_with_counts(&frame, detector)?;
        frame.timing.inference_complete = Some(current_timestamp_micros());

        frame.timing.send_result = Some(current_timestamp_micros());

        if let (Some(recv), Some(inf_start), Some(inf_end), Some(send)) = (
            frame.timing.receive_start,
            frame.timing.inference_start,
            frame.timing.inference_complete,
            frame.timing.send_result,
        ) {
            let prep_time = inf_start.saturating_sub(recv);
            let inference_time = inf_end.saturating_sub(inf_start);
            let send_prep = send.saturating_sub(inf_end);

            debug!(
                "ZP timing breakdown for seq={}: prep={}us, inference={}us, send_prep={}us",
                frame.sequence_id, prep_time, inference_time, send_prep
            );
        }

        Ok(InferenceMessage { sequence_id: frame.sequence_id, timing: frame.timing, inference })
    });

    loop {
        tokio::select! {
            Some(result) = result_rx.recv() => {
                info!(
                    "Sending inference result for seq={} to controller (detections={})",
                    result.sequence_id,
                    result.inference.detection_count
                );

                if ctrl_tx.send(Message::Result(result)).await.is_err() {
                    error!("Controller writer channel closed");
                    break;
                }
            }

            Some(frame) = frame_rx.recv() => {
                debug!("Forwarding frame seq={} to inference pipeline", frame.sequence_id);
                manager.update_pending_frame(frame);
            }

            msg = read_message(ctrl_reader) => {
                match msg? {
                    Message::Control(ControlMessage::Shutdown) => {
                        info!("Shutdown signal received from controller");

                        drop(frame_tx);
                        rsu_handler.abort();
                        manager.shutdown().await;

                        break;
                    }

                    other => warn!("Unexpected controller message during experiment: {other:?}"),
                }
            }
        }
    }

    info!("Experiment cycle complete - ready for next experiment");
    Ok(true)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    tracing_subscriber::fmt::init();

    loop {
        info!("Zone Processor connecting to controller at {}", controller_address());

        match TcpStream::connect(controller_address()).await {
            Ok(controller_stream) => {
                let (mut ctrl_reader, ctrl_writer) = controller_stream.into_split();
                let ctrl_tx = spawn_writer(ctrl_writer, 10);

                ctrl_tx.send(Message::Hello(DeviceId::ZoneProcessor(0))).await.ok();
                info!("Sent hello to controller");

                loop {
                    match run_experiment_cycle(&mut ctrl_reader, ctrl_tx.clone()).await {
                        Ok(true) => {
                            info!("Experiment complete - ready for next one");
                            sleep(Duration::from_secs(2)).await;
                        }
                        Ok(false) => {
                            info!("Clean shutdown received - exiting experiment loop");
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
