use inference::inference_manager::InferenceManager;
use inference::persistent::perform_onnx_inference_with_counts;
use network::connection::{wait_for_config, wait_for_start};
use network::framing::{read_message, read_message_stream};
use protocol::config::zone_processor_bind_address;
use protocol::{current_timestamp_micros, ControlMessage, DeviceId, InferenceMessage, Message};
use std::error::Error;
use std::sync::Arc;
use tokio::net::tcp::OwnedReadHalf;
use tokio::net::TcpListener;
use tokio::sync::{mpsc, Mutex};
use tokio::task::JoinHandle;
use tokio::time::{timeout, Duration};
use tracing::{error, info, warn};

pub async fn run_experiment_cycle(
    ctrl_reader: &mut OwnedReadHalf,
    ctrl_tx: mpsc::Sender<Message>,
) -> Result<bool, Box<dyn Error + Send + Sync>> {
    let Some(config) = wait_for_config(ctrl_reader).await? else {
        return Ok(false);
    };

    let num_roadside_units = config.num_roadside_units;

    info!(
        "Received experiment config: mode={:?}, model={}, codec={:?}, tier={:?}",
        config.mode, config.model_name, config.encoding_spec.codec, config.encoding_spec.tier
    );

    info!("Initializing ONNX inference manager with model '{}'", config.model_name);
    let mut manager =
        InferenceManager::new(config.model_name.clone(), &std::path::PathBuf::from("models"))?;

    info!(
        "Waiting for {num_roadside_units} RSUs to connect on {}...",
        zone_processor_bind_address()
    );
    let listener = TcpListener::bind(zone_processor_bind_address()).await?;

    let (result_tx, mut result_rx) = mpsc::unbounded_channel::<InferenceMessage>();

    manager.start_inference(result_tx, config.clone(), move |mut frame, detector, _cfg| {
        frame.timing.receive_start = Some(current_timestamp_micros());
        let inference = perform_onnx_inference_with_counts(&mut frame, detector)?;
        frame.timing.send_result = Some(current_timestamp_micros());

        Ok(InferenceMessage { sequence_id: frame.sequence_id, timing: frame.timing, inference })
    });

    // Wrap in Arc<Mutex<>> to share across tasks
    let manager = Arc::new(Mutex::new(manager));
    let mut connected_count = 0;
    let mut rsu_handles: Vec<JoinHandle<()>> = Vec::new();

    while connected_count < num_roadside_units {
        let (mut rsu_stream, rsu_addr) = listener.accept().await?;
        info!("RSU connected ({}/{}): {rsu_addr}", connected_count + 1, num_roadside_units);

        let hello = read_message_stream(&mut rsu_stream).await?;
        let rsu_device_id = match hello {
            Message::Hello(DeviceId::RoadsideUnit(id)) => {
                info!("RSU-{id:04} handshake successful");
                DeviceId::RoadsideUnit(id)
            }
            other => {
                warn!("Unexpected handshake from {rsu_addr}: {other:?}");
                continue;
            }
        };

        let manager_clone = Arc::clone(&manager);

        let handle = tokio::spawn(async move {
            let (mut reader, _writer) = rsu_stream.into_split();
            loop {
                match read_message(&mut reader).await {
                    Ok(Message::Frame(frame)) => {
                        manager_clone.lock().await.update_pending_frame(rsu_device_id, frame);
                    }
                    Ok(Message::Control(ControlMessage::Shutdown)) => break,
                    Err(_) => break,
                    _ => {}
                }
            }
            info!("RSU handler for {rsu_addr} terminated");
        });

        rsu_handles.push(handle);
        connected_count += 1;
    }

    drop(listener);
    info!("All RSUs connected. Notifying controller.");

    ctrl_tx.send(Message::Control(ControlMessage::ReadyToStart)).await.ok();
    wait_for_start(ctrl_reader).await?;
    info!("Experiment started - Processing frames from all RSUs");

    loop {
        tokio::select! {
            Some(result) = result_rx.recv() => {
                if ctrl_tx.send(Message::Result(result)).await.is_err() {
                    break;
                }
            }

            msg = read_message(ctrl_reader) => {
                match msg? {
                    Message::Control(ControlMessage::Shutdown) => {
                        info!("Shutdown received from controller");
                        break;
                    }
                    _ => {}
                }
            }
        }
    }

    info!("Waiting for RSU handlers to complete...");

    const RSU_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(10);

    match timeout(RSU_SHUTDOWN_TIMEOUT, futures::future::join_all(rsu_handles)).await {
        Ok(results) => {
            let mut panicked = 0;
            for (idx, result) in results.into_iter().enumerate() {
                if let Err(e) = result {
                    error!("RSU handler {} panicked: {:?}", idx, e);
                    panicked += 1;
                }
            }
            if panicked > 0 {
                warn!("{} RSU handlers panicked during shutdown", panicked);
            } else {
                info!("All RSU handlers completed successfully");
            }
        }
        Err(_) => {
            warn!(
                "RSU handlers did not complete within {:?}, forcing shutdown",
                RSU_SHUTDOWN_TIMEOUT
            );
        }
    }

    match Arc::try_unwrap(manager) {
        Ok(mutex) => {
            let mut manager = mutex.into_inner();
            manager.shutdown().await;
            info!("Inference manager shut down cleanly");
        }
        Err(arc) => {
            warn!("Cannot unwrap Arc - {} strong references remain", Arc::strong_count(&arc));

            let mut manager = arc.lock().await;
            manager.shutdown().await;
            info!("Inference manager shut down via Arc lock");
        }
    }

    info!("Experiment cycle complete");
    Ok(true)
}
