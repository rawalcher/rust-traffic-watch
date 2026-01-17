use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, Mutex};
use tokio::task::JoinHandle;
use tokio::time::{timeout, Duration};
use tracing::{debug, error, info, warn};

use inference::inference_manager::InferenceManager;
use inference::persistent::perform_onnx_inference_with_counts;
use network::connection::{signal_ready, wait_for_config, wait_for_start};
use network::framing::{read_message, read_message_stream, spawn_writer};
use protocol::config::{controller_address, zone_processor_bind_address};
use protocol::{current_timestamp_micros, ControlMessage, DeviceId, InferenceMessage, Message};

/// # Errors
pub async fn run_single_experiment(
    device_id: DeviceId,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let stream = TcpStream::connect(controller_address()).await?;
    let (mut ctrl_reader, ctrl_writer) = stream.into_split();
    let ctrl_tx = spawn_writer(ctrl_writer, 64);

    ctrl_tx.send(Message::Hello(device_id)).await?;
    info!("Connected to controller, sent Hello");

    let Some(config) = wait_for_config(&mut ctrl_reader).await? else {
        info!("Received shutdown before config");
        return Ok(());
    };

    let num_rsus = config.num_roadside_units;
    info!(
        "Received config: mode={:?}, model={}, expecting {} RSUs",
        config.mode, config.model_name, num_rsus
    );

    info!("Initializing inference with model '{}'", config.model_name);
    let mut manager = InferenceManager::new(config.model_name.clone(), &PathBuf::from("models"))?;

    let (result_tx, mut result_rx) = mpsc::unbounded_channel::<InferenceMessage>();

    manager.start_inference(result_tx, config.clone(), move |mut frame, detector, _cfg| {
        frame.timing.receive_start = Some(current_timestamp_micros());
        let inference = perform_onnx_inference_with_counts(&mut frame, detector)?;
        frame.timing.send_result = Some(current_timestamp_micros());
        Ok(InferenceMessage { sequence_id: frame.sequence_id, timing: frame.timing, inference })
    });

    let manager = Arc::new(Mutex::new(manager));

    let bind_addr = zone_processor_bind_address();
    info!("Listening for RSUs on {}", bind_addr);
    let listener = TcpListener::bind(&bind_addr).await?;

    let mut rsu_handles: Vec<JoinHandle<()>> = Vec::new();
    let mut connected_count = 0u8;

    while connected_count < num_rsus {
        let (mut rsu_stream, rsu_addr) = listener.accept().await?;
        info!("RSU connection from {} ({}/{})", rsu_addr, connected_count + 1, num_rsus);

        let hello = read_message_stream(&mut rsu_stream).await?;
        let rsu_id = match hello {
            Message::Hello(id @ DeviceId::RoadsideUnit(_)) => {
                info!("RSU {} handshake successful", id);
                id
            }
            other => {
                warn!("Expected Hello from RSU, got {:?}", other);
                continue;
            }
        };

        let manager_clone = Arc::clone(&manager);
        let handle = tokio::spawn(async move {
            let (mut reader, _writer) = rsu_stream.into_split();

            loop {
                match read_message(&mut reader).await {
                    Ok(Message::Frame(frame)) => {
                        manager_clone.lock().await.update_pending_frame(rsu_id, frame).await;
                    }
                    Ok(Message::Control(ControlMessage::Shutdown)) => {
                        info!("RSU {} sent shutdown", rsu_id);
                        break;
                    }
                    Ok(other) => {
                        debug!("RSU {} sent unexpected: {:?}", rsu_id, other);
                    }
                    Err(e) => {
                        debug!("RSU {} disconnected: {}", rsu_id, e);
                        break;
                    }
                }
            }
        });

        rsu_handles.push(handle);
        connected_count += 1;
    }

    drop(listener);
    info!("All {} RSUs connected, listener closed", num_rsus);

    signal_ready(&ctrl_tx).await?;
    info!("Signaled ready to controller");

    if !wait_for_start(&mut ctrl_reader).await? {
        info!("Received shutdown before start");
        shutdown_manager(manager).await;
        return Ok(());
    }

    info!("Experiment started - processing frames");

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
                    Ok(Message::Control(ControlMessage::Shutdown)) => {
                        info!("Received shutdown from controller");
                        break;
                    }
                    Ok(other) => {
                        debug!("Ignored unexpected message: {:?}", other);
                    }
                    Err(e) => {
                        error!("Controller read error: {}", e);
                        break;
                    }
                }
            }
        }
    }

    info!("Waiting for RSU handlers to complete...");
    let shutdown_timeout = Duration::from_secs(5);

    for (idx, handle) in rsu_handles.into_iter().enumerate() {
        match timeout(shutdown_timeout, handle).await {
            Ok(Ok(())) => debug!("RSU handler {} completed", idx),
            Ok(Err(e)) => warn!("RSU handler {} panicked: {:?}", idx, e),
            Err(_) => warn!("RSU handler {} timed out", idx),
        }
    }

    shutdown_manager(manager).await;

    info!("Experiment completed");
    Ok(())
}

async fn shutdown_manager(manager: Arc<Mutex<InferenceManager>>) {
    match Arc::try_unwrap(manager) {
        Ok(mutex) => {
            let mut m = mutex.into_inner();
            m.shutdown().await;
            info!("Inference manager shut down cleanly");
        }
        Err(arc) => {
            warn!(
                "Cannot unwrap manager Arc ({} refs), shutting down via lock",
                Arc::strong_count(&arc)
            );
            arc.lock().await.shutdown().await;
        }
    }
}
