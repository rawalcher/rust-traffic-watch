use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::tcp::OwnedReadHalf;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, Mutex};
use tokio::task::JoinHandle;
use tokio::time::{timeout, Duration};
use tracing::{debug, error, info, warn};

use inference::inference_manager::InferenceManager;
use inference::persistent::perform_onnx_inference_with_counts;
use network::connection::{establish_controller_connection, signal_ready, wait_for_start};
use network::framing::{read_message, read_message_stream};
use protocol::config::zone_processor_bind_address;
use protocol::{current_timestamp_micros, ControlMessage, DeviceId, InferenceMessage, Message};

/// # Errors
pub async fn run(device_id: DeviceId) -> Result<(), Box<dyn Error + Send + Sync>> {
    let (mut ctrl_reader, ctrl_tx, config) = establish_controller_connection(device_id).await?;

    info!("Initializing inference with model '{}'", config.model_name);
    let mut manager = InferenceManager::new(config.model_name.clone(), &PathBuf::from("models"))?;

    let (result_tx, result_rx) = mpsc::unbounded_channel::<InferenceMessage>();

    start_inference(&mut manager, result_tx, &config);

    let manager = Arc::new(Mutex::new(manager));

    let (shutdown_rsu_tx, shutdown_rsu_rx) = tokio::sync::broadcast::channel::<()>(16);

    let rsu_handles = accept_all_rsus(&config, Arc::clone(&manager), shutdown_rsu_rx).await?;

    signal_ready(&ctrl_tx).await?;
    info!("Signaled ready to controller");

    if !wait_for_start(&mut ctrl_reader).await? {
        info!("Received shutdown before start");
        let _ = shutdown_rsu_tx.send(());
        cleanup_rsu_handlers(rsu_handles).await;
        shutdown_manager(manager).await;
        return Ok(());
    }

    info!("Experiment started - processing frames");

    process_experiment_loop(ctrl_reader, ctrl_tx, result_rx).await;

    info!("Signaling all RSU handlers to stop");
    let _ = shutdown_rsu_tx.send(());

    cleanup_rsu_handlers(rsu_handles).await;
    shutdown_manager(manager).await;

    info!("Shutting down controller");
    Ok(())
}

fn start_inference(
    manager: &mut InferenceManager,
    result_tx: mpsc::UnboundedSender<InferenceMessage>,
    config: &protocol::types::ExperimentConfig,
) {
    manager.start_inference(result_tx, config.clone(), move |mut frame, detector, cfg| {
        frame.timing.receive_start = Some(current_timestamp_micros());
        let inference = perform_onnx_inference_with_counts(&mut frame, detector)?;
        let encoding_spec = cfg.encoding_spec.clone();

        frame.timing.send_result = Some(current_timestamp_micros());
        Ok(InferenceMessage {
            sequence_id: frame.sequence_id,
            inference,
            encoding_spec,
            timing: frame.timing,
        })
    });
}

async fn accept_all_rsus(
    config: &protocol::types::ExperimentConfig,
    manager: Arc<Mutex<InferenceManager>>,
    shutdown_rx: tokio::sync::broadcast::Receiver<()>,
) -> Result<Vec<JoinHandle<()>>, Box<dyn Error + Send + Sync>> {
    let bind_addr = zone_processor_bind_address();
    info!("Listening for RSUs on {}", bind_addr);
    let listener = TcpListener::bind(&bind_addr).await?;

    let num_rsus = config.num_roadside_units;
    let mut rsu_handles = Vec::new();
    let mut connected_count = 0u8;

    while connected_count < num_rsus {
        let (mut rsu_stream, rsu_addr) = listener.accept().await?;
        info!("RSU connection from {} ({}/{})", rsu_addr, connected_count + 1, num_rsus);

        let rsu_id = match read_message_stream(&mut rsu_stream).await? {
            Message::Hello(id @ DeviceId::RoadsideUnit(_)) => {
                info!("RSU {} handshake successful", id);
                id
            }
            other => {
                warn!("Expected Hello from RSU, got {:?}", other);
                continue;
            }
        };

        let handle =
            spawn_rsu_handler(rsu_stream, rsu_id, Arc::clone(&manager), shutdown_rx.resubscribe());
        rsu_handles.push(handle);
        connected_count += 1;
    }

    drop(listener);
    info!("All {} RSUs connected, listener closed", num_rsus);

    Ok(rsu_handles)
}

fn spawn_rsu_handler(
    rsu_stream: TcpStream,
    rsu_id: DeviceId,
    manager: Arc<Mutex<InferenceManager>>,
    mut shutdown_rx: tokio::sync::broadcast::Receiver<()>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let (mut reader, _writer) = rsu_stream.into_split();

        loop {
            tokio::select! {
                _ = shutdown_rx.recv() => {
                    info!("RSU {} handler received shutdown signal", rsu_id);
                    break;
                }

                msg = read_message(&mut reader) => {
                    match msg {
                        Ok(Message::Frame(frame)) => {
                            manager.lock().await.update_pending_frame(frame).await;
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
            }
        }

        info!("RSU {} handler stopped", rsu_id);
    })
}

async fn process_experiment_loop(
    mut ctrl_reader: OwnedReadHalf,
    ctrl_tx: mpsc::Sender<Message>,
    mut result_rx: mpsc::UnboundedReceiver<InferenceMessage>,
) {
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
}

async fn cleanup_rsu_handlers(rsu_handles: Vec<JoinHandle<()>>) {
    info!("Waiting for RSU handlers to complete...");
    let shutdown_timeout = Duration::from_secs(5);

    for (idx, handle) in rsu_handles.into_iter().enumerate() {
        match timeout(shutdown_timeout, handle).await {
            Ok(Ok(())) => debug!("RSU handler {} completed", idx),
            Ok(Err(e)) => warn!("RSU handler {} panicked: {:?}", idx, e),
            Err(_) => warn!("RSU handler {} timed out", idx),
        }
    }
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
