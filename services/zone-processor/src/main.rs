mod service;

use clap::Parser;
use inference::inference_manager::InferenceManager;
use inference::persistent::perform_onnx_inference_with_counts;

use network::framing::{read_message, read_message_stream, spawn_writer};
use protocol::config::{controller_address, zone_processor_bind_address};
use protocol::{
    current_timestamp_micros, ControlMessage, DeviceId, FrameMessage, InferenceMessage, Message,
};

use std::error::Error;
use std::sync::Arc;
use std::time::Duration;

use network::connection::{wait_for_config, wait_for_start};
use tokio::net::tcp::OwnedReadHalf;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, Mutex};
use tokio::time::sleep;
use tracing::{debug, error, info, warn};

#[derive(Parser, Debug)]
#[command(author, version, about = "Zone Processor (ZP)")]
struct Args {
    #[arg(short, long, default_value_t = 0)]
    id: u8,
}

async fn run_experiment_cycle(
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
        InferenceManager::new(config.model_name.clone(), std::path::PathBuf::from("models"))?;

    info!(
        "Waiting for {num_roadside_units} RSUs to connect on {}...",
        zone_processor_bind_address()
    );
    let listener = TcpListener::bind(zone_processor_bind_address()).await?;

    let mut rsu_mailboxes: Vec<Arc<Mutex<Option<FrameMessage>>>> =
        Vec::with_capacity(num_roadside_units as usize);

    while rsu_mailboxes.len() < num_roadside_units as usize {
        let (mut rsu_stream, rsu_addr) = listener.accept().await?;
        info!("RSU connected ({}/{}): {rsu_addr}", rsu_mailboxes.len() + 1, num_roadside_units);

        let hello = read_message_stream(&mut rsu_stream).await?;
        if let Message::Hello(DeviceId::RoadsideUnit(id)) = hello {
            info!("RSU-{id:04} handshake successful");
        } else {
            warn!("Unexpected handshake from {rsu_addr}: {hello:?}");
        }

        let mailbox = Arc::new(Mutex::new(Option::<FrameMessage>::None));
        rsu_mailboxes.push(mailbox.clone());

        tokio::spawn(async move {
            let (mut reader, _writer) = rsu_stream.into_split();
            loop {
                match read_message(&mut reader).await {
                    Ok(Message::Frame(frame)) => {
                        *mailbox.lock().await = Some(frame);
                    }
                    Ok(Message::Control(ControlMessage::Shutdown)) => break,
                    Err(_) => break,
                    _ => {}
                }
            }
            info!("RSU handler for {rsu_addr} terminated");
        });
    }

    drop(listener);
    info!("All RSUs connected. Notifying controller.");

    ctrl_tx.send(Message::Control(ControlMessage::ReadyToStart)).await.ok();
    wait_for_start(ctrl_reader).await?;
    info!("Experiment started - Processing frames via Round-Robin");

    let (result_tx, mut result_rx) = mpsc::unbounded_channel::<InferenceMessage>();

    manager.start_inference(result_tx, config.clone(), move |mut frame, detector, _cfg| {
        frame.timing.receive_start = Some(current_timestamp_micros());
        let inference = perform_onnx_inference_with_counts(&mut frame, detector)?;
        frame.timing.send_result = Some(current_timestamp_micros());

        Ok(InferenceMessage { sequence_id: frame.sequence_id, timing: frame.timing, inference })
    });

    let mut current_rsu_idx = 0;
    let mut interval = tokio::time::interval(Duration::from_micros(100));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

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
                        manager.shutdown().await;
                        break;
                    }
                    _ => {}
                }
            }

            _ = interval.tick() => {
                let mailbox = &rsu_mailboxes[current_rsu_idx];

                if let Some(frame) = mailbox.lock().await.take() {
                    debug!("Dispatching frame from RSU slot {} to inference", current_rsu_idx);
                    manager.update_pending_frame(frame);
                }

                current_rsu_idx = (current_rsu_idx + 1) % num_roadside_units as usize;
            }
        }
    }

    info!("Experiment cycle complete");
    Ok(true)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(
            |_| "zone_processor=info,inference=info,network=info,protocol=info,ort=warn".into(),
        ))
        .init();

    let args = Args::parse();
    let device_id = DeviceId::ZoneProcessor(args.id);

    info!("Starting {} (use --id to change)", device_id);

    loop {
        info!("Zone Processor connecting to controller at {}", controller_address());

        match TcpStream::connect(controller_address()).await {
            Ok(controller_stream) => {
                let (mut ctrl_reader, ctrl_writer) = controller_stream.into_split();
                let ctrl_tx = spawn_writer(ctrl_writer, 10);

                ctrl_tx.send(Message::Hello(device_id)).await.ok();
                info!("Sent hello to controller");

                loop {
                    match run_experiment_cycle(&mut ctrl_reader, ctrl_tx.clone()).await {
                        Ok(true) => {
                            info!("Experiment complete - ready for next one");
                            sleep(Duration::from_secs(2)).await;
                        }
                        Ok(false) => break,
                        Err(e) => {
                            error!("Experiment error: {e}");
                            break;
                        }
                    }
                }
            }
            Err(e) => {
                error!("Failed to connect to controller: {e}. Retrying...");
            }
        }
        sleep(Duration::from_secs(2)).await;
    }
}
