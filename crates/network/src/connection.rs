use std::collections::{HashMap, HashSet};
use std::time::Duration;

use crate::framing::{read_message, read_message_stream, send_message};
use anyhow::Result;
use log::info;
use protocol::config::{controller_bind_address, zone_processor_bind_address};
use protocol::types::ExperimentConfig;
use protocol::{ControlMessage, DeviceId, FrameMessage, InferenceMessage, Message};
use tokio::net::tcp::OwnedReadHalf;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, Mutex};
use tracing::{debug, error, warn};

pub type DeviceSender = mpsc::UnboundedSender<Message>;
pub type DeviceReceiver = mpsc::UnboundedReceiver<Message>;

static DEVICES: std::sync::LazyLock<Mutex<HashMap<DeviceId, DeviceSender>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));
static READY_DEVICES: std::sync::LazyLock<Mutex<HashSet<DeviceId>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashSet::new()));

#[derive(Clone)]
pub enum Role {
    Controller { result_handler: mpsc::UnboundedSender<InferenceMessage> },
    ZoneProcessor { frame_handler: mpsc::UnboundedSender<FrameMessage> },
}

/// # Errors
pub async fn start_controller_listener(role: Role) -> tokio::io::Result<()> {
    let addr = controller_bind_address();
    let listener = TcpListener::bind(&addr).await?;
    debug!("Controller listening on {}", addr);

    loop {
        let (stream, peer) = listener.accept().await?;
        debug!("Incoming connection from {}", peer);

        let role_clone = role.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_device_connection(stream, role_clone).await {
                error!("Connection error: {e:?}");
            }
        });
    }
}

/// # Errors
pub async fn wait_for_rsu_on_zone_processor(role: Role) -> tokio::io::Result<()> {
    let addr = zone_processor_bind_address();
    debug!("Zone Processor binding to {}", addr);

    let listener = TcpListener::bind(&addr).await.map_err(|e| {
        error!("Failed to bind to {}: {}", addr, e);
        e
    })?;

    let (stream, peer) = listener.accept().await?;
    debug!("Zone Processor accepted connection from {}", peer);

    handle_device_connection(stream, role).await.map_err(|e| std::io::Error::other(e.to_string()))
}

/// # Errors
pub async fn handle_device_connection(stream: TcpStream, role: Role) -> Result<()> {
    let mut stream = stream;
    let hello = read_message_stream(&mut stream).await?;

    let device_id = match hello {
        Message::Hello(id) => id,
        other => {
            warn!("Unexpected first message: {other:?}");
            return Ok(());
        }
    };

    debug!("Registered device: {device_id:?}");

    let (tx, mut rx): (DeviceSender, DeviceReceiver) = mpsc::unbounded_channel();
    DEVICES.lock().await.insert(device_id, tx);

    let (mut reader, mut writer) = stream.into_split();

    let write_task = tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            if let Err(e) = send_message(&mut writer, &msg).await {
                error!("Failed to send to {device_id:?}: {e:?}");
                break;
            }
        }
        debug!("{device_id:?} write task ended");
    });

    let read_task = tokio::spawn(async move {
        loop {
            match read_message(&mut reader).await {
                Ok(msg) => match (&role, msg) {
                    (Role::Controller { result_handler }, Message::Result(result)) => {
                        let _ = result_handler.send(result);
                    }
                    (Role::Controller { .. }, Message::Control(ControlMessage::ReadyToStart)) => {
                        mark_device_ready(device_id).await;
                        info!("{device_id:?} is now ready");
                    }
                    (Role::ZoneProcessor { frame_handler }, Message::Frame(frame)) => {
                        let _ = frame_handler.send(frame);
                    }
                    _ => {}
                },
                Err(e) => {
                    error!("{device_id:?} disconnected: {e:?}");
                    break;
                }
            }
        }
        debug!("{device_id:?} read task ended");
    });

    let _ = tokio::try_join!(write_task, read_task);
    DEVICES.lock().await.remove(&device_id);
    debug!("Removed {device_id:?} from DEVICES");
    Ok(())
}

pub async fn get_device_sender(id: &DeviceId) -> Option<DeviceSender> {
    DEVICES.lock().await.get(id).cloned()
}

pub async fn mark_device_ready(id: DeviceId) {
    READY_DEVICES.lock().await.insert(id);
}

pub async fn is_device_ready(id: &DeviceId) -> bool {
    READY_DEVICES.lock().await.contains(id)
}

pub async fn wait_for_devices(expected: &[DeviceId]) {
    loop {
        let connected = {
            let map = DEVICES.lock().await;
            expected.iter().all(|id| map.contains_key(id))
        };
        if connected {
            break;
        }

        for id in expected {
            let map = DEVICES.lock().await;
            if !map.contains_key(id) {
                info!("Waiting for {id:?} to connect...");
            }
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    info!("All devices connected");
}

pub async fn wait_for_device_readiness(expected: &[DeviceId]) {
    loop {
        let all_ready = {
            let ready = READY_DEVICES.lock().await;
            expected.iter().all(|id| ready.contains(id))
        };
        if all_ready {
            break;
        }

        for id in expected {
            if !is_device_ready(id).await {
                info!("Waiting for {id:?} to be ready...");
            }
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

pub async fn clear_ready_devices() {
    READY_DEVICES.lock().await.clear();
}

/// # Errors
pub async fn wait_for_config(reader: &mut OwnedReadHalf) -> Result<Option<ExperimentConfig>> {
    loop {
        match read_message(reader).await? {
            Message::Control(ControlMessage::ConfigureExperiment { config }) => {
                debug!(
                    "Received experiment config: mode={:?}, model={}",
                    config.mode, config.model_name
                );
                return Ok(Some(config));
            }
            Message::Control(ControlMessage::Shutdown) => {
                debug!("Shutdown signal received while waiting for config");
                return Ok(None);
            }
            msg => {
                warn!("Waiting for config, ignored unexpected message: {:?}", msg);
            }
        }
    }
}

/// # Errors
pub async fn wait_for_start(reader: &mut OwnedReadHalf) -> Result<()> {
    loop {
        match read_message(reader).await? {
            Message::Control(ControlMessage::BeginExperiment) => {
                info!("Received start signal from controller");
                return Ok(());
            }
            Message::Control(ControlMessage::Shutdown) => {
                return Err(anyhow::anyhow!("Shutdown signal received while waiting for start"));
            }
            msg => {
                warn!("Waiting for start, ignored unexpected message: {:?}", msg);
            }
        }
    }
}
