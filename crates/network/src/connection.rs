use std::collections::{HashMap, HashSet};
use std::time::Duration;

use anyhow::Result;
use protocol::config::{controller_bind_address, jetson_bind_address};
use log::info;
use once_cell::sync::Lazy;
use protocol::{ControlMessage, DeviceId, FrameMessage, InferenceMessage, Message};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, Mutex};
use tracing::{debug, error, warn};

use crate::framing::{read_message, read_message_stream, send_message};

pub type DeviceSender = mpsc::UnboundedSender<Message>;
pub type DeviceReceiver = mpsc::UnboundedReceiver<Message>;

static DEVICES: Lazy<Mutex<HashMap<DeviceId, DeviceSender>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));
static READY_DEVICES: Lazy<Mutex<HashSet<DeviceId>>> = Lazy::new(|| Mutex::new(HashSet::new()));

#[derive(Clone)]
pub enum Role {
    Controller {
        result_handler: mpsc::UnboundedSender<InferenceMessage>,
    },
    Jetson {
        frame_handler: mpsc::UnboundedSender<FrameMessage>,
    },
}

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

pub async fn wait_for_pi_on_jetson(role: Role) -> tokio::io::Result<()> {
    let addr = jetson_bind_address();
    debug!("Jetson binding to {}", addr);

    let listener = TcpListener::bind(&addr).await.map_err(|e| {
        error!("Failed to bind to {}: {}", addr, e);
        e
    })?;

    let (stream, peer) = listener.accept().await?;
    debug!("Jetson accepted connection from {}", peer);

    handle_device_connection(stream, role)
        .await
        .map_err(|e| std::io::Error::other(e.to_string()))
}

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
                    (Role::Jetson { frame_handler }, Message::Frame(frame)) => {
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
        tokio::time::sleep(Duration::from_millis(2500)).await;
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
        tokio::time::sleep(Duration::from_millis(2500)).await;
    }
}

pub async fn clear_ready_devices() {
    READY_DEVICES.lock().await.clear();
}
