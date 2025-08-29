use std::collections::{HashMap, HashSet};
use std::time::Duration;
use log::info;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Mutex, mpsc};
use once_cell::sync::Lazy;
use tracing::{debug, error, warn};
use crate::constants::{controller_bind_address, jetson_bind_address};
use crate::types::{Message, DeviceId, InferenceMessage, FrameMessage};
use crate::network::{read_message, send_message};
use crate::{read_message_stream, ControlMessage};

pub type DeviceSender = mpsc::UnboundedSender<Message>;
pub type DeviceReceiver = mpsc::UnboundedReceiver<Message>;

static DEVICES: Lazy<Mutex<HashMap<DeviceId, DeviceSender>>> = Lazy::new(|| Mutex::new(HashMap::new()));
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
    let listener = TcpListener::bind(controller_bind_address()).await?;
    debug!("Controller listening on {}", controller_bind_address());

    loop {
        let (stream, addr) = listener.accept().await?;
        debug!("Incoming connection from {}", addr);
        let role_clone = role.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_device_connection(stream, role_clone).await {
                error!("Connection error: {:?}", e);
            }
        });
    }
}

pub async fn wait_for_pi_on_jetson(role: Role) -> tokio::io::Result<()> {
    let bind_addr = jetson_bind_address();
    debug!("Attempting to bind to {}", bind_addr.clone());

    let listener = TcpListener::bind(bind_addr.clone()).await.map_err(|e| {
        error!("Failed to bind to {}: {}", bind_addr, e);
        e
    })?;

    debug!("Successfully bound to {}", bind_addr);
    let (stream, addr) = listener.accept().await?;
    debug!("Jetson accepted connection from {}", addr);

    handle_device_connection(stream, role).await.map_err(|e| {
        error!("Error in Jetson Pi handler: {:?}", e);
        std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
    })
}

pub async fn handle_device_connection(mut stream: TcpStream, role: Role) -> anyhow::Result<()> {
    let hello = read_message_stream(&mut stream).await.map_err(|e| anyhow::anyhow!(e))?;
    
    let device_id = match hello {
        Message::Hello(id) => id,
        other => {
            warn!("Unexpected first message: {:?}", other);
            return Ok(());
        }
    };

    debug!("Registered device: {:?}", device_id);

    let (tx, mut rx): (DeviceSender, DeviceReceiver) = mpsc::unbounded_channel();
    DEVICES.lock().await.insert(device_id, tx);

    let (mut reader, mut writer) = stream.into_split();

    let write_task = tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            if let Err(e) = send_message(&mut writer, &msg).await {
                error!("Failed to send to {:?}: {:?}", device_id, e);
                break;
            }
        }
    });

    let read_task = tokio::spawn(async move {
        loop {
            match read_message(&mut reader).await {
                Ok(msg) => {
                    debug!("[{}] -> {:?}", device_id, msg);

                    match (&role, msg) {
                        (Role::Controller { result_handler }, Message::Result(result)) => {
                            let _ = result_handler.send(result);
                        }
                        (Role::Controller { .. }, Message::Control(ControlMessage::ReadyToStart)) => {
                            mark_device_ready(device_id).await;
                            info!("{:?} is now ready", device_id);
                        }
                        (Role::Jetson { frame_handler }, Message::Frame(frame)) => {
                            let _ = frame_handler.send(frame);
                        }
                        _ => {}
                    }
                }
                Err(e) => {
                    error!("{:?} disconnected: {:?}", device_id, e);
                    break;
                }
            }
        }
    });
    
    let _ = tokio::try_join!(write_task, read_task);
    DEVICES.lock().await.remove(&device_id);
    
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
                info!("Waiting for {:?} to connect...", id);
            }
        }

        tokio::time::sleep(Duration::from_millis(2500)).await;
    }
    info!("All devices connected")
}

pub async fn wait_for_device_readiness(expected: &[DeviceId]) {
    loop {
        let mut all_ready = true;
        for id in expected {
            if !is_device_ready(id).await {
                info!("Waiting for {:?} to be ready...", id);
                all_ready = false;
            }
        }

        if all_ready {
            break;
        }

        tokio::time::sleep(Duration::from_millis(2500)).await;
    }
}

pub async fn clear_ready_devices() {
    READY_DEVICES.lock().await.clear();
}
