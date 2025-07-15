use std::collections::HashMap;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Mutex, mpsc};
use once_cell::sync::Lazy;
use tracing::{debug, error, warn};
use crate::constants::{controller_bind_address, jetson_bind_address};
use crate::types::{Message, DeviceId};
use crate::network::{read_message, read_message_stream, send_message};

pub type DeviceSender = mpsc::UnboundedSender<Message>;
pub type DeviceReceiver = mpsc::UnboundedReceiver<Message>;

static DEVICES: Lazy<Mutex<HashMap<DeviceId, DeviceSender>>> = Lazy::new(|| Mutex::new(HashMap::new()));

/// Call this in controller to listen for incoming device connections
pub async fn start_controller_listener() -> tokio::io::Result<()> {
    let listener = TcpListener::bind(controller_bind_address()).await?;
    debug!("Controller listening on {}", controller_bind_address());

    loop {
        let (stream, addr) = listener.accept().await?;
        debug!("Incoming connection from {}", addr);
        tokio::spawn(async move {
            if let Err(e) = handle_device_connection(stream).await {
                error!("Connection error: {:?}", e);
            }
        });
    }
}

pub async fn wait_for_pi_on_jetson() -> tokio::io::Result<()> {
    let listener = TcpListener::bind(jetson_bind_address()).await?;
    debug!("Jetson waiting for Pi on {}", jetson_bind_address());

    let (stream, addr) = listener.accept().await?;
    debug!("Jetson accepted connection from {}", addr);

    handle_device_connection(stream).await.map_err(|e| {
        error!("Error in Jetson Pi handler: {:?}", e);
        std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
    })
}


async fn handle_device_connection(mut stream: TcpStream) -> anyhow::Result<()> {
    let hello = read_message_stream(&mut stream).await
        .map_err(|e| anyhow::anyhow!(e))?;

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
                Ok(msg) => debug!("[{}] -> {:?}", device_id, msg),
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

/// Get sender to a connected device (if available)
pub async fn get_device_sender(id: &DeviceId) -> Option<DeviceSender> {
    DEVICES.lock().await.get(id).cloned()
}
