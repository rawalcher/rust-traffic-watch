use std::collections::HashMap;

use anyhow::Result;
use protocol::types::ExperimentConfig;
use protocol::{ControlMessage, DeviceId, Message};
use tokio::net::tcp::OwnedReadHalf;
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use crate::framing::{read_message, spawn_writer};

pub struct ConnectedDevice {
    pub id: DeviceId,
    pub tx: mpsc::Sender<Message>,
    pub reader: OwnedReadHalf,
}

pub struct ExperimentConnections {
    pub devices: HashMap<DeviceId, mpsc::Sender<Message>>,
    device_readers: HashMap<DeviceId, OwnedReadHalf>,
}

impl ExperimentConnections {
    #[must_use]
    pub fn new() -> Self {
        Self { devices: HashMap::new(), device_readers: HashMap::new() }
    }

    /// # Errors
    pub async fn accept_devices(
        &mut self,
        listener: &TcpListener,
        expected: &[DeviceId],
    ) -> Result<()> {
        info!("Waiting for {} devices to connect...", expected.len());

        while self.devices.len() < expected.len() {
            let (stream, peer_addr) = listener.accept().await?;
            debug!("Incoming connection from {}", peer_addr);

            let (mut reader, writer) = stream.into_split();

            match read_message(&mut reader).await {
                Ok(Message::Hello(device_id)) => {
                    if !expected.contains(&device_id) {
                        warn!("Unexpected device {} connected, rejecting", device_id);
                        continue;
                    }

                    if self.devices.contains_key(&device_id) {
                        warn!("Device {} already connected, rejecting duplicate", device_id);
                        continue;
                    }

                    let tx = spawn_writer(writer, 64);
                    self.devices.insert(device_id, tx);
                    self.device_readers.insert(device_id, reader);

                    info!(
                        "Device {} connected ({}/{})",
                        device_id,
                        self.devices.len(),
                        expected.len()
                    );
                }
                Ok(other) => {
                    warn!("Expected Hello, got {:?}", other);
                }
                Err(e) => {
                    warn!("Failed to read Hello: {}", e);
                }
            }
        }

        info!("All {} devices connected", expected.len());
        Ok(())
    }

    /// # Errors
    pub async fn send_config_to_all(&self, config: &ExperimentConfig) -> Result<()> {
        let msg = Message::Control(ControlMessage::ConfigureExperiment { config: config.clone() });

        for (id, tx) in &self.devices {
            if tx.send(msg.clone()).await.is_err() {
                error!("Failed to send config to {}", id);
            }
        }

        Ok(())
    }

    /// # Errors
    pub async fn wait_for_all_ready(&mut self) -> Result<()> {
        let mut ready_count = 0;
        let total = self.devices.len();

        info!("Waiting for {} devices to be ready...", total);

        for (device_id, reader) in &mut self.device_readers {
            loop {
                match read_message(reader).await? {
                    Message::Control(ControlMessage::ReadyToStart) => {
                        ready_count += 1;
                        info!("Device {} ready ({}/{})", device_id, ready_count, total);
                        break;
                    }
                    other => {
                        warn!("Waiting for ReadyToStart from {}, got {:?}", device_id, other);
                    }
                }
            }
        }

        info!("All {} devices ready", total);
        Ok(())
    }

    /// # Errors
    pub async fn signal_begin(&self) -> Result<()> {
        let msg = Message::Control(ControlMessage::BeginExperiment);

        for (id, tx) in &self.devices {
            if tx.send(msg.clone()).await.is_err() {
                error!("Failed to send BeginExperiment to {}", id);
            }
        }

        Ok(())
    }

    /// Send Shutdown to all devices
    pub async fn signal_shutdown(&self) {
        let msg = Message::Control(ControlMessage::Shutdown);

        for (id, tx) in &self.devices {
            if tx.send(msg.clone()).await.is_err() {
                debug!("Device {} already disconnected", id);
            }
        }
    }

    /// Get sender for a specific device
    #[must_use]
    pub fn get_sender(&self, id: &DeviceId) -> Option<&mpsc::Sender<Message>> {
        self.devices.get(id)
    }

    /// Take ownership of a device's reader (for spawning read tasks)
    pub fn take_reader(&mut self, id: &DeviceId) -> Option<OwnedReadHalf> {
        self.device_readers.remove(id)
    }

    /// Take all readers (consumes them from the struct)
    pub fn take_all_readers(&mut self) -> HashMap<DeviceId, OwnedReadHalf> {
        std::mem::take(&mut self.device_readers)
    }
}

impl Default for ExperimentConnections {
    fn default() -> Self {
        Self::new()
    }
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
                debug!("Shutdown received while waiting for config");
                return Ok(None);
            }
            msg => {
                warn!("Waiting for config, ignored: {msg:?}",);
            }
        }
    }
}

/// # Errors
pub async fn wait_for_start(reader: &mut OwnedReadHalf) -> Result<bool> {
    loop {
        match read_message(reader).await? {
            Message::Control(ControlMessage::BeginExperiment) => {
                info!("Received BeginExperiment signal");
                return Ok(true);
            }
            Message::Control(ControlMessage::Shutdown) => {
                info!("Shutdown received while waiting for start");
                return Ok(false);
            }
            msg => {
                warn!("Waiting for start, ignored: {msg:?}",);
            }
        }
    }
}

/// # Errors
pub async fn signal_ready(writer: &mpsc::Sender<Message>) -> Result<()> {
    writer
        .send(Message::Control(ControlMessage::ReadyToStart))
        .await
        .map_err(|_| anyhow::anyhow!("Failed to send ReadyToStart"))?;
    Ok(())
}
