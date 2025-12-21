use log::{info, warn};
use network::framing::read_message;
use protocol::types::ExperimentConfig;
use protocol::{ControlMessage, Message};
use std::error::Error;
use tokio::net::tcp::OwnedReadHalf;

#[async_trait::async_trait]
pub trait ControllerReaderExt {
    async fn wait_for_config(&mut self) -> Result<ExperimentConfig, Box<dyn Error + Send + Sync>>;
    async fn wait_for_start(&mut self) -> Result<(), Box<dyn Error + Send + Sync>>;
}

#[async_trait::async_trait]
impl ControllerReaderExt for OwnedReadHalf {
    async fn wait_for_config(&mut self) -> Result<ExperimentConfig, Box<dyn Error + Send + Sync>> {
        loop {
            match read_message(self).await? {
                Message::Control(ControlMessage::ConfigureExperiment { config }) => {
                    info!(
                        "Received experiment config: mode={:?}, model={}",
                        config.mode, config.model_name
                    );
                    return Ok(config);
                }
                Message::Control(ControlMessage::Shutdown) => {
                    return Err("Shutdown signal received while waiting for config".into());
                }
                msg => warn!("Waiting for config, ignored unexpected message: {:?}", msg),
            }
        }
    }

    async fn wait_for_start(&mut self) -> Result<(), Box<dyn Error + Send + Sync>> {
        loop {
            match read_message(self).await? {
                Message::Control(ControlMessage::BeginExperiment) => {
                    info!("Received start signal from controller");
                    return Ok(());
                }
                Message::Control(ControlMessage::Shutdown) => {
                    return Err("Shutdown signal received while waiting for start".into());
                }
                msg => warn!("Waiting for start, ignored unexpected message: {:?}", msg),
            }
        }
    }
}
