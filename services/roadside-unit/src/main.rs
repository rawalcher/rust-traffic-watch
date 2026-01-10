mod frame_loader;
mod service;

use log::{error, info};
use std::error::Error;
use std::time::Duration;
use tokio::net::TcpStream;
use tokio::time::sleep;

use network::framing::spawn_writer;
use protocol::config::controller_address;
use protocol::{DeviceId, Message};
use service::run_experiment_cycle;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    tracing_subscriber::fmt::init();

    loop {
        info!("Connecting to controller at {}", controller_address());

        match TcpStream::connect(controller_address()).await {
            Ok(controller_stream) => {
                let (mut ctrl_reader, ctrl_writer) = controller_stream.into_split();
                let ctrl_tx = spawn_writer(ctrl_writer, 10);

                ctrl_tx.send(Message::Hello(DeviceId::RoadsideUnit(0))).await.ok();

                loop {
                    match run_experiment_cycle(&mut ctrl_reader, ctrl_tx.clone()).await {
                        Ok(true) => info!("Ready for next experiment"),
                        Ok(false) => break,
                        Err(e) => {
                            error!("Experiment error: {e}");
                            break;
                        }
                    }
                }
            }
            Err(e) => error!("Connection failed: {e}. Retrying in 10s..."),
        }
        sleep(Duration::from_secs(10)).await;
    }
}
