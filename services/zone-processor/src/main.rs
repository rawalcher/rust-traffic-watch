mod service;

use clap::Parser;

use network::framing::spawn_writer;
use protocol::config::controller_address;
use protocol::{DeviceId, Message};

use std::error::Error;
use std::time::Duration;

use crate::service::run_experiment_cycle;
use tokio::net::TcpStream;
use tokio::time::sleep;
use tracing::{error, info};

#[derive(Parser, Debug)]
#[command(author, version, about = "Zone Processor (ZP)")]
struct Args {
    #[arg(short, long, default_value_t = 0)]
    id: u8,
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
