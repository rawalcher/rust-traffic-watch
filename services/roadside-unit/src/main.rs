mod frame_loader;
mod service;

use clap::Parser;
use log::{error, info};
use std::error::Error;
use std::time::Duration;
use tokio::net::TcpStream;
use tokio::time::sleep;

use network::framing::spawn_writer;
use protocol::config::controller_address;
use protocol::{DeviceId, Message};
use service::run_experiment_cycle;

#[derive(Parser, Debug)]
#[command(author, version, about = "Roadside Unit (RSU)")]
struct Args {
    #[arg(short, long, default_value_t = 0)]
    id: u8,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(
            |_| "roadside_unit=info,inference=info,network=info,protocol=info,ort=warn".into(),
        ))
        .without_time()
        .init();

    let args = Args::parse();
    let device_id = DeviceId::RoadsideUnit(args.id);

    info!("Starting {} (use --id to change)", device_id);

    loop {
        info!("Connecting to controller at {}", controller_address());

        match TcpStream::connect(controller_address()).await {
            Ok(controller_stream) => {
                let (mut ctrl_reader, ctrl_writer) = controller_stream.into_split();
                let ctrl_tx = spawn_writer(ctrl_writer, 10);

                ctrl_tx.send(Message::Hello(device_id)).await.ok();

                loop {
                    match run_experiment_cycle(&mut ctrl_reader, ctrl_tx.clone(), device_id).await {
                        Ok(true) => info!("Ready for next experiment"),
                        Ok(false) => break,
                        Err(e) => {
                            error!("Experiment error: {e}");
                            break;
                        }
                    }
                }
            }
            Err(e) => error!("Connection failed: {e}. Retrying in 2s..."),
        }
        sleep(Duration::from_secs(2)).await;
    }
}
