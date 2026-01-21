mod frame_loader;
mod service;

use clap::Parser;
use std::error::Error;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{error, info, warn};

use protocol::config::controller_address;
use protocol::DeviceId;

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
        info!("Connecting to controller at {}...", controller_address());

        match service::run(device_id).await {
            Ok(()) => {
                info!("Experiment completed successfully");
            }
            Err(e) => {
                let err_str = e.to_string();
                if err_str.contains("Connection refused") || err_str.contains("connection") {
                    warn!("Controller not available: {}", err_str);
                } else {
                    error!("Experiment error: {}", e);
                }
            }
        }

        info!("Waiting 3s before reconnecting...");
        sleep(Duration::from_secs(3)).await;
    }
}
