mod csv_writer;
mod experiment;
mod service;

use crate::experiment::Cli;
use clap::Parser;
use service::ControllerHarness;
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "controller=info,network=info,protocol=info,ort=warn".into()),
        )
        .without_time()
        .init();

    let cli = Cli::parse();
    let harness = ControllerHarness::new();

    experiment::execute(cli, &harness).await
}
