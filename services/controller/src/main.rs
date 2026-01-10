mod csv_writer;
mod experiment;
mod service;

use crate::experiment::Cli;
use clap::Parser;
use service::ControllerHarness;
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();
    let harness = ControllerHarness::new();

    experiment::execute(cli, &harness).await
}
