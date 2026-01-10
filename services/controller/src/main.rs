mod csv_writer;
mod experiment;
mod service;

use std::{env, error::Error};

use tracing_subscriber;

use experiment::{run_single_experiment, run_test_suite, TestConfig};
use service::ControllerHarness;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    tracing_subscriber::fmt::init();
    let args: Vec<String> = env::args().collect();

    let harness = ControllerHarness::new();

    let test_config = TestConfig::parse_args(&args);

    if args.iter().any(|a| a.starts_with("--model=")) {
        run_single_experiment(&args, &harness).await
    } else {
        run_test_suite(test_config, &harness).await
    }
}
