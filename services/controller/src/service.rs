use std::error::Error;
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio::time::{sleep, timeout, Duration, Instant};
use tracing::{debug, info, warn};

use crate::csv_writer::ConcurrentCsvWriter;
use network::connection::ExperimentConnections;
use network::framing::read_message;
use protocol::config::{
    compute_skip, controller_bind_address, fps_to_interval, MAX_FRAME_SEQUENCE,
};
use protocol::types::{ExperimentConfig, ExperimentMode};
use protocol::{
    current_timestamp_micros, ControlMessage, DeviceId, InferenceMessage, Message, TimingMetadata,
};

pub struct ControllerHarness;

const MAX_WAIT: Duration = Duration::from_secs(10);

impl ControllerHarness {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    pub async fn run_controller_with_retry(
        &self,
        config: ExperimentConfig,
        max_retries: u32,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let mut attempt = 0;

        loop {
            attempt += 1;

            match self.run_controller(config.clone()).await {
                Ok(()) => return Ok(()),
                Err(e) if attempt <= max_retries => {
                    warn!(
                        "Experiment failed (attempt {}/{}): {}. Retrying in 5s...",
                        attempt,
                        max_retries + 1,
                        e
                    );
                    sleep(Duration::from_secs(5)).await;
                }
                Err(e) => return Err(e),
            }
        }
    }

    pub async fn run_controller(
        &self,
        config: ExperimentConfig,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        log_experiment_start(&config);

        if config.num_roadside_units == 0 {
            return Err("Cannot run experiment with 0 Roadside Units.".into());
        }

        let mut connections = establish_connections(&config).await?;

        let csv_writer =
            ConcurrentCsvWriter::new(config.experiment_id.clone().as_str(), config.clone())?;
        let (result_tx, result_rx) = mpsc::unbounded_channel::<InferenceMessage>();

        let result_task = spawn_result_collector(csv_writer.clone(), result_rx);
        let read_tasks = spawn_device_readers(&mut connections, &result_tx.clone());

        connections.signal_begin().await?;
        info!("Experiment started!");

        let expected_results = send_pulses(&connections, &config).await;

        wait_for_results(&csv_writer, expected_results).await?;

        shutdown_experiment(connections, result_tx, result_task, read_tasks).await;

        let final_count = csv_writer.finalize()?;
        log_experiment_end(&config, final_count);

        sleep(Duration::from_secs(1)).await;
        Ok(())
    }
}

impl Default for ControllerHarness {
    fn default() -> Self {
        Self::new()
    }
}

fn log_experiment_start(config: &ExperimentConfig) {
    info!("========================================");
    info!("Starting experiment: {}", config.experiment_id);
    info!(
        "Mode: {:?}, Model: {}, FPS: {}, Duration: {}s, RSUs: {}",
        config.mode,
        config.model_name,
        config.fixed_fps,
        config.duration_seconds,
        config.num_roadside_units
    );
    info!("========================================");
}

fn log_experiment_end(config: &ExperimentConfig, final_count: usize) {
    info!("========================================");
    info!("Experiment {} completed", config.experiment_id);
    info!("Results saved: {}", final_count);
    info!("========================================");
}

async fn establish_connections(
    config: &ExperimentConfig,
) -> Result<ExperimentConnections, Box<dyn Error + Send + Sync>> {
    let rsu_devices: Vec<DeviceId> =
        (0..config.num_roadside_units).map(DeviceId::RoadsideUnit).collect();

    let required_devices = match config.mode {
        ExperimentMode::Local => rsu_devices,
        ExperimentMode::Offload => {
            let mut devices = rsu_devices;
            // TODO: dont forget to change if we ever to multi ZP
            devices.push(DeviceId::ZoneProcessor(0));
            devices
        }
    };

    let bind_addr = controller_bind_address();
    info!("Binding listener on {}", bind_addr);
    let listener = TcpListener::bind(&bind_addr).await?;

    let mut connections = ExperimentConnections::new();
    connections.accept_devices(&listener, &required_devices).await?;

    drop(listener);
    info!("All devices connected, listener closed");

    connections.send_config_to_all(config).await?;
    connections.wait_for_all_ready().await?;

    Ok(connections)
}

fn spawn_result_collector(
    csv_writer: ConcurrentCsvWriter,
    mut result_rx: mpsc::UnboundedReceiver<InferenceMessage>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        while let Some(mut result) = result_rx.recv().await {
            result.timing.controller_received = Some(current_timestamp_micros());

            if let Err(e) = csv_writer.write_result(&result).await {
                warn!("Failed to write result to CSV: {}", e);
            }

            let count = csv_writer.count().await;
            if count % 100 == 0 {
                info!("Processed {} results", count);
            }

            debug!(
                "Result {} seq={} mode={} detections={}",
                count,
                result.sequence_id,
                result.timing.mode.to_string(),
                result.inference.detection_count
            );
        }

        info!("Result collection task finished");
    })
}

fn spawn_device_readers(
    connections: &mut ExperimentConnections,
    result_tx: &mpsc::UnboundedSender<InferenceMessage>,
) -> Vec<JoinHandle<()>> {
    let device_readers = connections.take_all_readers();
    let mut read_handles = Vec::new();

    for (device_id, mut reader) in device_readers {
        let result_tx_clone = result_tx.clone();

        let handle = tokio::spawn(async move {
            loop {
                match read_message(&mut reader).await {
                    Ok(Message::Result(result)) => {
                        if result_tx_clone.send(result).is_err() {
                            debug!("Result channel closed for {}", device_id);
                            break;
                        }
                    }
                    Ok(Message::Control(ControlMessage::ReadyToStart)) => {
                        // Ignore late ready signals
                    }
                    Ok(other) => {
                        debug!("Device {} sent unexpected: {:?}", device_id, other);
                    }
                    Err(e) => {
                        debug!("Device {} read error (likely disconnected): {}", device_id, e);
                        break;
                    }
                }
            }
        });

        read_handles.push(handle);
    }

    read_handles
}

async fn send_pulses(connections: &ExperimentConnections, config: &ExperimentConfig) -> u64 {
    let start = Instant::now();
    let mut sequence_id: u64 = 0;
    let mut frame_number: u64 = 1;
    let pulse_interval = fps_to_interval(config.fixed_fps);
    let frame_skip = compute_skip(config.fixed_fps);

    let mut expected_results = 0u64;

    while start.elapsed().as_secs() < config.duration_seconds {
        let mut pulse_sent = false;

        for i in 0..config.num_roadside_units {
            let rsu_id = DeviceId::RoadsideUnit(i);
            if let Some(sender) = connections.get_sender(&rsu_id) {
                let timing = TimingMetadata {
                    source_device: rsu_id,
                    sequence_id,
                    frame_number,
                    mode: config.mode.clone(),
                    controller_sent_pulse: Some(current_timestamp_micros()),
                    controller_received: None,
                    capture_start: None,
                    encode_complete: None,
                    send_start: None,
                    receive_start: None,
                    queued_for_inference: None,
                    inference_start: None,
                    inference_complete: None,
                    send_result: None,
                };

                let pulse_msg = Message::Pulse(timing);
                if sender.send(pulse_msg).await.is_ok() {
                    pulse_sent = true;
                } else {
                    warn!("Failed to send pulse to {}", rsu_id);
                }
            }
        }

        if pulse_sent {
            sequence_id += 1;
            frame_number = advance_frame(frame_number, frame_skip);
            expected_results =
                expected_results.saturating_add(u64::from(config.num_roadside_units));
        }

        sleep(pulse_interval).await;
    }

    expected_results
}

#[allow(clippy::cast_precision_loss)]
async fn wait_for_results(
    csv_writer: &ConcurrentCsvWriter,
    expected_results: u64,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    info!("Finished sending {} pulses. Waiting for results...", expected_results);

    let wait_start = Instant::now();

    while wait_start.elapsed() < MAX_WAIT {
        let current_count = csv_writer.count().await;
        if current_count >= usize::try_from(expected_results)? {
            info!("All {} results received!", current_count);
            return Ok(());
        }
        sleep(Duration::from_millis(100)).await;
    }

    let final_count = csv_writer.count().await;
    info!(
        "Collected {} results out of {} expected ({:.1}%)",
        final_count,
        expected_results,
        (final_count as f64 / expected_results as f64) * 100.0
    );

    Ok(())
}

async fn shutdown_experiment(
    connections: ExperimentConnections,
    result_tx: mpsc::UnboundedSender<InferenceMessage>,
    result_task: JoinHandle<()>,
    read_tasks: Vec<JoinHandle<()>>,
) {
    info!("Sending shutdown signal to all devices...");
    connections.signal_shutdown().await;

    drop(result_tx);

    info!("Waiting for device read tasks to complete...");
    let read_timeout = Duration::from_secs(5);

    for (idx, handle) in read_tasks.into_iter().enumerate() {
        match timeout(read_timeout, handle).await {
            Ok(Ok(())) => debug!("Read task {} completed", idx),
            Ok(Err(e)) => warn!("Read task {} panicked: {:?}", idx, e),
            Err(_) => warn!("Read task {} timed out", idx),
        }
    }

    match timeout(Duration::from_secs(2), result_task).await {
        Ok(Ok(())) => debug!("Result collection task completed"),
        Ok(Err(e)) => warn!("Result collection task panicked: {:?}", e),
        Err(_) => warn!("Result collection task timed out"),
    }
}

const fn advance_frame(frame: u64, skip: u64) -> u64 {
    ((frame + skip - 1) % MAX_FRAME_SEQUENCE) + 1
}
