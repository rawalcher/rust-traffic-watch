use std::error::Error;
use std::sync::Arc;
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

    pub const fn advance_frame(frame: u64, skip: u64, max: u64) -> u64 {
        ((frame + skip - 1) % max) + 1
    }

    pub async fn run_controller(
        &self,
        config: ExperimentConfig,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
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

        if config.num_roadside_units == 0 {
            return Err("Cannot run experiment with 0 Roadside Units.".into());
        }

        let rsu_devices: Vec<DeviceId> =
            (0..config.num_roadside_units).map(DeviceId::RoadsideUnit).collect();

        let required_devices: Vec<DeviceId> = match config.mode {
            ExperimentMode::Local => rsu_devices.clone(),
            ExperimentMode::Offload => {
                let mut devices = rsu_devices.clone();
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

        connections.send_config_to_all(&config).await?;

        connections.wait_for_all_ready().await?;

        let csv_writer = ConcurrentCsvWriter::new(&config.experiment_id, config.clone())?;

        let (result_tx, mut result_rx) = mpsc::unbounded_channel::<InferenceMessage>();

        let csv_writer_clone = csv_writer.clone();
        let (stop_tx, mut stop_rx) = tokio::sync::watch::channel(false);

        let result_collection_task: JoinHandle<()> = tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = stop_rx.changed() => {
                        if *stop_rx.borrow() { break; }
                    }
                    maybe = result_rx.recv() => {
                        match maybe {
                            Some(mut result) => {
                                result.timing.controller_received = Some(current_timestamp_micros());

                                if let Err(e) = csv_writer_clone.write_result(&result).await {
                                    warn!("Failed to write result to CSV: {}", e);
                                }

                                let count = csv_writer_clone.count().await;
                                if count % 100 == 0 {
                                    info!("Processed {} results", count);
                                }

                                debug!(
                                    "Result {} seq={} mode={} detections={}",
                                    count,
                                    result.sequence_id,
                                    result.timing.mode_str(),
                                    result.inference.detection_count
                                );
                            }
                            None => break,
                        }
                    }
                }
            }
            info!("Result collection task finished");
        });

        let device_readers = connections.take_all_readers();

        let mut read_handles: Vec<JoinHandle<()>> = Vec::new();

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

        connections.signal_begin().await?;
        info!("Experiment started!");

        let start = Instant::now();
        let mut sequence_id: u64 = 0;
        let mut frame_number: u64 = 1;
        let pulse_interval = fps_to_interval(config.fixed_fps);
        let frame_skip = compute_skip(config.fixed_fps);

        let mut expected_results = 0u64;

        while start.elapsed().as_secs() < config.duration_seconds {
            let timing = TimingMetadata {
                sequence_id,
                frame_number,
                source_device: String::new(),
                mode: Some(config.mode.clone()),
                controller_sent_pulse: Some(current_timestamp_micros()),
                ..Default::default()
            };

            let pulse_msg = Message::Pulse(timing);

            let mut pulse_sent = false;
            for i in 0..config.num_roadside_units {
                let rsu_id = DeviceId::RoadsideUnit(i);
                if let Some(sender) = connections.get_sender(&rsu_id) {
                    if sender.send(pulse_msg.clone()).await.is_ok() {
                        pulse_sent = true;
                    } else {
                        warn!("Failed to send pulse to {}", rsu_id);
                    }
                }
            }

            if pulse_sent {
                sequence_id += 1;
                frame_number = Self::advance_frame(frame_number, frame_skip, MAX_FRAME_SEQUENCE);
                expected_results =
                    expected_results.saturating_add(u64::from(config.num_roadside_units));
            }

            sleep(pulse_interval).await;
        }

        info!("Finished sending {} pulses. Waiting for results...", expected_results);

        let wait_start = Instant::now();

        while wait_start.elapsed() < MAX_WAIT {
            let current_count = csv_writer.count().await;
            if current_count >= usize::try_from(expected_results)? {
                info!("All {} results received!", current_count);
                break;
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

        info!("Sending shutdown signal to all devices...");
        connections.signal_shutdown().await;

        let _ = stop_tx.send(true);
        drop(result_tx);

        info!("Waiting for device read tasks to complete...");
        let read_timeout = Duration::from_secs(5);

        for (idx, handle) in read_handles.into_iter().enumerate() {
            match timeout(read_timeout, handle).await {
                Ok(Ok(())) => debug!("Read task {} completed", idx),
                Ok(Err(e)) => warn!("Read task {} panicked: {:?}", idx, e),
                Err(_) => warn!("Read task {} timed out", idx),
            }
        }

        match timeout(Duration::from_secs(2), result_collection_task).await {
            Ok(Ok(())) => debug!("Result collection task completed"),
            Ok(Err(e)) => warn!("Result collection task panicked: {:?}", e),
            Err(_) => warn!("Result collection task timed out"),
        }

        let final_count = csv_writer.finalize()?;

        info!("========================================");
        info!("Experiment {} completed", config.experiment_id);
        info!("Results saved: {}", final_count);
        info!("========================================");

        sleep(Duration::from_secs(1)).await;

        Ok(())
    }
}

impl Default for ControllerHarness {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for ConcurrentCsvWriter {
    fn clone(&self) -> Self {
        Self { inner: Arc::clone(&self.inner), config: self.config.clone() }
    }
}
