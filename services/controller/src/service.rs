use std::error::Error;
use std::sync::Arc;
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use tokio::time::{sleep, timeout, Duration, Instant};
use tracing::{debug, info, warn};

use crate::csv_writer::ConcurrentCsvWriter;
use network::connection;
use network::connection::{
    get_device_sender, start_controller_listener, wait_for_device_readiness, wait_for_devices, Role,
};
use protocol::config::{compute_skip, fps_to_interval, MAX_FRAME_SEQUENCE};
use protocol::types::{ExperimentConfig, ExperimentMode};
use protocol::{
    current_timestamp_micros, ControlMessage, DeviceId, InferenceMessage, Message, TimingMetadata,
};

pub struct ControllerHarness {
    active_sink_tx: watch::Sender<Option<mpsc::UnboundedSender<InferenceMessage>>>,
    _forwarder_task: JoinHandle<()>,
    _listener_task: JoinHandle<()>,
}

impl ControllerHarness {
    pub fn new() -> Self {
        let (raw_tx, mut raw_rx) = mpsc::unbounded_channel::<InferenceMessage>();

        let listener_task = tokio::spawn(async move {
            if let Err(e) =
                start_controller_listener(Role::Controller { result_handler: raw_tx }).await
            {
                warn!("controller listener exited with error: {:?}", e);
            }
        });

        let (active_sink_tx, active_sink_rx) =
            watch::channel::<Option<mpsc::UnboundedSender<InferenceMessage>>>(None);

        let forwarder_task = tokio::spawn(async move {
            let sink_rx = active_sink_rx;
            while let Some(msg) = raw_rx.recv().await {
                if let Some(sink) = sink_rx.borrow().clone() {
                    let _ = sink.send(msg);
                }
            }
        });

        Self { active_sink_tx, _forwarder_task: forwarder_task, _listener_task: listener_task }
    }

    pub async fn run_controller_with_retry(
        &self,
        config: ExperimentConfig,
        max_retries: u32,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let mut attempt = 0;

        loop {
            attempt += 1;

            return match self.run_controller(config.clone()).await {
                Ok(()) => Ok(()),
                Err(e) if attempt <= max_retries => {
                    if e.to_string().contains("disconnected")
                        || e.to_string().contains("writer channel closed")
                    {
                        warn!(
                            "Device disconnected. Restarting experiment in 30s (attempt {}/{})",
                            attempt + 1,
                            max_retries + 1
                        );
                        sleep(Duration::from_secs(30)).await;
                        continue;
                    }
                    Err(e)
                }
                Err(e) => Err(e),
            };
        }
    }

    pub fn advance_frame(frame: u64, skip: u64, max: u64) -> u64 {
        ((frame + skip - 1) % max) + 1
    }

    pub async fn run_controller(
        &self,
        config: ExperimentConfig,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        info!("Starting experiment: {}", config.experiment_id);
        info!(
            "Mode: {:?}, Model: {}, FPS: {}, Duration: {}s",
            config.mode, config.model_name, config.fixed_fps, config.duration_seconds
        );

        if config.num_roadside_units == 0 {
            return Err("Cannot run experiment with 0 Roadside Units.".into());
        }

        connection::clear_ready_devices().await;

        let (inference_tx, mut inference_rx) = mpsc::unbounded_channel::<InferenceMessage>();

        // CHANGED: Use streaming CSV writer instead of buffering all results
        let csv_writer = ConcurrentCsvWriter::new(&config.experiment_id, config.clone())?;

        let _ = self.active_sink_tx.send(Some(inference_tx.clone()));

        let (stop_tx, mut stop_rx) = watch::channel(false);

        // CHANGED: Write results directly to CSV as they arrive
        let csv_writer_clone = csv_writer.clone();
        let result_collection_task = tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = stop_rx.changed() => {
                        if *stop_rx.borrow() { break; }
                    }
                    maybe = inference_rx.recv() => {
                        match maybe {
                            Some(mut result) => {
                                result.timing.controller_received = Some(current_timestamp_micros());

                                // Write to CSV immediately
                                if let Err(e) = csv_writer_clone.write_result(&result).await {
                                    warn!("Failed to write result to CSV: {}", e);
                                }

                                let count = csv_writer_clone.count().await;
                                if count % 100 == 0 {
                                    info!("Processed {} results", count);
                                }

                                debug!(
                                    "Controller received result {} for seq={} (mode={}, detections={})",
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

        let rsu_devices: Vec<DeviceId> =
            (0..config.num_roadside_units).map(|i| DeviceId::RoadsideUnit(i)).collect();

        let required_devices = match config.mode {
            ExperimentMode::Local => rsu_devices,
            ExperimentMode::Offload => {
                let mut devices = rsu_devices;
                devices.push(DeviceId::ZoneProcessor(0));
                devices
            }
        };

        info!("Waiting for required devices: {:?}", required_devices);
        wait_for_devices(&required_devices).await;

        let msg = Message::Control(ControlMessage::ConfigureExperiment { config: config.clone() });
        for id in &required_devices {
            if let Some(sender) = get_device_sender(id).await {
                sender.send(msg.clone())?;
            }
        }

        wait_for_device_readiness(&required_devices).await;

        let begin_msg = Message::Control(ControlMessage::BeginExperiment);
        for id in &required_devices {
            if let Some(sender) = get_device_sender(id).await {
                sender.send(begin_msg.clone())?;
            }
        }

        let start = Instant::now();
        let mut sequence_id: u64 = 0;
        let mut frame_number: u64 = 1;
        let pulse_interval = fps_to_interval(config.fixed_fps);

        let mut expected_results = 0u64;
        let frame_skip = compute_skip(config.fixed_fps);

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
                if let Some(sender) = get_device_sender(&rsu_id).await {
                    sender.send(pulse_msg.clone())?;
                    debug!("Sent pulse {} to {} (mode={:?})", sequence_id, rsu_id, config.mode);
                    pulse_sent = true;
                } else {
                    warn!("Could not get sender for {}. Skipping pulse.", rsu_id);
                }
            }

            if pulse_sent {
                sequence_id += 1;
                frame_number = Self::advance_frame(frame_number, frame_skip, MAX_FRAME_SEQUENCE);
                expected_results =
                    expected_results.saturating_add(config.num_roadside_units as u64);
            }

            sleep(pulse_interval).await;
        }

        info!(
            "Finished sending {} pulses ({} pulses/RSU). Waiting 5 seconds for results...",
            expected_results,
            expected_results / (config.num_roadside_units as u64)
        );
        sleep(Duration::from_secs(5)).await;

        // CHANGED: Get count from CSV writer instead of in-memory buffer
        let final_count = csv_writer.count().await;
        info!(
            "Collected {} results out of {} pulses sent ({:.1}% success rate)",
            final_count,
            expected_results,
            (final_count as f32 / expected_results as f32) * 100.0
        );

        let shutdown = Message::Control(ControlMessage::Shutdown);
        for id in &required_devices {
            if let Some(sender) = get_device_sender(id).await {
                let _ = sender.send(shutdown.clone());
            }
        }

        let _ = stop_tx.send(true);
        let _ = self.active_sink_tx.send(None);
        drop(inference_tx);

        match timeout(Duration::from_secs(2), result_collection_task).await {
            Ok(joined) => {
                if let Err(e) = joined {
                    warn!("Result collection task error: {:?}", e);
                }
            }
            Err(_) => {
                warn!("Result collector did not stop in time; it will end soon.");
            }
        }

        info!("Shutdown complete");

        // CHANGED: Finalize CSV writer instead of batch writing
        let final_count = csv_writer.finalize().await?;
        info!("Experiment {} completed with {} results saved", config.experiment_id, final_count);
        Ok(())
    }
}

impl Clone for ConcurrentCsvWriter {
    fn clone(&self) -> Self {
        Self { inner: Arc::clone(&self.inner), config: self.config.clone() }
    }
}
