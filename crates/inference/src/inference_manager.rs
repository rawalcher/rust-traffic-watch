use crate::persistent::PersistentOnnxDetector;
use protocol::types::ExperimentConfig;
use protocol::{current_timestamp_micros, DeviceId, FrameMessage, InferenceMessage};

use std::collections::HashMap;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, watch, Mutex};
use tracing::{error, info};

pub struct InferenceManager {
    pending_frames: Arc<Mutex<HashMap<DeviceId, FrameMessage>>>,
    frame_notify: Arc<watch::Sender<()>>,
    shutdown_tx: watch::Sender<bool>,
    inference_task: Option<tokio::task::JoinHandle<()>>,
}

impl InferenceManager {
    /// # Errors
    pub fn new(
        model_name: String,
        models_dir: &Path,
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        info!("Initializing Rust inference with model '{}'", model_name);

        let _ = PersistentOnnxDetector::new(model_name, models_dir.to_path_buf())?;

        let pending_frames = Arc::new(Mutex::new(HashMap::<DeviceId, FrameMessage>::new()));
        let (frame_notify, _) = watch::channel(());
        let (shutdown_tx, _) = watch::channel(false);

        Ok(Self {
            pending_frames,
            frame_notify: Arc::new(frame_notify),
            shutdown_tx,
            inference_task: None,
        })
    }

    pub fn start_inference<F>(
        &mut self,
        result_tx: mpsc::UnboundedSender<InferenceMessage>,
        config: ExperimentConfig,
        inference_fn: F,
    ) where
        F: Fn(
                FrameMessage,
                &mut PersistentOnnxDetector,
                &ExperimentConfig,
            ) -> Result<InferenceMessage, Box<dyn Error + Send + Sync>>
            + Send
            + 'static,
    {
        let pending_frames = Arc::clone(&self.pending_frames);
        let mut frame_notify_rx = self.frame_notify.subscribe();
        let mut shutdown_rx = self.shutdown_tx.subscribe();

        let model_name = config.model_name.clone();
        let models_dir = PathBuf::from("models");
        let num_rsus = config.num_roadside_units;

        let task = tokio::spawn(async move {
            info!("Starting inference worker task");

            let mut detector = match PersistentOnnxDetector::new(model_name, models_dir) {
                Ok(d) => d,
                Err(e) => {
                    error!("Failed to initialize detector: {e}");
                    return;
                }
            };

            let rsu_order: Vec<DeviceId> = (0..num_rsus).map(DeviceId::RoadsideUnit).collect();
            let mut current_rsu_idx = 0;

            loop {
                let frames_to_process: Vec<FrameMessage> = {
                    let mut pending = pending_frames.lock().await;

                    if pending.is_empty() {
                        Vec::new()
                    } else {
                        let mut frames = Vec::new();

                        for _ in 0..rsu_order.len() {
                            let rsu_id = rsu_order[current_rsu_idx];
                            current_rsu_idx = (current_rsu_idx + 1) % rsu_order.len();

                            if let Some(frame) = pending.remove(&rsu_id) {
                                frames.push(frame);
                            }
                        }

                        frames
                    }
                };

                for frame in frames_to_process {
                    let seq = frame.sequence_id;

                    match inference_fn(frame, &mut detector, &config) {
                        Ok(msg) => {
                            info!(
                                seq = msg.sequence_id,
                                detections = msg.inference.detection_count,
                                time_ms = msg.inference.processing_time_us / 1000,
                                "Inference completed"
                            );
                            if result_tx.send(msg).is_err() {
                                error!("Result channel closed, shutting down");
                                return;
                            }
                        }
                        Err(e) => {
                            error!(seq = seq, error = %e, "Inference failed");
                        }
                    }
                }

                tokio::select! {
                    _ = shutdown_rx.changed() => {
                        if *shutdown_rx.borrow() {
                            break;
                        }
                    }

                    _ = frame_notify_rx.changed() => {
                    }

                    () = tokio::time::sleep(Duration::from_millis(10)) => {
                    }
                }
            }

            detector.shutdown();
            info!("Inference worker task stopped.");
        });

        self.inference_task = Some(task);
    }

    pub async fn update_pending_frame(&self, mut frame: FrameMessage) {
        let seq = frame.sequence_id;
        let device_id = frame.timing.source_device;
        tracing::debug!("update_pending_frame: device={device_id:?} seq={seq}");
        // if it fails, we throw the frame away anyway
        frame.timing.queued_for_inference = Some(current_timestamp_micros());
        {
            let mut pending = self.pending_frames.lock().await;
            if let Some(old) = pending.insert(device_id, frame) {
                tracing::warn!(
                    "FRAME OVERWRITTEN: device={device_id:?} old_seq={} new_seq={seq}",
                    old.sequence_id
                );
            }
        }
        let _ = self.frame_notify.send(());
    }

    pub async fn shutdown(&mut self) {
        info!("Stopping inference worker...");
        let _ = self.shutdown_tx.send(true);

        if let Some(task) = self.inference_task.take() {
            let _ = task.await;
        }

        self.pending_frames.lock().await.clear();

        info!("Inference manager shut down.");
    }
}
