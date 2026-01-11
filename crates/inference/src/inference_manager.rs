use crate::persistent::PersistentOnnxDetector;
use protocol::types::ExperimentConfig;
use protocol::{FrameMessage, InferenceMessage};

use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;

use tokio::sync::{mpsc, watch};
use tracing::{error, info};

pub struct InferenceManager {
    frame_tx: Arc<watch::Sender<Option<FrameMessage>>>,
    shutdown_tx: watch::Sender<bool>,
    inference_task: Option<tokio::task::JoinHandle<()>>,
}

impl InferenceManager {
    pub fn new(
        model_name: String,
        models_dir: PathBuf,
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        info!("Initializing Rust inference with model '{}'", model_name);

        let _ = PersistentOnnxDetector::new(model_name, models_dir.clone())?;

        let (frame_tx, _frame_rx) = watch::channel::<Option<FrameMessage>>(None);
        let (shutdown_tx, _shutdown_rx) = watch::channel(false);

        Ok(Self { frame_tx: Arc::new(frame_tx), shutdown_tx, inference_task: None })
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
        let mut frame_rx = self.frame_tx.subscribe();
        let mut shutdown_rx = self.shutdown_tx.subscribe();

        let model_name = config.model_name.clone();
        let models_dir = PathBuf::from("models");

        let task = tokio::spawn(async move {
            info!("Starting inference worker task");

            let mut detector = match PersistentOnnxDetector::new(model_name, models_dir) {
                Ok(d) => d,
                Err(e) => {
                    error!("Failed to initialize detector: {e}");
                    return;
                }
            };

            loop {
                tokio::select! {
                    _ = shutdown_rx.changed() => {
                        if *shutdown_rx.borrow() {
                            break;
                        }
                    }

                    _ = frame_rx.changed() => {
                        let Some(frame) = frame_rx.borrow().clone() else {
                            continue;
                        };

                        let seq = frame.sequence_id;

                        match inference_fn(frame, &mut detector, &config) {
                            Ok(msg) => {
                                info!(
                                    "Inference completed: seq={} detections={} time={}ms",
                                    msg.sequence_id,
                                    msg.inference.detection_count,
                                    msg.inference.processing_time_us/1000,
                                );
                                if result_tx.send(msg).is_err() {
                                    break;
                                }
                            }
                            Err(e) => {
                                error!("Inference failed for {}: {}", seq, e);
                            }
                        }
                    }
                }
            }

            detector.shutdown();
            info!("Inference worker task stopped.");
        });

        self.inference_task = Some(task);
    }

    pub fn update_pending_frame(&self, frame: FrameMessage) {
        let _ = self.frame_tx.send(Some(frame));
    }

    pub async fn shutdown(&mut self) {
        info!("Stopping inference worker...");
        let _ = self.shutdown_tx.send(true);

        if let Some(task) = self.inference_task.take() {
            let _ = task.await;
        }

        info!("Inference manager shut down.");
    }
}
