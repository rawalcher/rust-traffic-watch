use protocol::types::ExperimentConfig;
use protocol::{FrameMessage, InferenceMessage};
use shared::python_detector::PersistentPythonDetector;
use std::error::Error;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, watch, Mutex};
use tokio::time::{sleep, timeout};
use tracing::{debug, error, info, warn};

pub struct ExperimentManager {
    detector: Arc<Mutex<Option<PersistentPythonDetector>>>,
    pending_frame: Arc<Mutex<Option<FrameMessage>>>,
    shutdown_tx: watch::Sender<bool>,
    shutdown_rx: watch::Receiver<bool>,
    inference_task: Option<tokio::task::JoinHandle<()>>,
}

impl ExperimentManager {
    /// # Errors
    ///
    /// Will return an `Err` if inference failed.
    pub fn new(
        model_name: String,
        script_path: String,
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        info!("Initializing experiment with model '{}'", model_name);

        let detector = PersistentPythonDetector::new(model_name, script_path)?;
        let (shutdown_tx, shutdown_rx) = watch::channel(false);

        Ok(Self {
            detector: Arc::new(Mutex::new(Some(detector))),
            pending_frame: Arc::new(Mutex::new(None)),
            shutdown_tx,
            shutdown_rx,
            inference_task: None,
        })
    }

    pub async fn is_ready(&self) -> bool {
        let mut detector_opt = self.detector.lock().await;
        detector_opt.as_mut().is_some_and(PersistentPythonDetector::is_alive)
    }

    pub fn start_inference<F, Fut>(
        &mut self,
        result_tx: mpsc::UnboundedSender<InferenceMessage>,
        config: ExperimentConfig,
        inference_fn: F,
    ) where
        F: Fn(FrameMessage, Arc<Mutex<Option<PersistentPythonDetector>>>, ExperimentConfig) -> Fut
            + Send
            + 'static,
        Fut: Future<Output = Result<InferenceMessage, Box<dyn Error + Send + Sync>>> + Send,
    {
        let pending_frame = Arc::clone(&self.pending_frame);
        let detector = Arc::clone(&self.detector);
        let mut shutdown_rx = self.shutdown_rx.clone();

        let task = tokio::spawn(async move {
            loop {
                if *shutdown_rx.borrow() {
                    info!("Inference task: shutdown signal received");
                    break;
                }

                let frame_opt = {
                    let mut pending = pending_frame.lock().await;
                    pending.take()
                };

                if let Some(frame_msg) = frame_opt {
                    if *shutdown_rx.borrow() {
                        info!(
                            "Inference task: dropping frame {} due to shutdown",
                            frame_msg.sequence_id
                        );
                        break;
                    }

                    let seq_id = frame_msg.sequence_id;
                    debug!("Starting inference for sequence_id={}", seq_id);

                    match inference_fn(frame_msg, Arc::clone(&detector), config.clone()).await {
                        Ok(inference_msg) => {
                            if result_tx.send(inference_msg).is_err() {
                                info!("Result channel closed, inference task exiting");
                                break;
                            }
                            info!("Completed inference for sequence_id={}", seq_id);
                        }
                        Err(e) => {
                            error!("Inference failed for sequence_id={}: {}", seq_id, e);
                        }
                    }
                } else {
                    tokio::select! {
                        _ = shutdown_rx.changed() => {
                            if *shutdown_rx.borrow() {
                                break;
                            }
                        }
                        () = sleep(Duration::from_millis(10)) => {}
                    }
                }
            }

            info!("Inference task loop ended");
        });

        self.inference_task = Some(task);
    }

    pub async fn update_pending_frame(&self, frame: FrameMessage) {
        let mut pending = self.pending_frame.lock().await;
        let seq_id = frame.sequence_id;

        if let Some(old) = pending.replace(frame) {
            info!("Dropped frame {} for newer frame {}", old.sequence_id, seq_id);
        }
        info!("Updated pending frame to sequence_id={}", seq_id);
    }

    pub async fn clear_pending_frame(&self) {
        let mut pending = self.pending_frame.lock().await;
        *pending = None;
    }

    /// # Errors
    ///
    /// Will return `Err` if Detector does not properly Shutdown
    pub async fn shutdown(&mut self) -> Result<(), String> {
        info!("Beginning experiment shutdown sequence");

        // Step 1: Clear pending work
        self.clear_pending_frame().await;
        info!("Cleared pending frames");

        // Step 2: Signal inference task to stop
        let _ = self.shutdown_tx.send(true);
        info!("Sent shutdown signal to inference task");

        // Step 3: Wait briefly for task to see the signal
        sleep(Duration::from_millis(50)).await;

        // Step 4: Shutdown Python detector (blocks until current inference completes)
        info!("Shutting down Python detector...");
        {
            let mut detector_opt = self.detector.lock().await;
            if let Some(mut detector) = detector_opt.take() {
                match detector.shutdown() {
                    Ok(()) => info!("Detector shutdown successful"),
                    Err(e) => error!("Detector shutdown error: {}", e),
                }
            }
        }

        // Step 5: Wait for inference task to complete
        if let Some(task) = self.inference_task.take() {
            match timeout(Duration::from_secs(2), task).await {
                Ok(Ok(())) => info!("Inference task completed cleanly"),
                Ok(Err(e)) => error!("Inference task panicked: {:?}", e),
                Err(_) => warn!("Inference task did not complete in time (likely already stopped)"),
            }
        }

        info!("Experiment shutdown complete");
        Ok(())
    }
}

impl Drop for ExperimentManager {
    fn drop(&mut self) {
        if self.inference_task.is_some() {
            warn!("ExperimentManager dropped with running inference task!");
        }
    }
}
