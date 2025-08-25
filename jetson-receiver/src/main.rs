use log::{debug, error, info, warn};
use shared::constants::*;
use shared::{
    current_timestamp_micros, perform_python_inference_with_counts,
    ControlMessage, DeviceId, ExperimentConfig, FrameMessage,
    InferenceMessage, Message, PersistentPythonDetector,
};
use std::error::Error;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::net::TcpStream;
use tokio::net::tcp::{OwnedReadHalf};
use tokio::sync::mpsc;
use shared::connection::{wait_for_pi_on_jetson, Role};
use shared::network::{read_message, send_message};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    env_logger::init();
    info!("Jetson connecting to controller at {}", controller_address());

    let controller_stream = TcpStream::connect(controller_address()).await?;
    let (mut ctrl_reader, mut ctrl_writer) = controller_stream.into_split();

    send_message(&mut ctrl_writer, &Message::Hello(DeviceId::Jetson)).await?;

    let config = wait_for_experiment_config(&mut ctrl_reader).await?;

    let pending_frame: Arc<Mutex<Option<FrameMessage>>> = Arc::new(Mutex::new(None));
    let pending_frame_network = Arc::clone(&pending_frame);
    let pending_frame_inference = Arc::clone(&pending_frame);

    let (frame_tx, mut frame_rx) = mpsc::unbounded_channel();

    debug!("Starting Pi connection handler...");
    tokio::spawn(async move {
        if let Err(e) = wait_for_pi_on_jetson(Role::Jetson { frame_handler: frame_tx }).await {
            error!("Pi connection handler failed: {:?}", e);
        }
    });

    tokio::spawn(async move {
        while let Some(frame) = frame_rx.recv().await {
            let seq_id = frame.sequence_id;
            let mut pending = pending_frame_network.lock().await;
            let old_seq = pending.as_ref().map(|f| f.sequence_id);
            *pending = Some(frame);

            if let Some(old) = old_seq {
                debug!("Jetson: Dropped frame {} for newer frame {}", old, seq_id);
            }
            info!("Jetson: Updated pending frame to sequence_id={}", seq_id);
        }
    });

    info!("Pi connection handler started, initializing detector...");

    let model_name = config.model_name.clone();
    let mut detector = tokio::task::spawn_blocking(move || {
        PersistentPythonDetector::new(
            model_name,
            INFERENCE_TENSORRT_PATH.to_string(),
        )
    })
        .await??;

    info!("Python detector ready, sending ReadyToStart");

    send_message(&mut ctrl_writer, &Message::Control(ControlMessage::ReadyToStart)).await?;

    wait_for_experiment_start(&mut ctrl_reader).await?;
    info!("Experiment started. Processing frames...");

    let (result_tx, mut result_rx) = mpsc::unbounded_channel::<InferenceMessage>();

    let inference_config = config.clone();
    let pending_frame_for_inference = Arc::clone(&pending_frame_inference);

    let inference_task = tokio::spawn(async move {
        loop {
            let frame = {
                let mut pending = pending_frame_for_inference.lock().await;
                pending.take()
            };

            if let Some(frame_msg) = frame {
                info!("Jetson: Starting inference for sequence_id={}", frame_msg.sequence_id);

                match handle_frame(frame_msg, &mut detector, &inference_config).await {
                    Ok(result) => {
                        let seq_id = result.sequence_id;
                        if let Err(e) = result_tx.send(result) {
                            error!("Failed to send result to channel: {}", e);
                            break;
                        }
                        info!("Jetson: Completed inference for sequence_id={}", seq_id);
                    }
                    Err(e) => {
                        error!("Failed to process frame: {}", e);
                    }
                }
            } else {
                tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
            }
        }
        detector.shutdown()
    });

    loop {
        tokio::select! {
            Some(result) = result_rx.recv() => {
                send_message(&mut ctrl_writer, &Message::Result(result)).await?;
            }
            msg = read_message(&mut ctrl_reader) => {
                match msg? {
                    Message::Control(ControlMessage::Shutdown) => {
                        info!("Shutdown received");
                        inference_task.abort();
                        break;
                    }
                    unexpected => {
                        warn!("Unexpected controller message: {:?}", unexpected);
                    }
                }
            }
        }
    }

    info!("Jetson done. Exiting.");
    Ok(())
}

async fn wait_for_experiment_config(
    reader: &mut OwnedReadHalf,
) -> Result<ExperimentConfig, Box<dyn Error + Send + Sync>> {
    wait_for_control_message(reader, |msg| {
        if let ControlMessage::StartExperiment { config } = msg {
            Some(config)
        } else {
            None
        }
    }, "experiment config").await
}

async fn wait_for_experiment_start(
    reader: &mut OwnedReadHalf,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    wait_for_control_message(reader, |msg| {
        if let ControlMessage::BeginExperiment = msg {
            Some(())
        } else {
            None
        }
    }, "experiment start").await
}

async fn handle_frame(
    mut frame: FrameMessage,
    detector: &mut PersistentPythonDetector,
    config: &ExperimentConfig,
) -> Result<InferenceMessage, Box<dyn Error + Send + Sync>> {

    frame.timing.jetson_received = Some(current_timestamp_micros());

    debug!("Processing frame {} with {} bytes", frame.sequence_id, frame.frame_data.len());

    let inference = perform_python_inference_with_counts(
        &frame,
        detector,
        &config.model_name,
        "offload",
    )?;

    info!("Jetson inference complete for sequence_id: {}, detections: {}, size: {}x{}",
          frame.sequence_id, inference.detection_count, inference.image_width, inference.image_height);

    frame.timing.jetson_sent_result = Some(current_timestamp_micros());

    Ok(InferenceMessage {
        sequence_id: frame.sequence_id,
        timing: frame.timing,
        inference,
    })
}

async fn wait_for_control_message<F, T>(
    reader: &mut OwnedReadHalf,
    matcher: F,
    context: &str,
) -> Result<T, Box<dyn Error + Send + Sync>>
where
    F: Fn(ControlMessage) -> Option<T>,
{
    loop {
        match read_message(reader).await? {
            Message::Control(ctrl_msg) => {
                match ctrl_msg {
                    ControlMessage::Shutdown => {
                        return Err(format!("Shutdown during {}", context).into());
                    }
                    other => {
                        if let Some(result) = matcher(other.clone()) {
                            return Ok(result);
                        }
                        warn!("Unexpected control message while waiting for {}: {:?}", context, other);
                    }
                }
            }
            other => {
                warn!("Expected Control message while waiting for {}, got: {:?}", context, other);
            }
        }
    }
}