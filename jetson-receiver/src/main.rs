use log::{debug, error, info, warn};
use shared::constants::*;
use shared::{
    current_timestamp_micros, perform_python_inference_with_counts,
    ControlMessage, DeviceId, ExperimentConfig, FrameMessage,
    InferenceMessage, Message, PersistentPythonDetector,
};
use std::error::Error;
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

    let (frame_tx, mut frame_rx) = mpsc::unbounded_channel();

 
    debug!("test1");
    wait_for_pi_on_jetson(Role::Jetson { frame_handler: frame_tx }).await?;
    info!("Pi connected, Jetson data listener running");

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
    info!("Experiment started. Waiting for frames from Pi...");

    loop {
        tokio::select! {
            Some(frame_msg) = frame_rx.recv() => {
                match handle_frame(frame_msg, &mut detector, &config).await {
                    Ok(result) => {
                        send_message(&mut ctrl_writer, &Message::Result(result)).await?;
                    }
                    Err(e) => {
                        error!("Failed to process frame: {}", e);
                    }
                }
            }

            msg = read_message(&mut ctrl_reader) => {
                match msg? {
                    Message::Control(ControlMessage::Shutdown) => {
                        info!("Shutdown received");
                        break;
                    }
                    unexpected => {
                        warn!("Unexpected controller message: {:?}", unexpected);
                    }
                }
            }
        }
    }

    detector.shutdown()?;
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
    frame.timing.jetson_inference_start = Some(current_timestamp_micros());

    let inference = perform_python_inference_with_counts(
        &mut frame,
        detector,
        &config.model_name,
        "offload",
    )?;

    frame.timing.jetson_inference_complete = Some(current_timestamp_micros());
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

