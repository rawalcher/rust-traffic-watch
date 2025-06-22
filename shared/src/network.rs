use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::Mutex;
use std::sync::Arc;
use crate::types::*;
use log::{debug, warn};

static CONTROLLER_CONNECTION: tokio::sync::OnceCell<Arc<Mutex<Option<TcpStream>>>> = tokio::sync::OnceCell::const_new();

async fn get_controller_connection() -> &'static Arc<Mutex<Option<TcpStream>>> {
    CONTROLLER_CONNECTION.get_or_init(|| async {
        Arc::new(Mutex::new(None))
    }).await
}

pub async fn send_message<T: serde::Serialize>(
    stream: &mut TcpStream,
    message: &T,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let serialized = bincode::serialize(message)?;
    let size = serialized.len() as u32;

    stream.write_all(&size.to_le_bytes()).await?;
    stream.write_all(&serialized).await?;
    stream.flush().await?;
    Ok(())
}

pub async fn receive_message<T: for<'de> serde::Deserialize<'de>>(
    stream: &mut TcpStream,
) -> Result<T, Box<dyn std::error::Error + Send + Sync>> {
    let mut size_buf = [0u8; 4];
    stream.read_exact(&mut size_buf).await?;
    let size = u32::from_le_bytes(size_buf) as usize;

    let mut buffer = vec![0u8; size];
    stream.read_exact(&mut buffer).await?;

    Ok(bincode::deserialize(&buffer)?)
}

pub async fn send_result_to_controller(
    timing: &TimingPayload,
    inference: InferenceResult,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let result = ProcessingResult {
        timing: timing.clone(),
        inference,
    };

    let message = ControlMessage::ProcessingResult(result);

    let connection = get_controller_connection().await;
    let mut stream_guard = connection.lock().await;

    if stream_guard.is_none() {
        let controller_addr = format!("{}:{}", crate::constants::CONTROLLER_ADDRESS, crate::constants::CONTROLLER_PORT);
        let new_stream = TcpStream::connect(&controller_addr).await?;
        *stream_guard = Some(new_stream);
        debug!("Created controller connection");
    }

    if let Some(ref mut stream) = *stream_guard {
        match send_message(stream, &message).await {
            Ok(()) => return Ok(()),
            Err(e) => {
                warn!("Controller connection failed, reconnecting: {}", e);
                *stream_guard = None;
            }
        }
    }

    let controller_addr = format!("{}:{}", crate::constants::CONTROLLER_ADDRESS, crate::constants::CONTROLLER_PORT);
    let mut new_stream = TcpStream::connect(&controller_addr).await?;
    send_message(&mut new_stream, &message).await?;
    *stream_guard = Some(new_stream);
    debug!("Reconnected to controller");

    Ok(())
}