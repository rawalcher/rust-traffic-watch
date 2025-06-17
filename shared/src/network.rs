use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use crate::types::*;

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
    let controller_addr = format!("{}:{}", crate::constants::CONTROLLER_ADDRESS, crate::constants::CONTROLLER_PORT);
    let mut controller_stream = TcpStream::connect(&controller_addr).await?;

    let result = ProcessingResult {
        timing: timing.clone(),
        inference,
    };

    let message = ControlMessage::ProcessingResult(result);
    send_message(&mut controller_stream, &message).await?;

    Ok(())
}