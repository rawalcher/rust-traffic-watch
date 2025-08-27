use log::{debug, error};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{tcp::{OwnedReadHalf, OwnedWriteHalf}, TcpStream};
use crate::Message;

pub async fn send_message(
    writer: &mut OwnedWriteHalf,
    message: &Message,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let serialized = bincode::serialize(message)?;
    let size = serialized.len() as u32;

    debug!("Sending message of {} bytes", size);

    writer.write_all(&size.to_le_bytes()).await?;
    writer.write_all(&serialized).await?;
    writer.flush().await?;
    Ok(())
}

pub async fn read_message(
    reader: &mut OwnedReadHalf,
) -> Result<Message, Box<dyn std::error::Error + Send + Sync>> {
    loop {
        let mut size_buf = [0u8; 4];
        reader.read_exact(&mut size_buf).await?;
        let size = u32::from_le_bytes(size_buf) as usize;

        let mut buffer = vec![0u8; size];
        reader.read_exact(&mut buffer).await?;

        match bincode::deserialize(&buffer) {
            Ok(msg) => {
                debug!("Received message of {} bytes", size);
                return Ok(msg);
            }
            Err(_e) => {
                error!("Corrupted message (probably sequence_id={}), dropping and continuing",
                       if buffer.len() >= 8 {
                           u64::from_le_bytes(buffer[0..8].try_into().unwrap_or([0;8]))
                       } else { 0 });
                // we do not care, continue
                // sometimes it tries to match sequence number of Pulse as Enum Messagetype
                continue;
            }
        }
    }
}

pub async fn send_message_stream(
    stream: &mut TcpStream,
    message: &Message,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let serialized = bincode::serialize(message)?;
    let size = serialized.len() as u32;

    stream.write_all(&size.to_le_bytes()).await?;
    stream.write_all(&serialized).await?;
    stream.flush().await?;
    Ok(())
}

pub async fn read_message_stream(
    stream: &mut TcpStream,
) -> Result<Message, Box<dyn std::error::Error + Send + Sync>> {
    let mut size_buf = [0u8; 4];
    stream.read_exact(&mut size_buf).await?;
    let size = u32::from_le_bytes(size_buf) as usize;

    let mut buffer = vec![0u8; size];
    stream.read_exact(&mut buffer).await?;

    Ok(bincode::deserialize(&buffer)?)
}
