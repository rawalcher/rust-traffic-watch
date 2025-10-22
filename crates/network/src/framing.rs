use anyhow::Result;
use log::error;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::TcpStream;
use tokio::sync::mpsc;
use protocol::Message;

#[inline]
pub fn frame_message(msg: &Message) -> Result<Vec<u8>> {
    let body = postcard::to_allocvec(msg)?;
    let mut framed = Vec::with_capacity(4 + body.len());
    framed.extend_from_slice(&(body.len() as u32).to_le_bytes());
    framed.extend_from_slice(&body);
    Ok(framed)
}

#[inline]
pub fn unframe_message(buf: &[u8]) -> Result<Message> {
    Ok(postcard::from_bytes(buf)?)
}

pub async fn send_message(writer: &mut OwnedWriteHalf, message: &Message) -> Result<()> {
    let buf = frame_message(message)?;
    writer.write_all(&buf).await?;
    Ok(())
}

pub fn spawn_writer(writer: OwnedWriteHalf, capacity: usize) -> mpsc::Sender<Message> {
    let (tx, mut rx) = mpsc::channel::<Message>(capacity);
    tokio::spawn(async move {
        let mut writer = writer;
        while let Some(msg) = rx.recv().await {
            match frame_message(&msg) {
                Ok(buf) => {
                    if let Err(e) = writer.write_all(&buf).await {
                        error!("writer: socket write error: {e}");
                        break;
                    }
                }
                Err(e) => error!("writer: serialize error: {e}"),
            }
        }
    });
    tx
}

pub async fn read_message(reader: &mut OwnedReadHalf) -> Result<Message> {
    let mut size_buf = [0u8; 4];
    reader.read_exact(&mut size_buf).await?;
    let size = u32::from_le_bytes(size_buf) as usize;

    let mut buffer = vec![0u8; size];
    reader.read_exact(&mut buffer).await?;

    Ok(postcard::from_bytes(&buffer)?)
}

pub async fn read_message_stream(stream: &mut TcpStream) -> Result<Message> {
    let mut size_buf = [0u8; 4];
    stream.read_exact(&mut size_buf).await?;
    let size = u32::from_le_bytes(size_buf) as usize;

    let mut buffer = vec![0u8; size];
    stream.read_exact(&mut buffer).await?;

    Ok(postcard::from_bytes(&buffer)?)
}
