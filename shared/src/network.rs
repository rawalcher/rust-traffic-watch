use bincode::Options;
use log::{error};
use std::error::Error;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::TcpStream;
use tokio::sync::mpsc;

use crate::Message;

#[inline]
fn bconf() -> impl Options {
    bincode::DefaultOptions::new().with_fixint_encoding()
}

#[inline]
fn frame_message(msg: &Message) -> Result<Vec<u8>, Box<dyn Error + Send + Sync>> {
    let body = bconf().serialize(msg)?;
    let mut framed = Vec::with_capacity(4 + body.len());
    framed.extend_from_slice(&(body.len() as u32).to_le_bytes());
    framed.extend_from_slice(&body);
    Ok(framed)
}

pub async fn send_message(
    writer: &mut OwnedWriteHalf,
    message: &Message,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let buf = frame_message(message)?;
    writer.write_all(&buf).await?;
    Ok(())
}

pub fn spawn_writer(wr: OwnedWriteHalf) -> mpsc::Sender<Message> {
    spawn_writer_with_capacity(wr, 1024)
}

pub fn spawn_writer_with_capacity(wr: OwnedWriteHalf, cap: usize) -> mpsc::Sender<Message> {
    let (tx, mut rx) = mpsc::channel::<Message>(cap);
    tokio::spawn(async move {
        let mut wr = wr;
        while let Some(msg) = rx.recv().await {
            match frame_message(&msg) {
                Ok(buf) => {
                    if let Err(e) = wr.write_all(&buf).await {
                        error!("writer: socket write error: {e}");
                        break; // socket died; exit task
                    }
                }
                Err(e) => error!("writer: serialize error: {e}"),
            }
        }
    });
    tx
}

pub async fn read_message(
    reader: &mut OwnedReadHalf,
) -> Result<Message, Box<dyn Error + Send + Sync>> {
    let mut size_buf = [0u8; 4];
    reader.read_exact(&mut size_buf).await?;
    let size = u32::from_le_bytes(size_buf) as usize;

    let mut buffer = vec![0u8; size];
    reader.read_exact(&mut buffer).await?;

    match bconf().deserialize::<Message>(&buffer) {
        Ok(msg) => Ok(msg),
        Err(e) => {
            error!("Deserialize failed: {e}. Closing connection.");
            Err(Box::<dyn Error + Send + Sync>::from(e))
        }
    }
}

pub async fn read_message_stream(
    stream: &mut TcpStream,
) -> Result<Message, Box<dyn Error + Send + Sync>> {
    let mut size_buf = [0u8; 4];
    stream.read_exact(&mut size_buf).await?;
    let size = u32::from_le_bytes(size_buf) as usize;

    let mut buffer = vec![0u8; size];
    stream.read_exact(&mut buffer).await?;

    Ok(bconf().deserialize::<Message>(&buffer)?)
}
