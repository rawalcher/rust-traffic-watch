use shared::FrameMessage;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;
use tokio::time::{sleep, Duration};
use tracing::{info, debug, error, warn};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    info!("Pi Sender starting...");

    let jetson_addr = "localhost:8080"; // TODO change
    let mut stream = TcpStream::connect(jetson_addr).await?;
    info!("Connected to Jetson at {}", jetson_addr);

    let mut sequence_id = 1u64;

    loop {
        let frame_data = match load_frame_from_image(sequence_id) {
            Ok(data) => data,
            Err(e) => {
                error!("Error loading frame {}: {}", sequence_id, e);
                break;
            }
        };

        let frame = FrameMessage {
            sequence_id,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)?
                .as_millis() as u64,
            frame_data, 
            width: 1920,
            height: 1080,
        };

        let serialized = bincode::serialize(&frame)?;
        let frame_size = serialized.len() as u32;

        stream.write_all(&frame_size.to_le_bytes()).await?;
        stream.write_all(&serialized).await?;
        stream.flush().await?;

        info!("Sent frame {} ({} bytes total, {} bytes image)",
                sequence_id, frame_size, frame.frame_data.len());

        // sends every frame
        sequence_id += 1;
        sleep(Duration::from_millis(100)).await;

        if sequence_id > 999 {
            info!("Reached end of sequence (frame 999)");
            break;
        }
    }

    info!("Pi Sender finished");
    Ok(())
}

fn load_frame_from_image(frame_number: u64) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    use tracing::debug;

    let filename = format!("pi-sender/sample/seq3-drone_{:07}.jpg", frame_number);
    let image_bytes = std::fs::read(&filename)?;
    debug!("Loaded frame {} ({} bytes)", frame_number, image_bytes.len());
    Ok(image_bytes)
}