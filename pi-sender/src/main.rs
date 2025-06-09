use shared::FrameMessage;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Pi Sender starting...");

    let jetson_addr = "localhost:8080"; // TODO change
    let mut stream = TcpStream::connect(jetson_addr).await?;
    println!("Connected to Jetson at {}", jetson_addr);

    let mut sequence_id = 0u64;

    loop {
        let frame = FrameMessage {
            sequence_id,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)?
                .as_millis() as u64,
            frame_data: create_test_frame_data(sequence_id),
            width: 640,
            height: 480,
        };

        let serialized = bincode::serialize(&frame)?;
        let frame_size = serialized.len() as u32;

        stream.write_all(&frame_size.to_le_bytes()).await?;
        stream.write_all(&serialized).await?;
        stream.flush().await?;

        println!("Sent frame {} ({} bytes)", sequence_id, frame_size);

        sequence_id += 1;

        sleep(Duration::from_millis(100)).await;

        if sequence_id >= 50 {
            break;
        }
    }

    println!("Pi Sender finished");
    Ok(())
}

fn create_test_frame_data(sequence_id: u64) -> Vec<u8> {
    
    let size = 640 * 480 * 3;
    let mut data = vec![0u8; size];

    for (i, byte) in data.iter_mut().enumerate() {
        *byte = ((i as u64 + sequence_id) % 256) as u8;
    }

    data
}
