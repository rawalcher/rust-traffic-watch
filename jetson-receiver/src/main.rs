use shared::FrameMessage;
use tokio::io::AsyncReadExt;
use tokio::net::{TcpListener, TcpStream};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Jetson Receiver starting...");

    let listener = TcpListener::bind("0.0.0.0:8080").await?;
    println!("Listening on port 8080...");

    let (stream, addr) = listener.accept().await?;
    println!("Pi connected from: {}", addr);

    process_frames(stream).await?;

    Ok(())
}

async fn process_frames(mut stream: TcpStream) -> Result<(), Box<dyn std::error::Error>> {
    let mut buffer = vec![0u8; 4];

    loop {
        match stream.read_exact(&mut buffer).await {
            Ok(_) => {},
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                println!("Pi disconnected");
                break;
            },
            Err(e) => return Err(e.into()),
        }

        let frame_size = u32::from_le_bytes([buffer[0], buffer[1], buffer[2], buffer[3]]) as usize;

        let mut frame_buffer = vec![0u8; frame_size];
        stream.read_exact(&mut frame_buffer).await?;

        let frame: FrameMessage = bincode::deserialize(&frame_buffer)?;

        println!(
            "Received frame {} - {}x{} - {} bytes - timestamp: {}",
            frame.sequence_id,
            frame.width,
            frame.height,
            frame.frame_data.len(),
            frame.timestamp
        );

        // TODO
        // Decompress the frame
        // Send to Python subprocess
        // Process results and send control messages back

        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }

    Ok(())
}