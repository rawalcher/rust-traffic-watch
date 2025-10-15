use crate::types::*;
use log::{debug, error, warn};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

pub struct PersistentPythonDetector {
    process: Option<Child>,
    state: DetectorState,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum DetectorState {
    Ready,
    ShuttingDown,
    Terminated,
}

impl PersistentPythonDetector {
    pub fn new(model_name: String, script_path: String) -> Result<Self, String> {
        debug!("Launching Python detector with model '{}'", model_name);

        let mut process = Command::new("python3")
            .arg(script_path)
            .arg(&model_name)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Spawn failed: {}", e))?;

        let ready = Self::wait_for_ready(&mut process, Duration::from_secs(600))?;

        if !ready {
            let _ = process.kill();
            let _ = process.wait();
            return Err("Python process did not become ready in time".into());
        }

        debug!("Python process is ready (PID: {})", process.id());

        Ok(Self {
            process: Some(process),
            state: DetectorState::Ready,
        })
    }

    fn wait_for_ready(process: &mut Child, timeout: Duration) -> Result<bool, String> {
        use std::time::Instant;

        let stdout = process.stdout.as_mut().ok_or("Missing stdout")?;
        let mut reader = BufReader::new(stdout);
        let start = Instant::now();

        let mut line = String::new();
        loop {
            if start.elapsed() > timeout {
                return Ok(false);
            }

            match reader.read_line(&mut line) {
                Ok(0) => return Err("Python process closed stdout before READY".into()),
                Ok(_) => {
                    if line.trim().contains("READY") {
                        return Ok(true);
                    }
                    line.clear();
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(10));
                    continue;
                }
                Err(e) => return Err(format!("Failed to read from Python: {}", e)),
            }
        }
    }

    pub fn detect_objects(&mut self, image_bytes: &[u8]) -> Result<PythonDetectionResult, String> {
        if self.state != DetectorState::Ready {
            return Err(format!("Detector not ready (state: {:?})", self.state));
        }

        self.send_image(image_bytes)?;
        self.receive_response()
    }

    pub fn shutdown(&mut self) -> Result<(), String> {
        if self.state == DetectorState::Terminated {
            return Ok(());
        }

        self.state = DetectorState::ShuttingDown;

        if let Some(mut process) = self.process.take() {
            let pid = process.id();
            debug!("Shutting down Python detector (PID: {})", pid);

            // Step 1: Close stdin to signal shutdown
            drop(process.stdin.take());

            // Step 2: Give process time to finish current work (up to 5 seconds)
            let wait_result = Self::wait_with_timeout(&mut process, Duration::from_secs(5));

            match wait_result {
                Ok(status) => {
                    debug!("Python process exited cleanly: {:?}", status);
                    self.state = DetectorState::Terminated;
                    Ok(())
                }
                Err(_) => {
                    // Step 3: Force kill if it didn't exit gracefully
                    warn!("Python process (PID: {}) did not exit gracefully, force killing", pid);

                    if let Err(e) = process.kill() {
                        error!("Failed to kill process: {}", e);
                        self.state = DetectorState::Terminated;
                        return Err(format!("Failed to kill process: {}", e));
                    }

                    // Wait for kill to complete
                    match process.wait() {
                        Ok(status) => {
                            warn!("Python process forcefully terminated: {:?}", status);
                            self.state = DetectorState::Terminated;
                            Ok(())
                        }
                        Err(e) => {
                            error!("Failed to wait after kill: {}", e);
                            self.state = DetectorState::Terminated;
                            Err(format!("Failed to wait after kill: {}", e))
                        }
                    }
                }
            }
        } else {
            self.state = DetectorState::Terminated;
            Ok(())
        }
    }

    fn wait_with_timeout(process: &mut Child, timeout: Duration) -> Result<std::process::ExitStatus, ()> {
        use std::time::Instant;

        let start = Instant::now();
        loop {
            match process.try_wait() {
                Ok(Some(status)) => return Ok(status),
                Ok(None) => {
                    if start.elapsed() > timeout {
                        return Err(());
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }
                Err(_) => return Err(()),
            }
        }
    }

    pub fn is_alive(&mut self) -> bool {
        self.state == DetectorState::Ready &&
            self.process.as_mut()
                .and_then(|p| p.try_wait().ok())
                .map(|s| s.is_none())
                .unwrap_or(false)
    }

    fn send_image(&mut self, image_bytes: &[u8]) -> Result<(), String> {
        let process = self.process.as_mut().ok_or("Process not running")?;
        let stdin = process.stdin.as_mut().ok_or("Missing stdin")?;

        stdin.write_all(&(image_bytes.len() as u32).to_le_bytes())
            .map_err(|e| format!("Write length failed: {}", e))?;

        stdin.write_all(image_bytes)
            .map_err(|e| format!("Write image failed: {}", e))?;

        stdin.flush()
            .map_err(|e| format!("Flush failed: {}", e))?;

        Ok(())
    }

    fn receive_response(&mut self) -> Result<PythonDetectionResult, String> {
        let process = self.process.as_mut().ok_or("Process not running")?;
        let stdout = process.stdout.as_mut().ok_or("Missing stdout")?;

        let mut length_buf = [0u8; 4];
        stdout.read_exact(&mut length_buf)
            .map_err(|e| format!("Read length failed: {}", e))?;

        let length = u32::from_le_bytes(length_buf) as usize;
        let mut buf = vec![0u8; length];

        stdout.read_exact(&mut buf)
            .map_err(|e| format!("Read data failed: {}", e))?;

        let text = String::from_utf8(buf)
            .map_err(|e| format!("Invalid UTF-8: {}", e))?;

        serde_json::from_str::<PythonDetectionResult>(&text).or_else(|_| {
            serde_json::from_str::<PythonErrorResult>(&text)
                .map_err(|e| format!("Parse error: {}", e))
                .and_then(|err| Err(err.error))
        })
    }
}

impl Drop for PersistentPythonDetector {
    fn drop(&mut self) {
        if self.state != DetectorState::Terminated {
            error!("PersistentPythonDetector dropped without explicit shutdown!");
            let _ = self.shutdown();
        }
    }
}

pub fn perform_python_inference_with_counts(
    frame_message: &FrameMessage,
    detector: &mut PersistentPythonDetector,
    model_name: &str,
    experiment_mode: &str,
) -> Result<InferenceResult, String> {
    let image_bytes = &frame_message.frame.frame_data;

    let start = Some(current_timestamp_micros());
    let result = detector.detect_objects(image_bytes)?;
    let duration = Some(current_timestamp_micros() - start.unwrap()).unwrap();

    Ok(InferenceResult {
        sequence_id: frame_message.sequence_id,
        detections: result.detections.clone(),
        processing_time_us: duration,
        frame_size_bytes: image_bytes.len() as u32,
        detection_count: result.detections.len() as u32,
        image_width: result.image_width,
        image_height: result.image_height,
        model_name: model_name.to_string(),
        experiment_mode: experiment_mode.to_string(),
    })
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PythonDetectionResult {
    pub detections: Vec<Detection>,
    pub image_width: u32,
    pub image_height: u32,
    pub counts: ObjectCounts,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct PythonErrorResult {
    pub error: String,
    pub detections: Vec<Detection>,
    pub image_width: u32,
    pub image_height: u32,
    pub counts: ObjectCounts,
}