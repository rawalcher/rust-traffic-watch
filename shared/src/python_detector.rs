use crate::constants::PYTHON_VENV_PATH;
use crate::types::*;
use log::{debug, error, warn};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, Command, Stdio};

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

pub struct PersistentPythonDetector {
    process: Option<Child>,
}

impl PersistentPythonDetector {
    pub fn new(model_name: String, script_path: String) -> Result<Self, String> {
        debug!("Launching Python detector with model '{}'", model_name);

        let mut process = Command::new(PYTHON_VENV_PATH)
            .arg(script_path)
            .arg(&model_name)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("Spawn failed: {}", e))?;

        let stdout = process.stdout.as_mut().ok_or("Missing stdout")?;
        let reader = BufReader::new(stdout);

        for line in reader.lines().flatten() {
            let trimmed = line.trim();
            if trimmed.contains("READY") {
                debug!("Python process is ready");
                return Ok(Self {
                    process: Some(process),
                });
            } else {
                debug!("Python output: {}", trimmed);
            }
        }

        let _ = process.kill();
        let _ = process.wait();
        Err("Did not receive READY from Python".into())
    }

    pub fn detect_objects(&mut self, image_bytes: &[u8]) -> Result<PythonDetectionResult, String> {
        self.send_image(image_bytes)?;
        self.receive_response()
    }

    fn send_image(&mut self, image_bytes: &[u8]) -> Result<(), String> {
        let process = self.process.as_mut().ok_or("Process not running")?;
        let stdin = process.stdin.as_mut().ok_or("Missing stdin".to_string())?;

        stdin
            .write_all(&(image_bytes.len() as u32).to_le_bytes())
            .map_err(|e| format!("Failed to write length: {}", e))?;

        stdin
            .write_all(image_bytes)
            .map_err(|e| format!("Failed to write image: {}", e))?;

        stdin
            .flush()
            .map_err(|e| format!("Failed to flush: {}", e))?;

        Ok(())
    }

    fn receive_response(&mut self) -> Result<PythonDetectionResult, String> {
        let process = self.process.as_mut().ok_or("Process not running")?;
        let stdout = process.stdout.as_mut().ok_or("Missing stdout")?;

        let mut length_buf = [0u8; 4];
        stdout
            .read_exact(&mut length_buf)
            .map_err(|e| format!("Failed to read response length: {}", e))?;

        let length = u32::from_le_bytes(length_buf) as usize;

        let mut buf = vec![0u8; length];
        stdout
            .read_exact(&mut buf)
            .map_err(|e| format!("Failed to read response data: {}", e))?;

        let text = String::from_utf8(buf).map_err(|e| format!("Invalid UTF-8: {}", e))?;

        serde_json::from_str::<PythonDetectionResult>(&text).or_else(|_| {
            serde_json::from_str::<PythonErrorResult>(&text)
                .map_err(|e| format!("Parse error: {}", e))
                .and_then(|err| Err(err.error))
        })
    }

    pub fn shutdown(&mut self) -> Result<(), String> {
        if let Some(mut process) = self.process.take() {
            debug!("Shutting down Python detector (PID: {:?})", process.id());

            drop(process.stdin.take());
            drop(process.stdout.take());

            for i in 0..20 {
                match process.try_wait() {
                    Ok(Some(status)) => {
                        debug!("Python process exited gracefully with status: {:?}", status);
                        return Ok(());
                    }
                    Ok(None) => {
                        std::thread::sleep(std::time::Duration::from_millis(100));
                        if i % 5 == 0 {
                            debug!("Waiting for Python process to exit... ({}/2s)", i * 100);
                        }
                    }
                    Err(e) => {
                        warn!("Error checking process status: {}", e);
                        break;
                    }
                }
            }

            warn!("Python process did not exit gracefully, sending SIGTERM...");
            if let Err(e) = signal_process(process.id(), libc::SIGTERM) {
                warn!("Failed to send SIGTERM: {}", e);
            }

            for _ in 0..10 {
                match process.try_wait() {
                    Ok(Some(status)) => {
                        debug!("Python process terminated with status: {:?}", status);
                        return Ok(());
                    }
                    Ok(None) => {
                        std::thread::sleep(std::time::Duration::from_millis(100));
                    }
                    Err(_) => break,
                }
            }

            error!("Python process still running, sending SIGKILL...");
            if let Err(e) = process.kill() {
                error!("Failed to SIGKILL process: {}", e);
                return Err(format!("Failed to kill process: {}", e));
            }

            match process.wait() {
                Ok(status) => {
                    debug!("Python process forcefully killed with status: {:?}", status);
                    Ok(())
                }
                Err(e) => {
                    error!("Failed to wait for killed process: {}", e);
                    Err(format!("Failed to wait for process: {}", e))
                }
            }
        } else {
            debug!("No Python process to shutdown");
            Ok(())
        }
    }

    pub fn is_alive(&mut self) -> bool {
        if let Some(ref mut process) = self.process {
            match process.try_wait() {
                Ok(None) => true,
                Ok(Some(_)) => {
                    self.process = None;
                    false
                }
                Err(_) => false,
            }
        } else {
            false
        }
    }
}

fn signal_process(pid: u32, signal: i32) -> Result<(), String> {
    use std::io::Error;
    unsafe {
        if libc::kill(pid as i32, signal) == 0 {
            Ok(())
        } else {
            Err(format!("kill() failed: {}", Error::last_os_error()))
        }
    }
}

impl Drop for PersistentPythonDetector {
    fn drop(&mut self) {
        if self.process.is_some() {
            error!("!!! PersistentPythonDetector dropped without explicit shutdown !!!");
            if let Err(e) = self.shutdown() {
                error!("Error during PersistentPythonDetector drop: {}", e);
            }
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
