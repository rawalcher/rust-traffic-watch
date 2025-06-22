use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, Command, Stdio};
use serde::{Deserialize, Serialize};
use crate::types::*;
use log::{debug, error};
use crate::constants::{PYTHON_VENV_PATH};

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
    process: Child,
}

impl PersistentPythonDetector {
    pub fn new(model_name: String, script_path: String) -> Result<Self, String> {
        debug!("Starting Python detector with model: {}", model_name);

        let mut process = Command::new(PYTHON_VENV_PATH)
            .arg(script_path)
            .arg(&model_name)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                error!("Failed to spawn Python process: {}", e);
                format!("Failed to spawn Python process: {}", e)
            })?;

        let stdout = process.stdout.as_mut().unwrap();
        let reader = BufReader::new(stdout);

        for line in reader.lines() {
            match line {
                Ok(line_text) => {
                    let line_trimmed = line_text.trim();
                    if line_trimmed.contains("READY") {
                        debug!("Python detector ready");
                        break;
                    } else {
                        debug!("Startup output: {}", line_trimmed);
                    }
                }
                Err(e) => {
                    let err = format!("Error reading from Python stdout: {}", e);
                    error!("{}", err);
                    return Err(err);
                }
            }
        }

        Ok(Self { process })
    }


    pub fn detect_objects(&mut self, image_bytes: &[u8]) -> Result<PythonDetectionResult, String> {
        self.send_image(image_bytes)?;
        self.receive_response()
    }

    fn send_image(&mut self, image_bytes: &[u8]) -> Result<(), String> {
        let stdin = self.process.stdin.as_mut().unwrap();

        let length = image_bytes.len() as u32;
        stdin.write_all(&length.to_le_bytes())
            .map_err(|e| format!("Failed to write image length: {}", e))?;

        stdin.write_all(image_bytes)
            .map_err(|e| format!("Failed to write image data: {}", e))?;

        stdin.flush()
            .map_err(|e| format!("Failed to flush stdin: {}", e))?;

        Ok(())
    }

    fn receive_response(&mut self) -> Result<PythonDetectionResult, String> {
        let stdout = self.process.stdout.as_mut().unwrap();

        let mut length_buf = [0u8; 4];
        stdout.read_exact(&mut length_buf)
            .map_err(|e| format!("Failed to read response length: {}", e))?;

        let response_length = u32::from_le_bytes(length_buf) as usize;

        let mut response_buf = vec![0u8; response_length];
        stdout.read_exact(&mut response_buf)
            .map_err(|e| format!("Failed to read response data: {}", e))?;

        let response_str = String::from_utf8(response_buf)
            .map_err(|e| format!("Invalid UTF-8 in response: {}", e))?;

        // Try to parse as success first, then as error
        serde_json::from_str::<PythonDetectionResult>(&response_str)
            .or_else(|_| {
                serde_json::from_str::<PythonErrorResult>(&response_str)
                    .map_err(|parse_err| format!("Failed to parse response: {}", parse_err))
                    .and_then(|error_result| Err(error_result.error))
            })
    }

    pub fn shutdown(&mut self) -> Result<(), String> {
        debug!("Shutting down Python detector");

        if let Some(stdin) = self.process.stdin.take() {
            drop(stdin);
        }

        self.process.wait()
            .map_err(|e| {
                error!("Failed to wait for process: {}", e);
                format!("Failed to wait for process: {}", e)
            })?;

        debug!("Python detector shut down");
        Ok(())
    }
}

impl Drop for PersistentPythonDetector {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

pub async fn perform_python_inference_with_counts(
    timing: &TimingPayload,
    detector: &mut PersistentPythonDetector,
    model_name: &str,
    experiment_mode: &str,
) -> Result<(InferenceResult, ObjectCounts), String> {
    let image_bytes = timing
        .frame_data
        .as_ref()
        .ok_or("No frame data in timing payload")?;

    let start_time = current_timestamp_micros();
    let result = detector.detect_objects(image_bytes)?;
    let end_time = current_timestamp_micros();

    let processing_time_us = end_time - start_time;

    let inference_result = InferenceResult {
        sequence_id: timing.sequence_id,
        detections: result.detections.clone(),
        processing_time_us,
        pi_hostname: timing.pi_hostname.clone(),
        frame_size_bytes: image_bytes.len() as u32,
        detection_count: result.detections.len() as u32,
        image_width: result.image_width,
        image_height: result.image_height,
        model_name: model_name.to_string(),
        experiment_mode: experiment_mode.to_string(),
    };

    Ok((inference_result, result.counts))
}