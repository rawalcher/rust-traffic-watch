use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, Command, Stdio};
use serde::{Deserialize, Serialize};
use crate::types::*;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PythonDetectionResult {
    pub detections: Vec<Detection>,
    pub confidence: f32,
    pub processing_time_us: u64,
    pub counts: ObjectCounts,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PythonErrorResult {
    pub error: String,
    pub detections: Vec<Detection>,
    pub confidence: f32,
    pub processing_time_us: u64,
    pub counts: ObjectCounts,
}

pub struct PersistentPythonDetector {
    process: Child,
    model_name: String,
}

impl PersistentPythonDetector {
    pub fn new(model_name: String) -> Result<Self, String> {
        let mut process = Command::new(".venv/bin/python")
            .arg("python/python_inference.py")
            .arg(&model_name)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to spawn Python process: {}", e))?;

        let stdout = process.stdout.as_mut().unwrap();
        let mut reader = BufReader::new(stdout);
        let mut ready_line = String::new();
        reader.read_line(&mut ready_line)
            .map_err(|e| format!("Failed to read ready signal: {}", e))?;

        if !ready_line.trim().eq("READY") {
            return Err(format!("Expected READY signal, got: {}", ready_line.trim()));
        }

        Ok(Self { process, model_name })
    }

    pub fn detect_objects(&mut self, image_bytes: &[u8]) -> Result<PythonDetectionResult, String> {
        let stdin = self.process.stdin.as_mut().unwrap();

        let length = image_bytes.len() as u32;
        stdin.write_all(&length.to_le_bytes())
            .map_err(|e| format!("Failed to write image length: {}", e))?;

        stdin.write_all(image_bytes)
            .map_err(|e| format!("Failed to write image data: {}", e))?;

        stdin.flush()
            .map_err(|e| format!("Failed to flush stdin: {}", e))?;

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

        match serde_json::from_str::<PythonDetectionResult>(&response_str) {
            Ok(result) => Ok(result),
            Err(_) => {
                match serde_json::from_str::<PythonErrorResult>(&response_str) {
                    Ok(error_result) => Err(error_result.error),
                    Err(parse_err) => Err(format!("Failed to parse response: {}, Data: {}", parse_err, response_str)),
                }
            }
        }
    }

    pub fn shutdown(&mut self) -> Result<(), String> {
        if let Some(stdin) = self.process.stdin.take() {
            drop(stdin);
        }

        self.process.wait()
            .map_err(|e| format!("Failed to wait for process: {}", e))?;

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
) -> Result<(InferenceResult, ObjectCounts), String> {
    let image_bytes = timing
        .frame_data
        .as_ref()
        .ok_or("No frame data in timing payload")?;

    let result = detector.detect_objects(image_bytes)?;

    let inference_result = InferenceResult {
        sequence_id: timing.sequence_id,
        detections: result.detections,
        confidence: result.confidence,
        processing_time_us: result.processing_time_us,
    };

    Ok((inference_result, result.counts))
}