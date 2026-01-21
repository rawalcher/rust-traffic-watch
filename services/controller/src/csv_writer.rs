use chrono::Utc;
use csv::Writer;
use std::error::Error;
use std::fs::File;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info};

use protocol::types::ExperimentConfig;
use protocol::InferenceMessage;

fn opt_str<T: ToString>(opt: Option<T>) -> String {
    opt.map(|v| v.to_string()).unwrap_or_default()
}

pub struct StreamingCsvWriter {
    writer: Writer<File>,
    count: usize,
}

impl StreamingCsvWriter {
    pub fn new(experiment_id: &str) -> Result<Self, Box<dyn Error + Send + Sync>> {
        std::fs::create_dir_all("logs")?;
        let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
        let filename = format!("logs/experiment_{experiment_id}_{timestamp}.csv");

        debug!("Creating CSV writer at {}", filename);
        let mut writer = Writer::from_path(&filename)?;

        writer.write_record([
            // Identifiers
            "sequence_id",
            "frame_number",
            "source_device",
            "mode",
            // Raw timestamps (microseconds since epoch)
            "controller_sent_pulse",
            "capture_start",
            "encode_complete",
            "send_start",
            "receive_start",
            "inference_start",
            "inference_complete",
            "send_result",
            "controller_received",
            // Computed latencies (microseconds)
            "total_latency_us",
            "rsu_overhead_us",
            "network_latency_us",
            "zp_processing_us",
            "zp_queueing_us",
            "inference_time_us",
            // Detection metadata
            "frame_size_bytes",
            "detection_count",
            "image_width",
            "image_height",
            "model_name",
            // Experiment configuration
            "codec",
            "tier",
            "resolution",
        ])?;

        writer.flush()?;
        info!("Created CSV file: {}", filename);

        Ok(Self { writer, count: 0 })
    }

    pub fn write_result(
        &mut self,
        result: &InferenceMessage,
        config: &ExperimentConfig,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let t = &result.timing;
        let i = &result.inference;

        let mode = if t.mode.is_some() { t.mode_str() } else { config.mode.to_string() };

        self.writer.write_record(&[
            // Identifiers
            t.sequence_id.to_string(),
            t.frame_number.to_string(),
            t.source_device.clone(),
            mode,
            // Raw timestamps
            opt_str(t.controller_sent_pulse),
            opt_str(t.capture_start),
            opt_str(t.encode_complete),
            opt_str(t.send_start),
            opt_str(t.receive_start),
            opt_str(t.inference_start),
            opt_str(t.inference_complete),
            opt_str(t.send_result),
            opt_str(t.controller_received),
            // Computed latencies
            opt_str(t.total_latency()),
            opt_str(t.rsu_overhead()),
            opt_str(t.network_latency()),
            opt_str(t.zp_processing()),
            opt_str(t.zp_queueing()),
            opt_str(t.inference_time()),
            // Detection metadata
            i.frame_size_bytes.to_string(),
            i.detection_count.to_string(),
            i.image_width.to_string(),
            i.image_height.to_string(),
            i.model_name.clone(),
            // Experiment configuration
            format!("{:?}", config.encoding_spec.codec),
            format!("{:?}", config.encoding_spec.tier),
            format!("{:?}", config.encoding_spec.resolution),
        ])?;

        self.count += 1;

        if self.count.is_multiple_of(10) {
            self.writer.flush()?;
        }

        Ok(())
    }

    pub fn finalize(mut self) -> Result<usize, Box<dyn Error + Send + Sync>> {
        self.writer.flush()?;
        info!("CSV file finalized with {} records", self.count);
        Ok(self.count)
    }

    pub const fn count(&self) -> usize {
        self.count
    }
}

pub struct ConcurrentCsvWriter {
    pub(crate) inner: Arc<Mutex<StreamingCsvWriter>>,
    pub(crate) config: ExperimentConfig,
}

impl ConcurrentCsvWriter {
    pub fn new(
        experiment_id: &str,
        config: ExperimentConfig,
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let writer = StreamingCsvWriter::new(experiment_id)?;
        Ok(Self { inner: Arc::new(Mutex::new(writer)), config })
    }

    pub async fn write_result(
        &self,
        result: &InferenceMessage,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        self.inner.lock().await.write_result(result, &self.config)
    }

    pub async fn count(&self) -> usize {
        self.inner.lock().await.count()
    }

    pub fn finalize(self) -> Result<usize, Box<dyn Error + Send + Sync>> {
        let writer = Arc::try_unwrap(self.inner)
            .map_err(|_| "Cannot finalize: writer still has outstanding references")?
            .into_inner();
        writer.finalize()
    }
}
