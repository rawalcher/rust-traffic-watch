use chrono::Utc;
use csv::Writer;
use std::error::Error;
use tracing::{debug, info, warn};

use protocol::types::ExperimentConfig;
use protocol::InferenceMessage;

fn opt_str<T: ToString>(opt: Option<T>) -> String {
    opt.map(|v| v.to_string()).unwrap_or_default()
}

pub fn generate_analysis_csv(
    results: &[InferenceMessage],
    experiment_id: &str,
    config: &ExperimentConfig,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    if results.is_empty() {
        warn!("No results to save for experiment: {}", experiment_id);
        return Ok(());
    }

    std::fs::create_dir_all("logs")?;
    let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
    let filename = format!("logs/experiment_{experiment_id}_{timestamp}.csv");

    debug!("Saving {} results to {}", results.len(), filename);
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
        "zp_overhead_us",
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

    for r in results {
        let t = &r.timing;
        let i = &r.inference;

        let mode = if t.mode.is_some() { t.mode_str() } else { config.mode.to_string() };

        writer.write_record(&[
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
            opt_str(t.zp_overhead()),
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
    }

    writer.flush()?;
    info!("Analysis saved to {} with {} records", filename, results.len());
    Ok(())
}
