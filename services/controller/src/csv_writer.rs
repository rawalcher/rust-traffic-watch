use chrono::Utc;
use csv::Writer;
use std::error::Error;
use tracing::{debug, info, warn};

use protocol::{ExperimentConfig, ExperimentMode, InferenceMessage};

fn opt_str<T: ToString>(opt: Option<T>) -> String {
    opt.map(|v| v.to_string()).unwrap_or_default()
}

fn calc_diff(a: Option<u64>, b: Option<u64>) -> u64 {
    match (a, b) {
        (Some(a), Some(b)) if a >= b => a - b,
        _ => 0,
    }
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
        "sequence_id",
        "frame_number",
        "pi_hostname",
        "pi_capture_start",
        "pi_sent_to_jetson",
        "jetson_received",
        "jetson_sent_result",
        "controller_sent_pulse",
        "controller_received",
        "pi_overhead_us",
        "jetson_overhead_us",
        "network_latency_us",
        "total_latency_us",
        "inference_us",
        "frame_size_bytes",
        "detection_count",
        "image_width",
        "image_height",
        "model_name",
        "experiment_mode",
        "codec",
        "tier",
        "resolution",
    ])?;

    for r in results {
        let t = &r.timing;
        let i = &r.inference;

        let pi_overhead = calc_diff(t.pi_sent_to_jetson, t.pi_capture_start);
        let jetson_overhead = calc_diff(t.jetson_sent_result, t.jetson_received);
        let total_latency = calc_diff(t.controller_received, t.controller_sent_pulse);

        let network_latency = match config.mode {
            ExperimentMode::Offload => total_latency.saturating_sub(pi_overhead + jetson_overhead),
            ExperimentMode::Local => total_latency.saturating_sub(i.processing_time_us),
        };

        writer.write_record(&[
            t.sequence_id.to_string(),
            t.frame_number.to_string(),
            t.pi_hostname.clone(),
            opt_str(t.pi_capture_start),
            opt_str(t.pi_sent_to_jetson),
            opt_str(t.jetson_received),
            opt_str(t.jetson_sent_result),
            opt_str(t.controller_sent_pulse),
            opt_str(t.controller_received),
            pi_overhead.to_string(),
            jetson_overhead.to_string(),
            network_latency.to_string(),
            total_latency.to_string(),
            i.processing_time_us.to_string(),
            i.frame_size_bytes.to_string(),
            i.detection_count.to_string(),
            i.image_width.to_string(),
            i.image_height.to_string(),
            i.model_name.clone(),
            i.experiment_mode.clone(),
            format!("{:?}", config.encoding_spec.codec),
            format!("{:?}", config.encoding_spec.tier),
            format!("{:?}", config.encoding_spec.resolution),
        ])?;
    }

    writer.flush()?;
    info!("Analysis saved to {} with {} records", filename, results.len());
    Ok(())
}
