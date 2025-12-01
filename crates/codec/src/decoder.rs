use anyhow::{Context, Result};

/// # Errors
/// Returns an error if the image cannot be decoded.
pub fn decompress_to_rgb(src_bytes: &[u8]) -> Result<(u32, u32, Vec<u8>)> {
    let img = image::load_from_memory(src_bytes)
        .with_context(|| format!(
            "Failed to decode image ({} bytes)",
            src_bytes.len()
        ))?;

    let rgb = img.to_rgb8();
    Ok((rgb.width(), rgb.height(), rgb.into_raw()))
}