use anyhow::Result;

pub fn decompress_to_rgb(src_bytes: &[u8]) -> Result<(u32, u32, Vec<u8>)> {
    let img = image::load_from_memory(src_bytes)?;
    let rgb = img.to_rgb8();
    Ok((rgb.width(), rgb.height(), rgb.into_raw()))
}
