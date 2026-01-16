use image::ImageReader;
use protocol::types::tiers::{codec_name, res_folder};
use protocol::types::EncodingSpec;
use std::error::Error;
use std::io::Cursor;
use std::path::PathBuf;

pub fn handle_frame(
    frame_number: u64,
    spec: &EncodingSpec,
) -> Result<(Vec<u8>, u32, u32), Box<dyn Error + Send + Sync>> {
    let folder_res = res_folder(spec.resolution);
    let name = codec_name(spec.codec);
    let folder_codec = name;
    let ext = name;
    let tier = spec.tier;

    // Path structure: roadside-unit/testImages/{resolution}/{codec}/seq3-drone_{number}_{tier}.{ext}
    let mut path = PathBuf::from("services/roadside-unit/testImages");
    path.push(folder_res);
    path.push(folder_codec);
    path.push(format!("seq3-drone_{frame_number:07}_{tier}.{ext}"));

    if !path.exists() {
        return Err(format!("Frame file not found: {:?}", path.display()).into());
    }

    let frame_data = std::fs::read(&path)?;

    let cursor = Cursor::new(&frame_data);
    let reader = ImageReader::new(cursor).with_guessed_format()?;
    let (width, height) = reader.into_dimensions()?;

    Ok((frame_data, width, height))
}
