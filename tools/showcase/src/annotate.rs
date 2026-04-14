use anyhow::Result;
use image::{DynamicImage, Rgba};
use imageproc::drawing::{draw_filled_rect_mut, draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use protocol::types::Detection;
use rusttype::{Font, Scale};

static FONT_DATA: &[u8] = include_bytes!("../../model-tester/fonts/Roboto-Thin.ttf");

#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn draw_detections(img: &mut DynamicImage, detections: &[Detection]) -> Result<()> {
    let font =
        Font::try_from_bytes(FONT_DATA).ok_or_else(|| anyhow::anyhow!("Failed to load font"))?;
    let scale = Scale::uniform(24.0);

    for det in detections {
        let (x, y, w, h) = (
            det.bbox[0] as i32,
            det.bbox[1] as i32,
            det.bbox[2].max(1.0) as u32,
            det.bbox[3].max(1.0) as u32,
        );

        let seed = det.class.bytes().map(u32::from).sum::<u32>();
        let color = Rgba([((seed * 50) % 255) as u8, ((seed * 100) % 255) as u8, 200, 255]);

        draw_hollow_rect_mut(img, Rect::at(x, y).of_size(w, h), color);

        let label = format!("{} {:.0}%", det.class, det.confidence * 100.0);
        let (text_w, text_h) = imageproc::drawing::text_size(scale, &font, &label);

        let label_y = (y - text_h).max(0);
        let bg_rect =
            Rect::at(x, label_y).of_size((text_w + 8).max(1) as u32, text_h.max(1) as u32);

        draw_filled_rect_mut(img, bg_rect, color);
        draw_text_mut(img, Rgba([255, 255, 255, 255]), x + 4, label_y, scale, &font, &label);
    }
    Ok(())
}
