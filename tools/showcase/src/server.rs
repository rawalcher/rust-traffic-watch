use axum::body::Body;
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use std::sync::Arc;

use crate::state::SharedState;

pub fn router(state: Arc<SharedState>) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/frame.jpg", get(frame))
        .route("/stats.json", get(stats))
        .with_state(state)
}

async fn index(State(state): State<Arc<SharedState>>) -> Html<String> {
    Html(format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Traffic Watch — {device}</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin: 0; background: #111; color: #eee; font-family: system-ui, sans-serif; display: flex; flex-direction: column; height: 100vh; }}
  header {{ padding: 1rem 1.5rem; background: #1c1c1c; border-bottom: 2px solid #5A7D9A; display: flex; justify-content: space-between; align-items: center; }}
  h1 {{ margin: 0; font-size: 1.4rem; color: #5A7D9A; }}
  main {{ flex: 1; display: grid; grid-template-columns: 1fr 320px; gap: 1rem; padding: 1rem; min-height: 0; }}
  .frame-wrap {{ display: flex; align-items: center; justify-content: center; background: #000; border-radius: 6px; overflow: hidden; }}
  .frame-wrap img {{ max-width: 100%; max-height: 100%; object-fit: contain; }}
  .stats {{ background: #1c1c1c; border-radius: 6px; padding: 1rem; display: flex; flex-direction: column; gap: 0.75rem; }}
  .stat {{ display: flex; justify-content: space-between; padding: 0.4rem 0; border-bottom: 1px solid #2a2a2a; }}
  .stat-label {{ color: #888; }}
  .stat-value {{ font-family: ui-monospace, monospace; color: #eee; }}
  .classes h3 {{ margin: 0.5rem 0; color: #5A7D9A; font-size: 0.95rem; }}
  .class-row {{ display: flex; justify-content: space-between; font-size: 0.9rem; padding: 0.2rem 0; }}
</style>
</head>
<body>
<header>
  <h1>Traffic Watch</h1>
  <div>{device}</div>
</header>
<main>
  <div class="frame-wrap"><img id="frame" alt="live frame" /></div>
  <div class="stats">
    <div class="stat"><span class="stat-label">Model</span><span class="stat-value" id="model">–</span></div>
    <div class="stat"><span class="stat-label">Inference</span><span class="stat-value" id="infer">–</span></div>
    <div class="stat"><span class="stat-label">Effective FPS</span><span class="stat-value" id="fps">–</span></div>
    <div class="stat"><span class="stat-label">Total frames</span><span class="stat-value" id="total">–</span></div>
    <div class="stat"><span class="stat-label">Detections</span><span class="stat-value" id="count">–</span></div>
    <div class="classes"><h3>Classes</h3><div id="classes"></div></div>
  </div>
</main>
<script>
  async function tick() {{
    try {{
      const r = await fetch('/stats.json', {{ cache: 'no-store' }});
      const s = await r.json();
      document.getElementById('model').textContent = s.model_name || '–';
      document.getElementById('infer').textContent = (s.inference_us / 1000).toFixed(1) + ' ms';
      document.getElementById('fps').textContent = s.effective_fps.toFixed(2);
      document.getElementById('total').textContent = s.total_inferences;
      document.getElementById('count').textContent = s.detection_count;
      const classesDiv = document.getElementById('classes');
      classesDiv.innerHTML = '';
      for (const [name, n] of Object.entries(s.class_counts).sort()) {{
        const row = document.createElement('div');
        row.className = 'class-row';
        row.innerHTML = `<span>${{name}}</span><span>${{n}}</span>`;
        classesDiv.appendChild(row);
      }}
      document.getElementById('frame').src = '/frame.jpg?t=' + s.frame_number + '_' + s.total_inferences;
    }} catch (e) {{ console.error(e); }}
  }}
  setInterval(tick, 200);
  tick();
</script>
</body>
</html>"#,
        device = state.device_label
    ))
}

async fn frame(State(state): State<Arc<SharedState>>) -> Response {
    let bytes = state.latest.read().jpeg_bytes.clone();
    if bytes.is_empty() {
        return (StatusCode::SERVICE_UNAVAILABLE, "warming up").into_response();
    }
    Response::builder()
        .header(header::CONTENT_TYPE, "image/jpeg")
        .header(header::CACHE_CONTROL, "no-store")
        .body(Body::from(bytes))
        .unwrap()
}

async fn stats(State(state): State<Arc<SharedState>>) -> axum::Json<serde_json::Value> {
    let latest = state.latest.read().clone();
    let total = *state.total_inferences.read();
    let elapsed_s = state.started.elapsed().as_secs_f64().max(0.001);
    #[allow(clippy::cast_precision_loss)]
    let fps = total as f64 / elapsed_s;

    axum::Json(serde_json::json!({
        "model_name": latest.model_name,
        "frame_number": latest.frame_number,
        "inference_us": latest.inference_us,
        "detection_count": latest.detection_count,
        "class_counts": latest.class_counts,
        "total_inferences": total,
        "effective_fps": fps,
        "device": state.device_label,
    }))
}
