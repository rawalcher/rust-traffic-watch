mod annotate;
mod server;
mod state;
mod worker;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use protocol::types::{EncodingSpec, ImageCodecKind, ImageResolutionType, Tier};
use std::path::PathBuf;
use tracing::info;

use crate::state::SharedState;
use crate::worker::{run_inference_loop, LoopConfig};

#[derive(Copy, Clone, Debug, ValueEnum)]
enum CodecArg {
    Jpg,
    Png,
    Webp,
}

impl From<CodecArg> for ImageCodecKind {
    fn from(v: CodecArg) -> Self {
        match v {
            CodecArg::Jpg => Self::JpgLossy,
            CodecArg::Png => Self::PngLossless,
            CodecArg::Webp => Self::WebpLossy,
        }
    }
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum ResArg {
    Fhd,
    Hd,
    Letterbox,
}

impl From<ResArg> for ImageResolutionType {
    fn from(v: ResArg) -> Self {
        match v {
            ResArg::Fhd => Self::FHD,
            ResArg::Hd => Self::HD,
            ResArg::Letterbox => Self::Letterbox,
        }
    }
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum TierArg {
    T1,
    T2,
    T3,
}

impl From<TierArg> for Tier {
    fn from(v: TierArg) -> Self {
        match v {
            TierArg::T1 => Self::T1,
            TierArg::T2 => Self::T2,
            TierArg::T3 => Self::T3,
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about = "Traffic Watch Showcase (Lange Nacht demo)")]
struct Args {
    #[arg(long, default_value = "yolov5n")]
    model: String,

    #[arg(long, value_enum, default_value_t = ResArg::Hd)]
    resolution: ResArg,

    #[arg(long, value_enum, default_value_t = CodecArg::Jpg)]
    codec: CodecArg,

    #[arg(long, value_enum, default_value_t = TierArg::T2)]
    tier: TierArg,

    #[arg(long)]
    fps: Option<u64>,

    #[arg(long, default_value_t = 6767)]
    port: u16,

    #[arg(long, default_value = "showcase")]
    device: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "showcase=info,inference=info,ort=warn".into()),
        )
        .without_time()
        .init();

    let args = Args::parse();

    let spec = EncodingSpec {
        codec: args.codec.into(),
        tier: args.tier.into(),
        resolution: args.resolution.into(),
    };

    let model_path = PathBuf::from("../../models").join(format!("{}.onnx", args.model));
    if !model_path.exists() {
        anyhow::bail!("Model file not found: {:?}", model_path.display());
    }

    let loop_cfg =
        LoopConfig { model_path, model_name: args.model.clone(), spec, fps_cap: args.fps };

    let state = SharedState::new(args.device.clone());

    let worker_state = state.clone();
    ort::init().with_name("showcase").commit()?;
    std::thread::Builder::new().name("inference-loop".into()).spawn(move || {
        if let Err(e) = run_inference_loop(&loop_cfg, &worker_state) {
            tracing::error!("Inference loop died: {e:?}");
        }
    })?;

    let app = server::router(state);
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", args.port)).await?;
    info!("Showcase running at http://0.0.0.0:{}/", args.port);

    axum::serve(listener, app).await?;
    Ok(())
}
