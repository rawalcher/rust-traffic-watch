pub const CONTROLLER_PORT: u16 = 9090;
pub const JETSON_PORT: u16 = 9091;

pub const CONTROLLER_ADDRESS: &str = "10.0.0.20";
pub const JETSON_ADDRESS: &str = "10.0.0.21";
pub const PI_ADDRESS: &str = "10.0.0.22";

pub const DEFAULT_MODEL: &str = "yolov5n";
pub const DEFAULT_DURATION_SECONDS: u64 = 30;
pub const DEFAULT_FPS: f32 = 1.0;

pub const MAX_FRAME_SEQUENCE: u64 = 30000;
pub const FRAME_WIDTH: u32 = 1920;
pub const FRAME_HEIGHT: u32 = 1080;

pub const PYTHON_VENV_PATH: &str = "python3";
pub const INFERENCE_PYTORCH_PATH: &str = "python/inference_pytorch.py";
pub const INFERENCE_TENSORRT_PATH: &str = "python/inference_tensorrt.py";

// pi and jetson connect here
pub fn controller_address() -> String {
    format!("{}:{}", CONTROLLER_ADDRESS, CONTROLLER_PORT)
}
pub fn controller_bind_address() -> String {
    format!("0.0.0.0:{}", CONTROLLER_PORT)
}

// pi connects here
pub fn jetson_data_address() -> String {
    format!("{}:{}", JETSON_ADDRESS, JETSON_PORT)
}
pub fn jetson_bind_address() -> String {
    format!("0.0.0.0:{}", JETSON_PORT)
}



