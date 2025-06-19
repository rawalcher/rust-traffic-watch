pub const CONTROLLER_PORT: u16 = 9090;
pub const PI_PORT: u16 = 8080;
pub const JETSON_PORT: u16 = 9092;

pub const PI_ADDRESS: &str = "192.168.68.57";
pub const JETSON_ADDRESS: &str = "192.168.68.59";
pub const CONTROLLER_ADDRESS: &str = "192.168.68.70";

pub const DEFAULT_MODEL: &str = "yolov5n";
pub const DEFAULT_DURATION_SECONDS: u64 = 900; // 15 minutes
pub const DEFAULT_FPS: f32 = 10.0;

pub const MAX_FRAME_SEQUENCE: u64 = 900;
pub const FRAME_WIDTH: u32 = 1920;
pub const FRAME_HEIGHT: u32 = 1080;

pub const PYTHON_VENV_PATH: &str = "python3";
pub const PYTHON_SCRIPT_PATH: &str = "python/python_inference.py";

pub fn pi_full_address() -> String {
    format!("{}:{}", PI_ADDRESS, PI_PORT)
}

pub fn jetson_full_address() -> String {
    format!("{}:{}", JETSON_ADDRESS, JETSON_PORT)
}

pub fn controller_bind_address() -> String {
    format!("{}:{}", CONTROLLER_ADDRESS, CONTROLLER_PORT)
}

pub fn jetson_bind_address() -> String {
    format!("{}:{}", JETSON_ADDRESS, JETSON_PORT)
}

pub fn pi_bind_address() -> String {
    format!("{}:{}", PI_ADDRESS, PI_PORT)
}