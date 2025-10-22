# Setup Guide

---

## Requirements

- **Rust** 1.75 or higher
- **Python** 3.8+ with virtual environment support
- **CUDA** and **TensorRT** (for Jetson deployment)
- Network connectivity between all devices
- Hardware:
    - Raspberry Pi 4 (4GB+ RAM recommended)
    - NVIDIA Jetson (Nano)
    - Controller machine (any system with Rust support)

---

## Repository Setup

```bash
git clone https://github.com/yourusername/rust-traffic-watch.git
cd rust-traffic-watch
cargo build --release
````

---

## Python Environment

It is recommended to use a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Raspberry Pi (PyTorch Inference)

```bash
pip install torch torchvision pillow numpy
```

### Jetson (TensorRT Inference)

```bash
pip install pycuda tensorrt opencv-python numpy
```

---

## YOLO Model Files

Download pretrained YOLOv5 models:

```bash
wget https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5n.pt
wget https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt
wget https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5m.pt
```

or let PyTorch handle it (will automatically Download it)

Place these in the project root.

---

## TensorRT Engine Generation (Jetson Only)

On Jetson, convert ONNX models to TensorRT engines:

TODO: Rewrite Script to Engine Files are properly Placed

```bash
chmod +x zone-processor/export_engines.sh
./zone-processor/export_engines.sh
```

---

## TODO Sample Data Section