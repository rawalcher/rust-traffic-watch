# Usage Guide

---

## Starting the System

Run each component on the appropriate device:

**Controller** (main machine):
```bash
cargo run --release --bin controller
````

**Jetson Receiver** (on NVIDIA Jetson):

```bash
cargo run --release --bin jetson-receiver
```

**Pi Sender** (on Raspberry Pi):

```bash
cargo run --release --bin pi-sender
```

---

## Running Experiments

### Single Experiment

```bash
# Local inference on Raspberry Pi
cargo run --release --bin controller -- --model=yolov5s --fps=10 --local

# Remote inference on Jetson
cargo run --release --bin controller -- --model=yolov5m --fps=5 --remote
```

### Automated Test Suite

```bash
# Full test suite
cargo run --release --bin controller

# Quick test mode
cargo run --release --bin controller -- --quick

# Custom configuration
cargo run --release --bin controller -- \
  --models=yolov5n,yolov5s \
  --fps=1,5,10 \
  --duration=300
```

---

## Command-Line Options

| Option       | Description                   | Example           |
|--------------|-------------------------------|-------------------|
| `--model`    | Specify YOLO model            | `--model=yolov5m` |
| `--fps`      | Frame processing rate         | `--fps=15`        |
| `--duration` | Experiment duration (seconds) | `--duration=300`  |
| `--local`    | Local inference only          | `--local`         |
| `--remote`   | Remote inference only         | `--remote`        |
| `--quick`    | Quick test mode               | `--quick`         |