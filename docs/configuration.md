# Configuration Guide

This document describes how to configure **Rust Traffic Watch**.

---

## Network Setup

Edit `shared/src/constants.rs` to configure device IP addresses:

```rust
pub const CONTROLLER_ADDRESS: &str = "10.0.0.20";
pub const JETSON_ADDRESS: &str = "10.0.0.21";
pub const PI_ADDRESS: &str = "10.0.0.25";
````

---

## Experiment Parameters

Default parameters are also set in `shared/src/constants.rs`:

```rust
pub const DEFAULT_MODEL: &str = "yolov5n";
pub const DEFAULT_DURATION_SECONDS: u64 = 600;
pub const DEFAULT_SEND_FPS: f32 = 1.0;
```

You can modify these defaults or override them via command-line options (see [usage.md](usage.md)).