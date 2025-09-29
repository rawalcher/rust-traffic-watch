#!/bin/bash
set -euo pipefail

OUTDIR="$HOME/yolo_engines"
mkdir -p "$OUTDIR/engines"
cd "$OUTDIR"

# deps (Jetson/Py3.8-friendly)
pip3 install --user --no-cache-dir "numpy==1.23.5" "onnx==1.14.1" "protobuf<4"

# YOLOv5 v6.2 → ONNX (640)
[ -d yolov5-v6.2 ] || git clone -b v6.2 https://github.com/ultralytics/yolov5.git yolov5-v6.2
cd yolov5-v6.2
for w in yolov5n.pt yolov5s.pt yolov5m.pt; do
  [ -f "$w" ] || wget -q "https://github.com/ultralytics/yolov5/releases/download/v6.2/$w"
  python3 export.py --weights "$w" --include onnx --imgsz 640 640 --opset 12 --device cpu
done
cd ..

# ONNX → TensorRT FP16
for onnx in yolov5-v6.2/*.onnx; do
  base="$(basename "${onnx%.onnx}")"
  /usr/src/tensorrt/bin/trtexec --onnx="$onnx" --saveEngine="engines/${base}_fp16.engine" --workspace=2048 --fp16
done

echo "Done → $OUTDIR/engines"
ls -lh engines/*.engine