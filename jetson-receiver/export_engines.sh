#!/bin/bash
set -euo pipefail

OUTDIR="$HOME/yolo_engines"
mkdir -p "$OUTDIR/engines"
cd "$OUTDIR"

echo "[1/3] Pin numpy/onnx (Jetson/Py3.8 friendly)…"
python3 - <<'PY'
import sys, subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "--no-cache-dir",
                       "numpy==1.23.5", "onnx==1.14.1", "protobuf<4"])
PY

echo "[2/3] Clone YOLOv5 v6.2 and export ONNX (static 640)…"
if [ ! -d yolov5-v6.2 ]; then
  git clone -b v6.2 https://github.com/ultralytics/yolov5.git yolov5-v6.2
fi

cd yolov5-v6.2
for w in yolov5n.pt yolov5s.pt yolov5m.pt; do
  if [ ! -f "$w" ]; then
    wget -q "https://github.com/ultralytics/yolov5/releases/download/v6.2/$w"
  fi
  echo "Exporting $w -> ${w%.pt}.onnx"
  python3 export.py \
    --weights "$w" \
    --include onnx \
    --imgsz 640 640 \
    --opset 12 \
    --device cpu
done
cd ..

echo "[3/3] Build FP16 TensorRT engines…"
for onnx in yolov5-v6.2/*.onnx; do
  base="$(basename "${onnx%.onnx}")"
  echo " - $onnx -> engines/${base}_fp16.engine"
  /usr/src/tensorrt/bin/trtexec \
    --onnx="$onnx" \
    --saveEngine="engines/${base}_fp16.engine" \
    --workspace=2048 \
    --fp16
done

echo "Done. Engines in: $OUTDIR/engines"
ls -lh engines/*.engine
