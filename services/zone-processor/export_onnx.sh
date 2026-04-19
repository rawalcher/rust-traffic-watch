#!/bin/bash
set -euo pipefail

OUTDIR="$HOME/yolo_onnx"
mkdir -p "$OUTDIR"
cd "$OUTDIR"

pip3 install --user --no-cache-dir "numpy==1.23.5" "onnx==1.14.1" "protobuf<4" "seaborn"

[ -d yolov5-v6.2 ] || git clone -b v6.2 https://github.com/ultralytics/yolov5.git yolov5-v6.2
cd yolov5-v6.2

for w in yolov5n.pt yolov5s.pt yolov5m.pt; do
  [ -f "$w" ] || wget -q "https://github.com/ultralytics/yolov5/releases/download/v6.2/$w"
  python3 export.py --weights "$w" --include onnx --imgsz 640 640 --opset 12 --device cpu
  cp "${w%.pt}.onnx" "$OUTDIR/"
done

echo "Done → $OUTDIR"
ls -lh "$OUTDIR"/*.onnx