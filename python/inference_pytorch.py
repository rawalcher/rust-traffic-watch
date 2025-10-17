import io
import json
import struct
import sys
import warnings
import time
import threading
import os

import torch
from PIL import Image

warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*")

LEGACY_HUB_MODELS = {"yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x", "custom"}
IDLE_TIMEOUT_SECONDS = 120

class PersistentPyTorchInferenceServer:
    def __init__(self, model_name='yolov5s'):
        self.api = None
        self.model_name = model_name
        self.last_activity = time.time()
        self.start_time = time.time()
        self.inference_count = 0
        self.should_exit = False

        print(f"Loading model: {model_name}", file=sys.stderr, flush=True)

        use_ultralytics = (
                model_name.endswith('u') or
                model_name not in LEGACY_HUB_MODELS
        )

        if use_ultralytics:
            try:
                from ultralytics import YOLO
            except Exception as e:
                print(f"Ultralytics not available, falling back to torch.hub: {e}", file=sys.stderr, flush=True)
                use_ultralytics = False

        if use_ultralytics:
            weights = model_name if model_name.endswith('.pt') else f"{model_name}.pt"
            self.model = YOLO(weights)
            self.api = "ultralytics"
        else:
            # Legacy torch.hub path
            if model_name not in LEGACY_HUB_MODELS:
                print(f"'{model_name}' not available via torch.hub; using 'yolov5n' instead.", file=sys.stderr,
                      flush=True)
                model_name = "yolov5n"
            self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True, trust_repo=True)
            self.model.eval()
            self.api = "hub"

        torch.set_num_threads(2)

        # Warmup
        with torch.no_grad():
            if self.api == "hub":
                _ = self.model(torch.zeros(1, 3, 640, 640))
            else:
                _ = self.model.predict(Image.new("RGB", (640, 640)), verbose=False)

        # COCO traffic-ish classes by ID: person(0), bicycle(1), car(2), motorcycle(3), bus(5), train(6), truck(7)
        self.traffic_classes = {0, 1, 2, 3, 5, 6, 7}
        self.class_mapping = {
            'car': 'cars', 'truck': 'trucks', 'bus': 'buses',
            'motorcycle': 'motorcycles', 'bicycle': 'bicycles', 'person': 'pedestrians',
        }
        self.vehicle_classes = {'car', 'truck', 'bus', 'motorcycle', 'bicycle'}

        print("READY", flush=True)

        def activity_watchdog():
            while not self.should_exit:
                now = time.time()
                idle_time = now - self.last_activity

                if idle_time > IDLE_TIMEOUT_SECONDS:
                    print(f"Idle timeout ({IDLE_TIMEOUT_SECONDS}s), processed {self.inference_count} frames, exiting",
                          file=sys.stderr, flush=True)
                    os._exit(0)

                time.sleep(5)

        watchdog = threading.Thread(target=activity_watchdog, daemon=True)
        watchdog.start()

    def _infer_hub(self, image):
        with torch.no_grad():
            results = self.model(image)
        df = results.pandas().xyxy[0]
        return df

    def _infer_ultralytics(self, image):
        # returns a list of Results; take first
        res = self.model(image, verbose=False)
        r = res[0]
        # Build a DataFrame-like dict list compatible with downstream code
        names = r.names
        boxes = r.boxes

        rows = []
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().tolist()
            conf = boxes.conf.cpu().tolist()
            cls = boxes.cls.cpu().tolist()
            for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
                k = int(k)
                rows.append({
                    'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2,
                    'confidence': float(c),
                    'class': k,
                    'name': names.get(k, str(k)) if isinstance(names, dict) else (
                        names[k] if k < len(names) else str(k))
                })
        return rows

    def process_frame(self, image_bytes):
        self.last_activity = time.time()
        self.inference_count += 1

        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        width, height = image.size

        # Run inference
        if self.api == "hub":
            df = self._infer_hub(image)
            # Filter to relevant classes by numeric id
            traffic_df = df[df['class'].isin(self.traffic_classes)]
            rows = [{
                'class': row['name'],
                'bbox': [float(row['xmin']), float(row['ymin']),
                         float(row['xmax'] - row['xmin']), float(row['ymax'] - row['ymin'])],
                'confidence': float(row['confidence'])
            } for _, row in traffic_df.iterrows()]
        else:
            rows_all = self._infer_ultralytics(image)
            rows = []
            for row in rows_all:
                if int(row['class']) in self.traffic_classes:
                    rows.append({
                        'class': row['name'],
                        'bbox': [float(row['xmin']), float(row['ymin']),
                                 float(row['xmax'] - row['xmin']), float(row['ymax'] - row['ymin'])],
                        'confidence': float(row['confidence'])
                    })

        # Aggregate counts
        counts = {key: 0 for key in self.class_mapping.values()}
        counts.update({'total_vehicles': 0, 'total_objects': len(rows)})

        for det in rows:
            obj_class = det['class']
            if obj_class in self.class_mapping:
                counts[self.class_mapping[obj_class]] += 1
                if obj_class in self.vehicle_classes:
                    counts['total_vehicles'] += 1

        return {
            'detections': rows,
            'image_width': width,
            'image_height': height,
            'counts': counts
        }

    def send_response(self, data):
        json_data = json.dumps(data).encode('utf-8')
        sys.stdout.buffer.write(struct.pack('<I', len(json_data)))
        sys.stdout.buffer.write(json_data)
        sys.stdout.buffer.flush()

    def run_server(self):
        empty_counts = {key: 0 for key in self.class_mapping.values()}
        empty_counts.update({'total_vehicles': 0, 'total_objects': 0})

        while True:
            try:
                self.last_activity = time.time()

                length_bytes = sys.stdin.buffer.read(4)
                if len(length_bytes) != 4:
                    break
                frame_length = struct.unpack('<I', length_bytes)[0]
                frame_data = sys.stdin.buffer.read(frame_length)
                if len(frame_data) != frame_length:
                    break

                result = self.process_frame(frame_data)
                self.send_response(result)
            except Exception as e:
                error_msg = f"Error processing frame: {str(e)}"
                print(error_msg, file=sys.stderr, flush=True)
                self.send_response({
                    'error': str(e),
                    'detections': [],
                    'image_width': 0,
                    'image_height': 0,
                    'counts': empty_counts
                })


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else 'yolov5s'
    server = PersistentPyTorchInferenceServer(model_name)
    server.run_server()


if __name__ == "__main__":
    main()
