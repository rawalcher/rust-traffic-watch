import io
import json
import struct
import sys
import time
import torch
from PIL import Image


class PersistentPiInferenceServer:
    def __init__(self, model_name='yolov5s'):
        self.setup_device()

        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True, trust_repo=True)
        self.model.eval()

        if self.device.type == 'cuda':
            self.model = self.model.to(self.device)
            torch.backends.cudnn.benchmark = True
        else:
            torch.set_num_threads(1)

        self.traffic_classes = {0, 1, 2, 3, 5, 6, 7}
        self.class_mapping = {
            'car': 'cars', 'truck': 'trucks', 'bus': 'buses',
            'motorcycle': 'motorcycles', 'bicycle': 'bicycles', 'person': 'pedestrians'
        }
        self.vehicle_classes = {'car', 'truck', 'bus', 'motorcycle', 'bicycle'}

        print("READY", flush=True)

    def setup_device(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def process_frame(self, image_bytes):
        start_time = time.time()

        image = Image.open(io.BytesIO(image_bytes))

        if self.device.type == 'cuda':
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    results = self.model(image)
        else:
            with torch.no_grad():
                results = self.model(image)

        df = results.pandas().xyxy[0]
        traffic_df = df[df['class'].isin(self.traffic_classes)]

        detections = [
            {
                'class': row['name'],
                'bbox': [float(row['xmin']), float(row['ymin']),
                         float(row['xmax'] - row['xmin']), float(row['ymax'] - row['ymin'])],
                'confidence': float(row['confidence'])
            }
            for _, row in traffic_df.iterrows()
        ]

        counts = {key: 0 for key in self.class_mapping.values()}
        counts.update({'total_vehicles': 0, 'total_objects': len(detections)})

        for det in detections:
            obj_class = det['class']
            if obj_class in self.class_mapping:
                counts[self.class_mapping[obj_class]] += 1
                if obj_class in self.vehicle_classes:
                    counts['total_vehicles'] += 1

        processing_time_us = int((time.time() - start_time) * 1_000_000)
        max_confidence = max((d['confidence'] for d in detections), default=0.0)

        return {
            'detections': detections,
            'confidence': max_confidence,
            'processing_time_us': processing_time_us,
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
                error_result = {
                    'error': str(e),
                    'detections': [],
                    'confidence': 0.0,
                    'processing_time_us': 0,
                    'counts': empty_counts
                }
                self.send_response(error_result)


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else 'yolov5s'
    server = PersistentPiInferenceServer(model_name)
    server.run_server()


if __name__ == "__main__":
    main()