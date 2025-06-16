import io
import json
import struct
import sys
import time
import torch
from PIL import Image


class PersistentPiInferenceServer:
    def __init__(self, model_name='yolov5s'):
        # Load model once at startup
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True, trust_repo=True)
        self.model.eval()
        torch.set_num_threads(1)

        # Traffic classes from COCO
        self.traffic_classes = {0, 1, 2, 3, 5, 6, 7}  # person, bicycle, car, motorcycle, bus, train, truck

        # Signal ready
        print("READY", flush=True)

    def process_frame(self, image_bytes):
        """Process single frame and return results"""
        start_time = time.time()

        # Load image from bytes
        image = Image.open(io.BytesIO(image_bytes))

        # Inference
        with torch.no_grad():
            results = self.model(image)

        # Extract detections
        detections = []
        df = results.pandas().xyxy[0]

        for _, row in df.iterrows():
            if int(row['class']) in self.traffic_classes:
                detections.append({
                    'class': row['name'],
                    'bbox': [float(row['xmin']), float(row['ymin']),
                            float(row['xmax'] - row['xmin']), float(row['ymax'] - row['ymin'])],
                    'confidence': float(row['confidence'])
                })

        # Count objects
        counts = {'cars': 0, 'trucks': 0, 'buses': 0, 'motorcycles': 0,
                 'bicycles': 0, 'pedestrians': 0, 'total_vehicles': 0, 'total_objects': len(detections)}

        for det in detections:
            obj_class = det['class']
            if obj_class == 'car':
                counts['cars'] += 1
                counts['total_vehicles'] += 1
            elif obj_class == 'truck':
                counts['trucks'] += 1
                counts['total_vehicles'] += 1
            elif obj_class == 'bus':
                counts['buses'] += 1
                counts['total_vehicles'] += 1
            elif obj_class == 'motorcycle':
                counts['motorcycles'] += 1
                counts['total_vehicles'] += 1
            elif obj_class == 'bicycle':
                counts['bicycles'] += 1
                counts['total_vehicles'] += 1
            elif obj_class == 'person':
                counts['pedestrians'] += 1

        processing_time_us = int((time.time() - start_time) * 1_000_000)

        return {
            'detections': detections,
            'confidence': max([d['confidence'] for d in detections]) if detections else 0.0,
            'processing_time_us': processing_time_us,
            'counts': counts
        }

    def run_server(self):
        """Main server loop - reads frames and processes them"""
        while True:
            try:
                # Read frame length (4 bytes, little endian)
                length_bytes = sys.stdin.buffer.read(4)
                if len(length_bytes) != 4:
                    break

                frame_length = struct.unpack('<I', length_bytes)[0]

                # Read frame data
                frame_data = sys.stdin.buffer.read(frame_length)
                if len(frame_data) != frame_length:
                    break

                # Process frame
                result = self.process_frame(frame_data)

                # Send result
                result_json = json.dumps(result)
                result_bytes = result_json.encode('utf-8')

                # Send result length + result data
                sys.stdout.buffer.write(struct.pack('<I', len(result_bytes)))
                sys.stdout.buffer.write(result_bytes)
                sys.stdout.buffer.flush()

            except Exception as e:
                # Send error response
                error_result = {
                    'error': str(e),
                    'detections': [],
                    'confidence': 0.0,
                    'processing_time_us': 0,
                    'counts': {'cars': 0, 'trucks': 0, 'buses': 0, 'motorcycles': 0,
                              'bicycles': 0, 'pedestrians': 0, 'total_vehicles': 0, 'total_objects': 0}
                }
                error_json = json.dumps(error_result)
                error_bytes = error_json.encode('utf-8')

                sys.stdout.buffer.write(struct.pack('<I', len(error_bytes)))
                sys.stdout.buffer.write(error_bytes)
                sys.stdout.buffer.flush()

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else 'yolov5s'
    server = PersistentPiInferenceServer(model_name)
    server.run_server()

if __name__ == "__main__":
    main()