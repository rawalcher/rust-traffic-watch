# inference_tensorrt.py
import json
import struct
import sys
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class PersistentTRTInferenceServer:
    def __init__(self, engine_path='yolov5s.engine'):
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.allocate_buffers()
        self.model_input_size = (640, 640)  # default for yolov5s

        self.names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
                      'bus', 'train', 'truck', 'boat', 'traffic light']  # add more if needed
        self.traffic_classes = {0, 1, 2, 3, 5, 6, 7}
        self.class_mapping = {
            'car': 'cars', 'truck': 'trucks', 'bus': 'buses',
            'motorcycle': 'motorcycles', 'bicycle': 'bicycles', 'person': 'pedestrians'
        }
        self.vehicle_classes = {'car', 'truck', 'bus', 'motorcycle', 'bicycle'}

        print("READY", flush=True)

    def load_engine(self, path):
        with open(path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def allocate_buffers(self):
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append((host_mem, device_mem))
            else:
                self.outputs.append((host_mem, device_mem))

    def preprocess(self, image_bytes):
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        height, width = img.shape[:2]
        img_resized = cv2.resize(img, self.model_input_size)
        img_input = img_resized.astype(np.float32) / 255.0
        img_input = img_input.transpose((2, 0, 1))  # HWC → CHW
        img_input = np.expand_dims(img_input, axis=0)
        return img_input, width, height

    def postprocess(self, output, input_shape, orig_w, orig_h):
        predictions = output.reshape(-1, 7)
        detections = []
        counts = {key: 0 for key in self.class_mapping.values()}
        counts.update({'total_vehicles': 0, 'total_objects': 0})

        for det in predictions:
            if det[6] < 0.25:
                continue

            class_id = int(det[5])
            if class_id not in self.traffic_classes or class_id >= len(self.names):
                continue

            class_name = self.names[class_id]

            x1 = float(det[0]) * orig_w / input_shape[1]
            y1 = float(det[1]) * orig_h / input_shape[0]
            x2 = float(det[2]) * orig_w / input_shape[1]
            y2 = float(det[3]) * orig_h / input_shape[0]
            confidence = float(det[6])

            detections.append({
                'class': class_name,
                'bbox': [x1, y1, x2 - x1, y2 - y1],
                'confidence': confidence
            })

            if class_name in self.class_mapping:
                counts[self.class_mapping[class_name]] += 1
                if class_name in self.vehicle_classes:
                    counts['total_vehicles'] += 1

        counts['total_objects'] = len(detections)
        return detections, counts

    def infer(self, image_bytes):
        input_tensor, width, height = self.preprocess(image_bytes)
        np.copyto(self.inputs[0][0], input_tensor.ravel())

        cuda.memcpy_htod_async(self.inputs[0][1], self.inputs[0][0], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0][0], self.outputs[0][1], self.stream)
        self.stream.synchronize()

        detections, counts = self.postprocess(self.outputs[0][0], self.model_input_size, width, height)
        return {
            'detections': detections,
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
                length_bytes = sys.stdin.buffer.read(4)
                if len(length_bytes) != 4:
                    break
                frame_length = struct.unpack('<I', length_bytes)[0]
                frame_data = sys.stdin.buffer.read(frame_length)
                if len(frame_data) != frame_length:
                    break

                result = self.infer(frame_data)
                self.send_response(result)
            except Exception as e:
                self.send_response({
                    'error': str(e),
                    'detections': [],
                    'image_width': 0,
                    'image_height': 0,
                    'counts': empty_counts
                })

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else 'yolov5s'
    engine_path = f"{model_name}_trt.engine"

    server = PersistentTRTInferenceServer(engine_path)
    server.run_server()

if __name__ == "__main__":
    main()