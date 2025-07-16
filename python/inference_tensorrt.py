# inference_tensorrt.py
import json
import struct
import sys
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
from PIL import Image
import io

# Initialize CUDA driver
cuda.init()

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class PersistentTRTInferenceServer:
    def __init__(self, engine_path='yolov5s.engine'):
        try:
            self.cuda_device = cuda.Device(0)
            self.cuda_context = self.cuda_device.make_context()

            print(f"CUDA context initialized successfully", file=sys.stderr, flush=True)

            self.engine_path = engine_path
            self.engine = self.load_engine(engine_path)

            if self.engine is None:
                raise RuntimeError("Failed to load TensorRT engine")

            self.context = self.engine.create_execution_context()

            if self.context is None:
                raise RuntimeError("Failed to create execution context")

            self.allocate_buffers()
            self.model_input_size = (640, 640)

            self.names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
                          'bus', 'train', 'truck', 'boat', 'traffic light']

            self.traffic_classes = {0, 1, 2, 3, 5, 6, 7}  # Same as PyTorch version
            self.class_mapping = {
                'car': 'cars', 'truck': 'trucks', 'bus': 'buses',
                'motorcycle': 'motorcycles', 'bicycle': 'bicycles', 'person': 'pedestrians'
            }
            self.vehicle_classes = {'car', 'truck', 'bus', 'motorcycle', 'bicycle'}

            print("READY", flush=True)

        except Exception as e:
            print(f"Error initializing TensorRT server: {e}", file=sys.stderr, flush=True)
            self.cleanup()
            raise

    def cleanup(self):
        try:
            if hasattr(self, 'cuda_context'):
                self.cuda_context.pop()
                print("CUDA context cleaned up", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"Error during cleanup: {e}", file=sys.stderr, flush=True)

    def __del__(self):
        self.cleanup()

    def load_engine(self, path):
        print(f"Loading TensorRT engine from: {path}", file=sys.stderr, flush=True)
        try:
            with open(path, 'rb') as f:
                engine_data = f.read()

            with trt.Runtime(TRT_LOGGER) as runtime:
                engine = runtime.deserialize_cuda_engine(engine_data)

            if engine is None:
                raise RuntimeError(f"Failed to deserialize engine from {path}")

            print(f"Successfully loaded engine with {engine.num_bindings} bindings", file=sys.stderr, flush=True)
            return engine

        except FileNotFoundError:
            print(f"Engine file not found: {path}", file=sys.stderr, flush=True)
            raise
        except Exception as e:
            print(f"Error loading engine: {e}", file=sys.stderr, flush=True)
            raise

    def allocate_buffers(self):
        try:
            self.inputs = []
            self.outputs = []
            self.bindings = []
            self.stream = cuda.Stream()

            print(f"Engine has {self.engine.num_bindings} bindings", file=sys.stderr, flush=True)

            for i in range(self.engine.num_bindings):
                binding_name = self.engine.get_binding_name(i)
                shape = self.engine.get_binding_shape(i)
                dtype = trt.nptype(self.engine.get_binding_dtype(i))

                print(f"Binding {i}: {binding_name}, shape: {shape}, dtype: {dtype}", file=sys.stderr, flush=True)

                size = trt.volume(shape) * self.engine.max_batch_size

                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)

                self.bindings.append(int(device_mem))

                if self.engine.binding_is_input(i):
                    self.inputs.append((host_mem, device_mem))
                    print(f"Input binding: {binding_name}", file=sys.stderr, flush=True)
                else:
                    self.outputs.append((host_mem, device_mem))
                    print(f"Output binding: {binding_name}", file=sys.stderr, flush=True)

            print(f"Allocated {len(self.inputs)} inputs and {len(self.outputs)} outputs", file=sys.stderr, flush=True)

        except Exception as e:
            print(f"Error allocating buffers: {e}", file=sys.stderr, flush=True)
            raise

    def preprocess(self, image_bytes):
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            img = np.array(image)
            height, width = img.shape[:2]

            print(f"Original image size: {width}x{height}", file=sys.stderr, flush=True)

            if width != 640 or height != 640:
                img_resized = cv2.resize(img, (640, 640))
            else:
                img_resized = img

            img_input = img_resized.astype(np.float32) / 255.0
            img_input = img_input.transpose((2, 0, 1))  # HWC → CHW
            img_input = np.expand_dims(img_input, axis=0)

            return img_input, width, height
        except Exception as e:
            print(f"Error in preprocessing: {e}", file=sys.stderr, flush=True)
            raise

    def postprocess(self, output, input_shape, orig_w, orig_h):
        try:
            counts = {key: 0 for key in self.class_mapping.values()}
            counts.update({'total_vehicles': 0, 'total_objects': 0})
            detections = []

            print(f"Raw output shape: {output.shape}", file=sys.stderr, flush=True)
            print(f"Output min/max: {output.min()}/{output.max()}", file=sys.stderr, flush=True)

            if len(output.shape) == 1:
                if output.shape[0] == 25200 * 85:
                    output = output.reshape(1, 25200, 85)
                elif output.shape[0] % 85 == 0:
                    num_detections = output.shape[0] // 85
                    output = output.reshape(1, num_detections, 85)
                else:
                    print(f"Cannot reshape output with shape {output.shape}", file=sys.stderr, flush=True)
                    return detections, counts

            print(f"Reshaped output shape: {output.shape}", file=sys.stderr, flush=True)

            valid_detections = 0
            for detection in output[0]:
                if len(detection) < 85:
                    continue

                x, y, w, h = detection[:4]
                confidence = detection[4]
                class_scores = detection[5:]

                if confidence < 0.25:
                    continue

                class_id = np.argmax(class_scores)
                class_confidence = class_scores[class_id]

                final_confidence = confidence * class_confidence

                if final_confidence < 0.25:
                    continue

                if class_id not in self.traffic_classes or class_id >= len(self.names):
                    continue

                class_name = self.names[class_id]
                valid_detections += 1

                x1 = (x - w/2) * orig_w / input_shape[1]
                y1 = (y - h/2) * orig_h / input_shape[0]
                x2 = (x + w/2) * orig_w / input_shape[1]
                y2 = (y + h/2) * orig_h / input_shape[0]

                detections.append({
                    'class': class_name,
                    'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    'confidence': float(final_confidence)
                })

                if class_name in self.class_mapping:
                    counts[self.class_mapping[class_name]] += 1
                    if class_name in self.vehicle_classes:
                        counts['total_vehicles'] += 1

            counts['total_objects'] = len(detections)
            print(f"Processed {valid_detections} valid detections, found {len(detections)} traffic objects", file=sys.stderr, flush=True)

            return detections, counts
        except Exception as e:
            print(f"Error in postprocessing: {e}", file=sys.stderr, flush=True)
            return [], counts

    def infer(self, image_bytes):
        try:
            self.cuda_context.push()

            input_tensor, width, height = self.preprocess(image_bytes)

            np.copyto(self.inputs[0][0], input_tensor.ravel())
            cuda.memcpy_htod_async(self.inputs[0][1], self.inputs[0][0], self.stream)

            success = self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            if not success:
                raise RuntimeError("TensorRT inference failed")

            cuda.memcpy_dtoh_async(self.outputs[0][0], self.outputs[0][1], self.stream)
            self.stream.synchronize()

            self.cuda_context.pop()

            detections, counts = self.postprocess(self.outputs[0][0], self.model_input_size, width, height)

            return {
                'detections': detections,
                'image_width': width,
                'image_height': height,
                'counts': counts
            }
        except Exception as e:
            print(f"Error in inference: {e}", file=sys.stderr, flush=True)
            try:
                self.cuda_context.pop()
            except:
                pass
            raise

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
    engine_path = f"{model_name}_trt.engine"

    server = None
    try:
        server = PersistentTRTInferenceServer(engine_path)
        server.run_server()
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
    finally:
        if server:
            server.cleanup()

if __name__ == "__main__":
    main()