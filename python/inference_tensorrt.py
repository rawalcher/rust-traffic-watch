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
                      'bus', 'train', 'truck', 'boat', 'traffic light']
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

        if width != 640 or height != 640:
            img_resized = cv2.resize(img, (640, 640))
        else:
            img_resized = img

        img_input = img_resized.astype(np.float32) / 255.0
        img_input = img_input.transpose((2, 0, 1))  # HWC → CHW
        img_input = np.expand_dims(img_input, axis=0)
        return img_input, width, height

    def postprocess(self, output, input_shape, orig_w, orig_h):
        detections = []
        counts = {key: 0 for key in self.class_mapping.values()}
        counts.update({'total_vehicles': 0, 'total_objects': 0})

        output = output.reshape(1, 25200, 85)[0]

        obj_conf = output[:, 4]
        mask = obj_conf > 0.25
        output = output[mask]

        if len(output) == 0:
            return detections, counts

        class_probs = output[:, 5:]
        class_ids = np.argmax(class_probs, axis=1)
        class_confs = class_probs[np.arange(len(class_probs)), class_ids]

        scores = obj_conf[mask] * class_confs

        traffic_mask = np.isin(class_ids, list(self.traffic_classes))
        if not traffic_mask.any():
            return detections, counts

        boxes = output[traffic_mask, :4]
        scores = scores[traffic_mask]
        class_ids = class_ids[traffic_mask]

        x_c = boxes[:, 0]
        y_c = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]

        x1 = (x_c - w/2) * orig_w / input_shape[1]
        y1 = (y_c - h/2) * orig_h / input_shape[0]
        x2 = (x_c + w/2) * orig_w / input_shape[1]
        y2 = (y_c + h/2) * orig_h / input_shape[0]

        boxes_for_nms = np.stack([x1, y1, x2, y2], axis=1)
        keep = self.apply_nms(boxes_for_nms, scores, 0.45)

        for i in keep:
            class_id = class_ids[i]
            if class_id < len(self.names):
                class_name = self.names[class_id]

                detections.append({
                    'class': class_name,
                    'bbox': [float(x1[i]), float(y1[i]),
                             float(x2[i] - x1[i]), float(y2[i] - y1[i])],
                    'confidence': float(scores[i])
                })

                if class_name in self.class_mapping:
                    counts[self.class_mapping[class_name]] += 1
                    if class_name in self.vehicle_classes:
                        counts['total_vehicles'] += 1

        counts['total_objects'] = len(detections)
        return detections, counts

    def apply_nms(self, boxes, scores, iou_threshold):
        """Apply Non-Maximum Suppression"""
        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def infer(self, image_bytes):
        input_tensor, width, height = self.preprocess(image_bytes)
        np.copyto(self.inputs[0][0], input_tensor.ravel())

        cuda.memcpy_htod_async(self.inputs[0][1], self.inputs[0][0], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        for output in self.outputs:
            cuda.memcpy_dtoh_async(output[0], output[1], self.stream)
        self.stream.synchronize()

        output = self.outputs[-1][0]

        detections, counts = self.postprocess(output, self.model_input_size, width, height)
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