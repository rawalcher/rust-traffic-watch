import json
import os
import struct
import sys
import threading
import time

import cv2
import numpy as np
import pycuda.autoinit  # noqa
import pycuda.driver as cuda
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
PROCESS_TTL_SECONDS = 120

# Match PyTorch behavior
TRAFFIC_CLASS_IDS = {0, 1, 2, 3, 5, 6, 7}
ID_TO_NAME = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    6: 'train',
    7: 'truck',
}
CLASS_MAPPING = {
    'car': 'cars', 'truck': 'trucks', 'bus': 'buses',
    'motorcycle': 'motorcycles', 'bicycle': 'bicycles', 'person': 'pedestrians'
}
VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle', 'bicycle'}

CONF_THR = 0.25  # Ultralytics default
IOU_THR = 0.45  # Ultralytics default (class-aware NMS)
NET_SIZE = 640  # engines built with imgsz=640


def letterbox_bgr(img, new_size=NET_SIZE, color=(114, 114, 114)):
    """Resize with unchanged aspect ratio using padding (Ultralytics-style)."""
    h, w = img.shape[:2]
    r = min(new_size / w, new_size / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    padw, padh = (new_size - nw) // 2, (new_size - nh) // 2
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = cv2.copyMakeBorder(
        resized, padh, new_size - nh - padh, padw, new_size - nw - padw,
        cv2.BORDER_CONSTANT, value=color
    )
    return out, r, padw, padh, w, h


def nms_class_aware(boxes, scores, classes, iou_thr=IOU_THR):
    """Per-class greedy NMS to match Ultralytics defaults."""
    keep_global = []
    if len(boxes) == 0:
        return keep_global
    for cid in np.unique(classes):
        idxs = np.where(classes == cid)[0]
        if idxs.size == 0:
            continue
        b = boxes[idxs]
        s = scores[idxs]
        x1, y1, x2, y2 = [b[:, i] for i in range(4)]
        areas = (x2 - x1) * (y2 - y1)
        order = s.argsort()[::-1]
        keep_local = []
        while order.size > 0:
            i = order[0]
            keep_local.append(i)
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
            inds = np.where(ovr <= iou_thr)[0]
            order = order[inds + 1]
        # Map local indices back to global
        keep_global.extend(list(idxs[keep_local]))
    return keep_global


class PersistentTRTInferenceServer:
    def __init__(self, engine_path):
        def kill_after_ttl():
            time.sleep(PROCESS_TTL_SECONDS)
            print(f"TTL expired ({PROCESS_TTL_SECONDS}s), self-terminating", file=sys.stderr, flush=True)
            os._exit(0)

        ttl_thread = threading.Thread(target=kill_after_ttl, daemon=True)
        ttl_thread.start()

        print(f"Loading engine: {engine_path}", file=sys.stderr, flush=True)

        self.runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Buffers
        self.bindings = [None] * self.engine.num_bindings
        self.host = [None] * self.engine.num_bindings
        self.device = [None] * self.engine.num_bindings
        self.stream = cuda.Stream()

        # Preallocate for static shapes; for dynamic, we re-alloc at runtime
        for idx in range(self.engine.num_bindings):
            shape = self.engine.get_binding_shape(idx)
            dt = trt.nptype(self.engine.get_binding_dtype(idx))
            vol = int(np.prod(shape)) if not any(s < 0 for s in shape) else 1
            self.host[idx] = cuda.pagelocked_empty(vol, dt)
            self.device[idx] = cuda.mem_alloc(self.host[idx].nbytes)
            self.bindings[idx] = int(self.device[idx])

        # Mirrors PyTorch script meta
        self.traffic_classes = TRAFFIC_CLASS_IDS
        self.class_mapping = CLASS_MAPPING
        self.vehicle_classes = VEHICLE_CLASSES

        # Signal readiness on stdout (your Rust waits for this)
        print("READY", flush=True)

    def _ensure_shapes(self, n, c, h, w):
        """Set dynamic shape if needed and (re)allocate buffers."""
        inp_idx = 0  # assume first binding == input
        if any(s < 0 for s in self.engine.get_binding_shape(inp_idx)):
            self.context.set_binding_shape(inp_idx, (n, c, h, w))
        for idx in range(self.engine.num_bindings):
            shape = tuple(self.context.get_binding_shape(idx))
            dt = trt.nptype(self.engine.get_binding_dtype(idx))
            vol = int(np.prod(shape))
            if self.host[idx].size != vol:
                self.host[idx] = cuda.pagelocked_empty(vol, dt)
                self.device[idx] = cuda.mem_alloc(self.host[idx].nbytes)
                self.bindings[idx] = int(self.device[idx])

    def infer(self, image_bytes):
        # Decode input
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return self._make_response([], 0, 0, error="Decode failed")
        lb, r, padw, padh, orig_w, orig_h = letterbox_bgr(img, NET_SIZE)

        # BGR->RGB, HWC->CHW, normalize to 0..1
        x = lb[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        x = np.expand_dims(x, 0).copy()  # (1,3,640,640)

        # Shapes/buffers
        self._ensure_shapes(1, 3, NET_SIZE, NET_SIZE)

        # HtoD
        inp_idx = 0
        np.copyto(self.host[inp_idx], x.ravel())
        cuda.memcpy_htod_async(self.device[inp_idx], self.host[inp_idx], self.stream)

        # Execute
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)

        # DtoH for all outputs
        out_arrays = []
        for idx in range(self.engine.num_bindings):
            if not self.engine.binding_is_input(idx):
                cuda.memcpy_dtoh_async(self.host[idx], self.device[idx], self.stream)
        self.stream.synchronize()
        for idx in range(self.engine.num_bindings):
            if not self.engine.binding_is_input(idx):
                shape = tuple(self.context.get_binding_shape(idx))
                out_arrays.append(np.array(self.host[idx]).reshape(shape))

        if not out_arrays:
            return self._make_response([], orig_w, orig_h, error="No outputs")

        # Pick the largest output as detection head (raw YOLOv5: (1, N, 85))
        out = max(out_arrays, key=lambda a: a.size)
        raw = out.reshape(out.shape[0], -1, out.shape[-1])[0]

        if raw.shape[-1] < 85:
            return self._make_response([], orig_w, orig_h, error=f"Unexpected output shape {raw.shape}")

        # Decode like Ultralytics: score = obj_conf * class_conf
        obj = raw[:, 4]
        keep = obj > CONF_THR
        raw = raw[keep]
        if raw.size == 0:
            return self._make_response([], orig_w, orig_h)

        cls_probs = raw[:, 5:]
        cls_ids = np.argmax(cls_probs, axis=1)
        cls_conf = cls_probs[np.arange(len(cls_probs)), cls_ids]
        scores = cls_conf * obj[keep]

        # Filter to traffic classes first
        tmask = np.isin(cls_ids, list(self.traffic_classes))
        if not tmask.any():
            return self._make_response([], orig_w, orig_h)
        raw = raw[tmask]
        scores = scores[tmask]
        cls_ids = cls_ids[tmask]

        # xywh(net pixels) -> xyxy(net pixels)
        x_c, y_c, w, h = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
        x1 = x_c - w / 2
        y1 = y_c - h / 2
        x2 = x_c + w / 2
        y2 = y_c + h / 2

        # Undo letterbox/pad to original image pixels
        x1 = (x1 - padw) / r
        y1 = (y1 - padh) / r
        x2 = (x2 - padw) / r
        y2 = (y2 - padh) / r
        x1 = np.clip(x1, 0, orig_w);
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w);
        y2 = np.clip(y2, 0, orig_h)

        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

        # Class-aware NMS to match Ultralytics defaults
        keep_idx = nms_class_aware(boxes_xyxy, scores, cls_ids, IOU_THR)

        detections = []
        for i in keep_idx:
            cid = int(cls_ids[i])
            if cid not in TRAFFIC_CLASS_IDS:
                continue
            name = ID_TO_NAME.get(cid, str(cid))
            xx1, yy1, xx2, yy2 = boxes_xyxy[i]
            detections.append({
                "class": name,
                "bbox": [float(xx1), float(yy1), float(xx2 - xx1), float(yy2 - yy1)],
                "confidence": float(scores[i])
            })

        return self._make_response(detections, orig_w, orig_h)

    def _make_response(self, detections, width, height, error=None):
        counts = {k: 0 for k in self.class_mapping.values()}
        counts.update({"total_vehicles": 0, "total_objects": len(detections)})
        for d in detections:
            obj_class = d["class"]
            if obj_class in self.class_mapping:
                counts[self.class_mapping[obj_class]] += 1
                if obj_class in self.vehicle_classes:
                    counts["total_vehicles"] += 1
        if error is not None:
            return {
                "error": error,
                "detections": [],
                "image_width": int(width),
                "image_height": int(height),
                "counts": counts
            }
        return {
            "detections": detections,
            "image_width": int(width),
            "image_height": int(height),
            "counts": counts
        }

    def send_response(self, data):
        blob = json.dumps(data).encode("utf-8")
        sys.stdout.buffer.write(struct.pack("<I", len(blob)))
        sys.stdout.buffer.write(blob)
        sys.stdout.buffer.flush()

    def run_server(self):
        empty_counts = {k: 0 for k in self.class_mapping.values()}
        empty_counts.update({"total_vehicles": 0, "total_objects": 0})
        while True:
            try:
                length_bytes = sys.stdin.buffer.read(4)
                if len(length_bytes) != 4:
                    break
                frame_len = struct.unpack("<I", length_bytes)[0]
                frame = sys.stdin.buffer.read(frame_len)
                if len(frame) != frame_len:
                    break
                result = self.infer(frame)
                self.send_response(result)
            except Exception as e:
                err = str(e)
                print(f"ERR {err}", file=sys.stderr, flush=True)
                self.send_response({
                    "error": err,
                    "detections": [],
                    "image_width": 0,
                    "image_height": 0,
                    "counts": empty_counts
                })


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "yolov5n"
    engine_path = arg if arg.endswith(".engine") else f"{arg}_fp16.engine"
    server = PersistentTRTInferenceServer(engine_path)
    server.run_server()


if __name__ == "__main__":
    main()
