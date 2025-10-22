import glob
import os
import random
import re
import subprocess
import sys
from typing import List

import cv2

from inference_pytorch import PersistentPyTorchInferenceServer

CONF_RED_THRESHOLD = 0.50

try:
    import psutil
except:
    psutil = None

try:
    import torch
except:
    torch = None


def _mib(n): return f"{n / (1024 ** 2):.1f} MiB"


def mem_report(tag: str):
    parts = [f"[MEM] {tag}"]
    if psutil:
        parts.append(f"CPU RSS={_mib(psutil.Process(os.getpid()).memory_info().rss)}")
    else:
        parts.append("CPU RSS=?")
    if torch and torch.cuda.is_available():
        try:
            a = torch.cuda.memory_allocated()
            r = torch.cuda.memory_reserved()
            p = torch.cuda.max_memory_allocated()
            parts.append(f"CUDA alloc={_mib(a)} res={_mib(r)} peak={_mib(p)}")
        except:
            parts.append("CUDA stats err")
    else:
        # AMD fallback via rocm-smi
        try:
            out = subprocess.check_output(["rocm-smi", "--showmemuse", "--json"], stderr=subprocess.DEVNULL, text=True)
            parts.append(f"ROCm mem={out.strip()[:120]}...")
        except:
            parts.append("GPU mem n/a")
    print(", ".join(parts))


def draw_detections(img, dets):
    for d in dets:
        x, y, w, h = map(int, d["bbox"])
        conf = float(d.get("confidence", 0.0))
        cls = d.get("class", "")
        color = (0, 0, 255) if conf < CONF_RED_THRESHOLD else (0, 255, 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        label = f"{cls} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img, (x, y - th - 6), (x + tw + 4, y), color, -1)
        cv2.putText(img, label, (x + 2, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, lineType=cv2.LINE_AA)


def extract_seq_num(p: str) -> str:
    b = os.path.basename(p)
    m = re.search(r'_([0-9]+)\.(?:jpe?g|png)$', b, re.I) or re.search(r'(\d+)\.(?:jpe?g|png)$', b, re.I)
    return m.group(1) if m else "unknown"


def list_images_in_dir(d: str, max_seq: int = 35000) -> List[str]:
    files = []
    for pat in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        files.extend(glob.glob(os.path.join(d, pat)))
    out = []
    for f in files:
        s = extract_seq_num(f)
        if s.isdigit() and int(s) <= max_seq:
            out.append(f)
    return out or files


def run_models_on_images(paths: List[str], models: List[str]):
    if torch and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    mem_report("before init")
    dets = {m: PersistentPyTorchInferenceServer(m) for m in models}
    for p in paths:
        with open(p, "rb") as f:
            img_bytes = f.read()
        base = cv2.imread(p);
        seq = extract_seq_num(p)
        for m in models:
            if torch and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            mem_report(f"pre {m}:{os.path.basename(p)}")
            res = dets[m].process_frame(img_bytes)
            if torch and torch.cuda.is_available():
                torch.cuda.synchronize()
            mem_report(f"post {m}:{os.path.basename(p)}")
            img = base.copy()
            draw_detections(img, res.get("detections", []))
            outp = os.path.join(os.path.dirname(p), f"{m}_{seq}.jpg")
            cv2.imwrite(outp, img)
            print("Saved:", outp)


def main():
    if len(sys.argv) < 2:
        print("Usage: python inference_with_boxes.py <image_or_dir> [model1 model2 ...]")
        sys.exit(1)
    target = sys.argv[1]
    models = sys.argv[2:] if len(sys.argv) > 2 else ["yolov5n", "yolov5s", "yolov5m"]
    sample_dir = target if os.path.isdir(target) else os.path.dirname(os.path.abspath(target))
    imgs = list_images_in_dir(sample_dir, 35000)
    if not imgs:
        print("No images found:", sample_dir);
        sys.exit(1)
    chosen = random.sample(imgs, min(6, len(imgs)))
    print("Selected:", ", ".join(os.path.basename(c) for c in chosen))
    run_models_on_images(chosen, models)


if __name__ == "__main__":
    main()
