"""YOLO 모델 1개의 리소스 사용량을 잰다. 모델마다 새 프로세스로 실행해야 메모리 값이 섞이지 않는다.

사용법: python resource.py <모델.pt>   → 결과를 JSON 한 줄로 출력
        python resource.py --all       → 11개 모델을 각각 새 프로세스로 돌려 resource.json 저장
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
WEIGHTS = ROOT / "weights"
SAMPLE = ROOT / "results" / "_crop_sample.jpg"
IMGSZ = 960
MODELS = ["yolov8n", "yolov8s", "yolov8m", "yolo11n", "yolo11s", "yolo11m", "yolo11l",
          "yolo12s", "yolo26n", "yolo26s", "yolo26m",
          "yolov5su", "yolov9s", "yolov10s", "rtdetr-l", "yolov8s-worldv2"]


def measure(name: str) -> dict:
    import cv2
    import numpy as np
    import psutil
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    frame = cv2.imdecode(np.fromfile(str(SAMPLE), np.uint8), 1)
    proc = psutil.Process()
    path = WEIGHTS / f"{name}.pt"
    out = {"model": name, "file_MB": path.stat().st_size / 1e6}

    # CPU: 추론 중 늘어난 프로세스 메모리(RSS)와 1장 처리 시간
    rss0 = proc.memory_info().rss
    model = YOLO(str(path))
    out["params_M"] = sum(p.numel() for p in model.model.parameters()) / 1e6
    out["gflops"] = get_flops(model.model, IMGSZ)
    model.predict(frame, imgsz=IMGSZ, device="cpu", verbose=False)
    t = []
    for _ in range(10):
        s = time.perf_counter()
        model.predict(frame, imgsz=IMGSZ, device="cpu", verbose=False)
        t.append(time.perf_counter() - s)
    out["cpu_ms"] = 1000 * sorted(t)[len(t) // 2]
    out["ram_MB"] = (proc.memory_info().rss - rss0) / 1e6

    # GPU: 1장 처리 시간과 최대 VRAM
    torch.cuda.reset_peak_memory_stats()
    model = YOLO(str(path))
    for _ in range(5):
        model.predict(frame, imgsz=IMGSZ, device=0, verbose=False)
    t = []
    for _ in range(50):
        torch.cuda.synchronize()
        s = time.perf_counter()
        model.predict(frame, imgsz=IMGSZ, device=0, verbose=False)
        torch.cuda.synchronize()
        t.append(time.perf_counter() - s)
    out["gpu_ms"] = 1000 * sorted(t)[len(t) // 2]
    out["vram_MB"] = torch.cuda.max_memory_allocated() / 1e6
    return out


def main():
    if sys.argv[1] != "--all":
        print(json.dumps(measure(sys.argv[1])))
        return
    rows = []
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    for m in MODELS:
        r = subprocess.run([sys.executable, __file__, m], capture_output=True, text=True, encoding="utf-8", env=env)
        if r.returncode:
            print(m, "실패:", r.stderr[-500:], flush=True)
            continue
        row = json.loads(r.stdout.strip().splitlines()[-1])
        rows.append(row)
        print(f"{m:8s} file {row['file_MB']:5.1f}MB  {row['gflops']:6.1f}GFLOPs  "
              f"GPU {row['gpu_ms']:5.1f}ms {row['vram_MB']:5.0f}MB  CPU {row['cpu_ms']:6.0f}ms {row['ram_MB']:5.0f}MB",
              flush=True)
    (ROOT / "resource.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
