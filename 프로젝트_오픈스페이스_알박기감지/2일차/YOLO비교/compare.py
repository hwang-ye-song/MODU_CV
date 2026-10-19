"""스터디룸 웹캠 영상으로 YOLO 버전별 검출 성능을 비교한다.

영상은 Edge 화면 녹화라서 카메라 영역만 잘라서 쓴다.
정답(장면에 실제로 있는 개수)은 프레임을 직접 보고 센 값이다.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).parent
VIDEO = sys.argv[1]
CROP = (194, 1015, 15, 1615)          # y1, y2, x1, x2 (카메라 영역)
STEP = 14                              # 약 0.5초마다 1프레임
IMGSZ = 960                            # src/detector.py 와 같은 값
CONF = 0.25

MODELS = [
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt",
    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt",
    "yolo12s.pt",
    "yolo26n.pt", "yolo26s.pt", "yolo26m.pt",
]
# 버전별 구조가 다른 모델 (크기는 s/l급으로 맞춤)
#   yolov5su: v5 백본 + 앵커 없는 헤드 / yolov9s: GELAN·PGI / yolov10s: NMS 없는 이중 할당
#   rtdetr-l: 트랜스포머 디코더(DETR 계열) / yolov8s-worldv2: 텍스트 프롬프트로 클래스 지정
ARCH_MODELS = ["yolov5su.pt", "yolov9s.pt", "yolov10s.pt", "rtdetr-l.pt", "yolov8s-worldv2.pt"]
# YOLO-World 프롬프트. COCO 이름 대신 'table'로 물어보고 dining table로 채점한다
WORLD_PROMPTS = ["person", "laptop", "cell phone", "mouse", "backpack", "table", "chair"]
ALIAS = {"table": "dining table"}
# 장면의 실제 개수 (프레임 확인 결과)
GT = {"person": 3, "dining table": 1, "laptop": 3, "cell phone": 2,
      "mouse": 1, "backpack": 1}


def load_frames():
    cap = cv2.VideoCapture(VIDEO)
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % STEP == 0:
            y1, y2, x1, x2 = CROP
            frames.append(f[y1:y2, x1:x2].copy())
        i += 1
    return frames


def imwrite(path: Path, img):
    cv2.imencode(".jpg", img)[1].tofile(str(path))


def main():
    frames = load_frames()
    out_img = ROOT / "results"
    out_img.mkdir(exist_ok=True)
    imwrite(out_img / "_crop_sample.jpg", frames[0])
    models = sys.argv[2:] or MODELS
    old = ROOT / "results.json"
    report = (json.loads(old.read_text(encoding="utf-8")) if sys.argv[2:] and old.exists()
              else {"n_frames": len(frames), "gt": GT, "imgsz": IMGSZ, "conf": CONF, "models": {}})

    for w in models:
        model = YOLO(w)
        if "world" in w:
            model.set_classes(WORLD_PROMPTS)
        model.predict(frames[0], imgsz=IMGSZ, conf=CONF, verbose=False)  # 워밍업
        counts = {k: [] for k in GT}
        confs = {k: [] for k in GT}
        others: dict[str, int] = {}
        times = []
        for idx, f in enumerate(frames):
            t = time.perf_counter()
            r = model.predict(f, imgsz=IMGSZ, conf=CONF, verbose=False)[0]
            times.append(time.perf_counter() - t)
            names = [ALIAS.get(r.names[int(c)], r.names[int(c)]) for c in r.boxes.cls.tolist()]
            scores = r.boxes.conf.tolist()
            for k in GT:
                counts[k].append(names.count(k))
                confs[k] += [s for n, s in zip(names, scores) if n == k]
            for n in names:
                if n not in GT:
                    others[n] = others.get(n, 0) + 1
            if idx == 0:
                imwrite(out_img / f"{Path(w).stem}.jpg", r.plot(line_width=2, font_size=14))

        stem = Path(w).stem
        cls = {}
        for k, gt in GT.items():
            c = np.array(counts[k])
            cls[k] = {
                "mean": float(c.mean()),
                "found_rate": float(np.minimum(c, gt).sum() / (gt * len(c))),  # 재현율 근사
                "over": float(np.maximum(c - gt, 0).mean()),                 # 초과 검출(오검출 근사)
                "conf": float(np.mean(confs[k])) if confs[k] else 0.0,
            }
        recall = float(np.mean([v["found_rate"] for v in cls.values()]))
        report["models"][stem] = {
            "ms": float(np.mean(times) * 1000),
            "params_M": sum(p.numel() for p in model.model.parameters()) / 1e6,
            "classes": cls,
            "recall_avg": recall,
            "others": dict(sorted(others.items(), key=lambda x: -x[1])),
        }
        print(f"{stem:10s} {np.mean(times)*1000:6.1f}ms  recall {recall:.2f}  "
              + "  ".join(f"{k}:{cls[k]['mean']:.1f}" for k in GT), flush=True)

    (ROOT / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
