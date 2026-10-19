"""1차.mp4 마지막 구간(왼쪽 사람이 책상에 엎드림)에서 모델별로 엎드린 사람을 잡는지 본다.

엎드린 사람 영역: 960x540 화면 왼쪽 위 (x 0~330, y 80~400). 사람 박스 중심이 이 안에 있으면 그 사람으로 본다.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).parent
VIDEO = sys.argv[1]
START, END = 15730, 15814
REGION = (0, 80, 330, 400)  # x1, y1, x2, y2
MODELS = ["yolov8n", "yolov8s", "yolov8m", "yolo11n", "yolo11s", "yolo11m", "yolo11l",
          "yolo12s", "yolo26n", "yolo26s", "yolo26m",
          "yolov5su", "yolov9s", "yolov10s", "rtdetr-l", "yolov8s-worldv2"]
SHOW = 15790  # 캡처할 프레임


def main():
    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, START)
    frames = []
    for _ in range(START, END):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    out = ROOT / "1차" / "sleep"
    out.mkdir(parents=True, exist_ok=True)
    rows = {}
    for m in MODELS:
        model = YOLO(str(ROOT / "weights" / f"{m}.pt"))
        if "world" in m:
            model.set_classes(["person"])
        best = []
        for i, f in enumerate(frames):
            r = model.predict(f, imgsz=960, conf=0.1, classes=[0], verbose=False)[0]
            s = 0.0
            for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                if REGION[0] <= cx <= REGION[2] and REGION[1] <= cy <= REGION[3]:
                    s = max(s, c)
            best.append(s)
            if START + i == SHOW:
                cv2.imencode(".jpg", r.plot(line_width=2, font_size=12))[1].tofile(str(out / f"{m}.jpg"))
        b = np.array(best)
        rows[m] = {"n": len(b), "rate_25": float((b >= 0.25).mean()), "rate_35": float((b >= 0.35).mean()),
                   "mean": float(b[b > 0].mean()) if (b > 0).any() else 0.0, "min": float(b.min())}
        v = rows[m]
        print(f"{m:16s} 0.35이상 {v['rate_35']*100:5.1f}%  0.25이상 {v['rate_25']*100:5.1f}%  "
              f"평균 {v['mean']:.2f}  최저 {v['min']:.2f}", flush=True)
    (ROOT / "sleep_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
