"""파인튜닝 전후 모델을 학습에 안 쓴 영상 2개로 비교한다.

A. 1차.mp4 (천장 고정 카메라) 뒤 20%: 2초마다 1장. 카메라가 고정이라 테이블 정답 박스가 모든 프레임에서 같다.
   + 마지막 엎드린 구간(15730~15813)에서 엎드린 사람을 기준값 0.35로 잡는지
B. 라즈베리파이 웹캠 영상(18초): 0.5초마다 1장, 카메라 영역만 잘라서 사용. 사람·물건 검출률도 같이 본다.

테이블은 '가장 신뢰도 높은 dining table 박스가 정답 박스와 IoU 0.5 이상'이면 맞힌 것으로 센다.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).parent
W = ROOT.parent / "YOLO비교" / "weights"
VID_1CHA = r"G:/내 드라이브/Cloud_Downloads/1차.mp4"
VID_WEBCAM = r"G:/내 드라이브/Cloud_Downloads/Raspberry Pi USB Webcam - [InPrivate] - Microsoft_ Edge 2026-10-18 14-25-32.mp4"
GT_1CHA = [0.08, 0.13, 0.74, 1.0]
GT_WEBCAM = [0.10, 0.13, 0.74, 1.0]
CROP = (194, 1015, 15, 1615)
SLEEP = (15730, 15814)
TEST_START = 12650  # 데이터셋 v2가 1차.mp4 앞 80%를 학습에 써서, 그 뒤만 채점한다
SLEEP_REGION = (0, 80, 330, 400)
ITEMS_GT = {"person": 3, "laptop": 3, "cell phone": 2, "mouse": 1, "backpack": 1}
MODELS = {
    "yolo11s (원본)": W / "yolo11s.pt",
    "yolo11s (테이블 파인튜닝)": ROOT / "runs" / "yolo11s_table" / "weights" / "best.pt",
    "yolo11s (v2 백본 고정)": ROOT / "runs" / "v2_freeze" / "weights" / "best.pt",
    "yolo11s (v2 전체 학습)": ROOT / "runs" / "v2_full" / "weights" / "best.pt",
    "yolo26s (v2)": ROOT / "runs" / "v2_yolo26s" / "weights" / "best.pt",
    "yolo26m (v2)": ROOT / "runs" / "v2_yolo26m" / "weights" / "best.pt",
    "yolo26l (v2)": ROOT / "runs" / "v2_yolo26l" / "weights" / "best.pt",
    "yolo26s (원본)": W / "yolo26s.pt",
    "yolo11l": W / "yolo11l.pt",
    "yolov9s": W / "yolov9s.pt",
    "yolov10s": W / "yolov10s.pt",
}


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def frames_1cha():
    cap = cv2.VideoCapture(VID_1CHA)
    out, i = [], 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % 24 == 0 and i >= TEST_START:
            _, f = cap.retrieve()
            out.append(f)
        i += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, SLEEP[0])
    sleep = [cap.read()[1] for _ in range(*SLEEP)]
    return out, sleep


def frames_webcam():
    cap = cv2.VideoCapture(VID_WEBCAM)
    out, i = [], 0
    y1, y2, x1, x2 = CROP
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % 14 == 0:
            out.append(f[y1:y2, x1:x2].copy())
        i += 1
    return out


def table_score(model, frames, gt, save=None):
    hit, det, confs, ious = 0, 0, [], []
    for k, f in enumerate(frames):
        r = model.predict(f, imgsz=960, conf=0.25, verbose=False)[0]
        tb = [(c, b) for b, c, s in zip(r.boxes.xyxyn.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())
              if int(s) == 60]
        if save is not None and k == save[1]:
            cv2.imencode(".jpg", r.plot(line_width=2, font_size=12))[1].tofile(str(save[0]))
        if not tb:
            continue
        c, b = max(tb)
        det += 1
        confs.append(c)
        ious.append(iou(b, gt))
        hit += iou(b, gt) >= 0.5
    n = len(frames)
    return {"correct": hit / n, "detected": det / n, "conf": float(np.mean(confs)) if confs else 0.0,
            "iou": float(np.mean(ious)) if ious else 0.0}


def main():
    f1, sleep = frames_1cha()
    fw = frames_webcam()
    print("frames", len(f1), len(sleep), len(fw), flush=True)
    shots = ROOT / "eval_shots"
    shots.mkdir(exist_ok=True)
    res = {}
    for name, w in MODELS.items():
        m = YOLO(str(w))
        tag = w.parent.parent.name if w.name == "best.pt" else w.stem
        a = table_score(m, f1, GT_1CHA, (shots / f"1cha_{tag}.jpg", 200))
        b = table_score(m, fw, GT_WEBCAM, (shots / f"webcam_{tag}.jpg", 0))
        s35 = 0
        for f in sleep:
            r = m.predict(f, imgsz=960, conf=0.1, classes=[0], verbose=False)[0]
            best = max([c for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())
                        if SLEEP_REGION[0] <= (x1 + x2) / 2 <= SLEEP_REGION[2]
                        and SLEEP_REGION[1] <= (y1 + y2) / 2 <= SLEEP_REGION[3]] or [0])
            s35 += best >= 0.35
        items = {k: 0 for k in ITEMS_GT}
        for f in fw:
            r = m.predict(f, imgsz=960, conf=0.25, verbose=False)[0]
            nm = [r.names[int(c)] for c, s in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist())
                  if not (int(c) == 0 and s < 0.35)]
            for k, g in ITEMS_GT.items():
                items[k] += min(nm.count(k), g) / g
        items = {k: v / len(fw) for k, v in items.items()}
        res[name] = {"1cha_table": a, "webcam_table": b, "sleep_035": s35 / len(sleep), "webcam_items": items}
        print(name, json.dumps(res[name], ensure_ascii=False), flush=True)
    (ROOT / "eval_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
