"""공개 저장소용 이미지: 사람 영역을 모자이크한 뒤 검출 박스를 다시 그린다.

모자이크 영역은 사람 검출이 강한 두 모델(파인튜닝 yolo26s v2, yolov10s)을 낮은 기준(0.15)으로 돌려 합친다.
그 위에 보여 줄 모델의 검출 결과(박스·이름·신뢰도)를 그린다.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).parent
W = ROOT.parent / "YOLO비교" / "weights"
OUT = Path(sys.argv[1])
VID_WEBCAM = r"G:/내 드라이브/Cloud_Downloads/Raspberry Pi USB Webcam - [InPrivate] - Microsoft_ Edge 2026-10-18 14-25-32.mp4"
VID_1CHA = r"G:/내 드라이브/Cloud_Downloads/1차.mp4"
CROP = (194, 1015, 15, 1615)
V2_26S = ROOT / "runs" / "v2_yolo26s" / "weights" / "best.pt"
V2_11S = ROOT / "runs" / "v2_full" / "weights" / "best.pt"

MASKERS = [YOLO(str(V2_26S)), YOLO(str(W / "yolov10s.pt"))]
COLORS = {0: (255, 80, 40), 60: (40, 160, 255)}


def save(path: Path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile(str(path))


def mosaic(img):
    out = img.copy()
    h, w = img.shape[:2]
    for m in MASKERS:
        r = m.predict(img, imgsz=960, conf=0.15, classes=[0], verbose=False)[0]
        for x1, y1, x2, y2 in r.boxes.xyxy.int().tolist():
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            roi = out[y1:y2, x1:x2]
            small = cv2.resize(roi, (max(1, (x2 - x1) // 24), max(1, (y2 - y1) // 24)))
            out[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    return out


def draw(img, result, title=None, classes=None):
    out = img.copy()
    names = result.names
    th = max(2, out.shape[1] // 500)
    for (x1, y1, x2, y2), c, s in zip(result.boxes.xyxy.int().tolist(), result.boxes.cls.tolist(),
                                      result.boxes.conf.tolist()):
        c = int(c)
        if classes is not None and c not in classes:
            continue
        col = COLORS.get(c, (60, 200, 60))
        cv2.rectangle(out, (x1, y1), (x2, y2), col, th)
        label = f"{names[c]} {s:.2f}"
        fs = out.shape[1] / 1800
        (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        cv2.rectangle(out, (x1, max(0, y1 - tht - 8)), (x1 + tw + 6, y1), col, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th)
    if title:
        fs = out.shape[1] / 1100
        cv2.putText(out, title, (12, out.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 4)
        cv2.putText(out, title, (12, out.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th)
    return out


def frame(video, idx, crop=False):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    _, f = cap.read()
    if crop:
        y1, y2, x1, x2 = CROP
        f = f[y1:y2, x1:x2].copy()
    return f


def main():
    webcam = frame(VID_WEBCAM, 0, crop=True)
    sleep = frame(VID_1CHA, 15790)
    empty = frame(VID_1CHA, 13441)
    webcam_m, sleep_m, empty_m = mosaic(webcam), mosaic(sleep), mosaic(empty)

    # 1) YOLO 16종 사전학습 비교 (웹캠 첫 프레임)
    zoo = ["yolov8n", "yolov8s", "yolov8m", "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo12s",
           "yolo26n", "yolo26s", "yolo26m", "yolov5su", "yolov9s", "yolov10s", "rtdetr-l", "yolov8s-worldv2"]
    keep = {0, 24, 26, 39, 41, 60, 63, 64, 66, 67, 73}
    for n in zoo:
        m = YOLO(str(W / f"{n}.pt"))
        if "world" in n:
            m.set_classes(["person", "laptop", "cell phone", "mouse", "backpack", "table", "chair"])
            r = m.predict(webcam, imgsz=960, conf=0.25, verbose=False)[0]
            save(OUT / "YOLO비교" / "검출화면" / f"{n}.jpg", cv2.resize(draw(webcam_m, r, n), (960, 492)))
            continue
        r = m.predict(webcam, imgsz=960, conf=0.25, verbose=False)[0]
        save(OUT / "YOLO비교" / "검출화면" / f"{n}.jpg", cv2.resize(draw(webcam_m, r, n, keep), (960, 492)))
    print("zoo done", flush=True)

    # 2) 파인튜닝 전후 (사람·테이블만)
    pairs = [("yolo11s_원본", W / "yolo11s.pt"), ("yolo11s_v2", V2_11S),
             ("yolo26s_원본", W / "yolo26s.pt"), ("yolo26s_v2", V2_26S)]
    for tag, w in pairs:
        m = YOLO(str(w))
        for scene, img, img_m in [("웹캠", webcam, webcam_m), ("엎드림", sleep, sleep_m), ("빈자리", empty, empty_m)]:
            r = m.predict(img, imgsz=960, conf=0.25, verbose=False)[0]
            # OpenCV는 한글을 못 그려서 그림 안 제목은 영어로 쓴다
            title = tag.replace("_원본", " original").replace("_v2", " v2 (fine-tuned)")
            save(OUT / "파인튜닝" / "결과화면" / f"{scene}_{tag}.jpg", draw(img_m, r, title, {0, 60}))
    print("finetune done", flush=True)


if __name__ == "__main__":
    main()
