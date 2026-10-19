"""잊지 않게 다시 학습하기 위한 데이터셋 v2.

v1(휴대폰 영상 181장)에 사람 데이터를 더한다.
- 1차.mp4 앞 80%(프레임 0~12650, 2초마다): 테이블은 고정 정답 박스, 사람·물건은 yolov10s 자동 라벨
- 엎드린 사람 사진 6장(3번씩 복사): 사람·물건은 yolov10s(기준 0.2), 책상은 교사 모델 4개 후보
1차.mp4 뒤 20%(엎드린 구간 포함)와 웹캠 영상은 시험용이라 넣지 않는다.
"""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).parent
W = ROOT.parent / "YOLO비교" / "weights"
V1 = ROOT / "dataset"
V2 = ROOT / "dataset_v2"
VID_1CHA = r"G:/내 드라이브/Cloud_Downloads/1차.mp4"
TRAIN_END = 12650
GT_1CHA = [0.08, 0.13, 0.74, 1.0]
KEEP = {0, 24, 25, 26, 28, 39, 41, 63, 64, 66, 67, 73}
EOP = sorted((ROOT.parent / "YOLO검증" / "images").glob("엎드림_*.jpg"))
TEACHERS = ["yolo11l", "yolo11m", "yolov9s", "rtdetr-l"]


def yl(c, b):
    x1, y1, x2, y2 = [min(max(v, 0.0), 1.0) for v in b]
    return f"{c} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"


def save(path, img):
    cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(str(path))


def others(model, img, conf):
    r = model.predict(img, imgsz=960, conf=conf, verbose=False)[0]
    return [(int(c), b) for b, c in zip(r.boxes.xyxyn.tolist(), r.boxes.cls.tolist()) if int(c) in KEEP]


def nms_tables(cands, thr=0.5):
    def iou(a, b):
        x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
        i = max(0, x2 - x1) * max(0, y2 - y1)
        return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)
    keep = []
    for c, b in sorted(cands, reverse=True):
        if all(iou(b, k) < thr for k in keep):
            keep.append(b)
    return keep


def main():
    if V2.exists():
        shutil.rmtree(V2)
    shutil.copytree(V1 / "images", V2 / "images")
    shutil.copytree(V1 / "labels", V2 / "labels")
    stu = YOLO(str(W / "yolov10s.pt"))
    log = {"1cha": [], "eop": {}}

    # 1차.mp4 앞 80%
    cap = cv2.VideoCapture(VID_1CHA)
    i = 0
    while i < TRAIN_END:
        ok = cap.grab()
        if not ok:
            break
        if i % 24 == 0:
            _, f = cap.retrieve()
            n = f"c{i:05d}"
            sp = "val" if (i // 24) % 10 == 5 else "train"
            save(V2 / "images" / sp / f"{n}.jpg", f)
            lines = [yl(60, GT_1CHA)] + [yl(c, b) for c, b in others(stu, f, 0.35)]
            (V2 / "labels" / sp / f"{n}.txt").write_text("\n".join(lines))
            log["1cha"].append(n)
        i += 1

    # 엎드린 사람 사진
    teachers = [YOLO(str(W / f"{t}.pt")) for t in TEACHERS]
    for p in EOP:
        img = cv2.imdecode(np.fromfile(str(p), np.uint8), 1)
        cands = []
        for t in teachers:
            r = t.predict(img, imgsz=960, conf=0.3, classes=[60], verbose=False)[0]
            cands += list(zip(r.boxes.conf.tolist(), r.boxes.xyxyn.tolist()))
        tables = nms_tables(cands)
        oth = others(stu, img, 0.2)
        log["eop"][p.stem] = {"tables": tables, "others": oth}
        lines = [yl(60, b) for b in tables] + [yl(c, b) for c, b in oth]
        for k in range(3):
            n = f"eop{p.stem[-1]}_{k}"
            save(V2 / "images" / "train" / f"{n}.jpg", img)
            (V2 / "labels" / "train" / f"{n}.txt").write_text("\n".join(lines))

    data = (V1 / "data.yaml").read_text(encoding="utf-8").replace(str(V1.resolve()), str(V2.resolve()))
    (V2 / "data.yaml").write_text(data, encoding="utf-8")
    (ROOT / "dataset_v2_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    for sp in ("train", "val"):
        print(sp, len(list((V2 / "images" / sp).glob("*.jpg"))))


if __name__ == "__main__":
    main()
