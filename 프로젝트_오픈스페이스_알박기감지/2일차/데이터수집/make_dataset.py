"""휴대폰으로 테이블 주변을 돌며 찍은 영상에서 테이블 학습용 데이터셋을 만든다.

1) 3프레임마다 1장 추출
2) 테이블(dining table): 테이블을 잘 잡던 모델 4개의 후보 중 신뢰도 가장 높은 박스 → 사람이 눈으로 검수
3) 사람·물건: yolov10s 자동 라벨 (엎드린 사람·물건 검출이 가장 좋았던 모델)
클래스 번호는 COCO 80개 그대로 둔다 (0=person, 60=dining table ...). 그래야 기존 코드가 그대로 돈다.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).parent
W = ROOT.parent / "YOLO비교" / "weights"
VIDEO = sys.argv[1]
STEP = 3
TABLE = 60
# src/detector.py 의 사람·물건 클래스
KEEP = {0, 24, 25, 26, 28, 39, 41, 63, 64, 66, 67, 73}
TEACHERS = ["yolo11l", "yolo11m", "yolov9s", "rtdetr-l"]


def save(path, img):
    cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(str(path))


def main():
    img_dir = ROOT / "frames"
    img_dir.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(VIDEO)
    names, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % STEP == 0:
            n = f"f{i:04d}"
            save(img_dir / f"{n}.jpg", f)
            names.append(n)
        i += 1
    print("frames", len(names))

    # 테이블 후보
    cand = {n: [] for n in names}
    for t in TEACHERS:
        m = YOLO(str(W / f"{t}.pt"))
        for n in names:
            f = cv2.imdecode(np.fromfile(str(img_dir / f"{n}.jpg"), np.uint8), 1)
            r = m.predict(f, imgsz=960, conf=0.05, classes=[TABLE], verbose=False)[0]
            for b, c in zip(r.boxes.xyxyn.tolist(), r.boxes.conf.tolist()):
                cand[n].append({"model": t, "conf": c, "box": b})
        print("teacher done", t, flush=True)

    # 사람·물건
    other = {}
    m = YOLO(str(W / "yolov10s.pt"))
    for n in names:
        f = cv2.imdecode(np.fromfile(str(img_dir / f"{n}.jpg"), np.uint8), 1)
        r = m.predict(f, imgsz=960, conf=0.3, verbose=False)[0]
        other[n] = [{"cls": int(c), "conf": s, "box": b} for b, c, s in
                    zip(r.boxes.xyxyn.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()) if int(c) in KEEP]

    table = {}
    for n in names:
        best = max(cand[n], key=lambda d: d["conf"]) if cand[n] else None
        table[n] = best
    (ROOT / "auto_labels.json").write_text(json.dumps({"table": table, "other": other, "cand": cand}, indent=1),
                                          encoding="utf-8")
    found = sum(v is not None for v in table.values())
    print(f"table candidate in {found}/{len(names)} frames")


if __name__ == "__main__":
    main()
