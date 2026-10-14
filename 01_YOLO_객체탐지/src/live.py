"""학습한 모델로 웹캠 화면의 물품마다 정상/불량 박스를 실시간으로 표시한다.

    python src/live.py
    python src/live.py --conf 0.4 --cam 1

화면 상단: 전체 판정 (불량이 하나라도 있으면 NG) + 정상/불량 개수 + FPS
키:  q = 종료   s = 현재 화면 저장(results/live_*.jpg)
"""
import argparse
import time

import cv2
from ultralytics import YOLO

from common import BEST_WEIGHTS, COLORS, RESULTS_DIR, get_device


def draw(frame, result):
    """박스를 그리고 (정상 개수, 불량 개수) 를 돌려준다."""
    n_ok = n_ng = 0
    for box, cls, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()):
        name = result.names[int(cls)]
        color = COLORS.get(name, (255, 255, 0))
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, f"{name} {conf:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if name == "defect":
            n_ng += 1
        else:
            n_ok += 1
    return n_ok, n_ng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(BEST_WEIGHTS))
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.5, help="이 값 이상인 박스만 표시")
    args = ap.parse_args()

    model = YOLO(args.weights)
    device = get_device()
    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"카메라 {args.cam} 을(를) 열 수 없습니다.")

    prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        r = model.predict(frame, conf=args.conf, device=device, verbose=False)[0]
        n_ok, n_ng = draw(frame, r)

        now = time.time()
        fps, prev = 1.0 / max(now - prev, 1e-6), now

        verdict, color = ("NG", COLORS["defect"]) if n_ng else ("OK", COLORS["normal"])
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (30, 30, 30), -1)
        cv2.putText(frame, verdict, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame, f"normal {n_ok} | defect {n_ng} | {fps:.0f} FPS", (100, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("live", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            out = RESULTS_DIR / f"live_{verdict}_{time.strftime('%H%M%S')}.jpg"
            cv2.imwrite(str(out), frame)
            print("저장:", out)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
