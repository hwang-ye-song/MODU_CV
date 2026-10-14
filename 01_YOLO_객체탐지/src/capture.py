"""웹캠으로 학습용 사진을 찍어 data/raw/images 에 저장한다.

    python src/capture.py            # 기본 카메라(0)
    python src/capture.py --cam 1    # 외장 카메라

키:  space = 저장   q = 종료

팁: 한 장에 정상/불량 물품을 섞어서 여러 개 놓고, 위치·각도·조명을 바꿔가며 찍는다.
"""
import argparse
import time

import cv2

from common import RAW_IMG_DIR, list_images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)  # Windows 에서 카메라가 빨리 열린다
    if not cap.isOpened():
        raise SystemExit(f"카메라 {args.cam} 을(를) 열 수 없습니다.")

    count = len(list_images(RAW_IMG_DIR))
    flash_until = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        view = frame.copy()
        cv2.putText(view, f"saved: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(view, "[space] save  [q] quit", (10, view.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        if time.time() < flash_until:
            cv2.rectangle(view, (0, 0), (view.shape[1] - 1, view.shape[0] - 1), (0, 255, 255), 6)
        cv2.imshow("capture", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            # 화면 글자가 들어가지 않은 원본 frame 을 저장
            name = f"img_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}.jpg"
            cv2.imwrite(str(RAW_IMG_DIR / name), frame)
            count += 1
            flash_until = time.time() + 0.3

    cap.release()
    cv2.destroyAllWindows()
    print(f"저장된 사진 {count}장 → {RAW_IMG_DIR}")


if __name__ == "__main__":
    main()
