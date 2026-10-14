"""좌석 영역(ROI) 지정 도구.

사용법:
    python src/roi_editor.py --source 영상.mp4 --out seats.json
    python src/roi_editor.py --source 0            # 웹캠

조작:
    왼쪽 클릭   꼭짓점 추가
    Enter/우클릭 현재 다각형을 좌석으로 확정 (책상 + 의자가 들어가게 그리기)
    z          마지막 꼭짓점(없으면 마지막 좌석) 취소
    s          저장 후 종료
    q / Esc    저장하지 않고 종료
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def grab_frame(source: str, at_sec: float = 0.0) -> np.ndarray:
    if Path(source).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        img = cv2.imread(source)
        if img is None:
            raise SystemExit(f"이미지를 열 수 없습니다: {source}")
        return img
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if at_sec:
        cap.set(cv2.CAP_PROP_POS_MSEC, at_sec * 1000)
    for _ in range(5 if source.isdigit() else 1):   # 웹캠은 첫 몇 장이 어둡다
        ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"프레임을 읽을 수 없습니다: {source}")
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="영상/이미지 경로 또는 웹캠 번호")
    ap.add_argument("--out", default="seats.json")
    ap.add_argument("--at", type=float, default=0.0, help="영상에서 기준 프레임을 뽑을 시각(초)")
    args = ap.parse_args()

    frame = grab_frame(args.source, args.at)
    out_path = Path(args.out)
    seats, cur = [], []
    if out_path.exists():
        seats = json.loads(out_path.read_text(encoding="utf-8")).get("seats", [])
        print(f"기존 좌석 {len(seats)}개를 불러왔습니다.")

    def close_poly():
        nonlocal cur
        if len(cur) >= 3:
            seats.append({"id": f"S{len(seats) + 1}", "polygon": cur})
            print(f"좌석 {seats[-1]['id']} 추가")
        cur = []

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            cur.append([x, y])
        elif ev == cv2.EVENT_RBUTTONDOWN:
            close_poly()

    print("좌석 영역 지정: 왼쪽 클릭으로 꼭짓점 → Enter(우클릭)로 좌석 확정 → "
          "다 그렸으면 s로 저장 (z: 되돌리기, q: 저장 안 하고 종료)")
    win = "ROI editor (click: point, Enter: close, z: undo, s: save, q: quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        view = frame.copy()
        for s in seats:
            p = np.array(s["polygon"], np.int32)
            cv2.polylines(view, [p], True, (0, 255, 0), 2)
            cv2.putText(view, s["id"], tuple(p.min(axis=0) + [5, 25]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if cur:
            cv2.polylines(view, [np.array(cur, np.int32)], False, (0, 200, 255), 2)
            for pt in cur:
                cv2.circle(view, tuple(pt), 4, (0, 200, 255), -1)
        cv2.imshow(win, view)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10):
            close_poly()
        elif k == ord("z"):
            if cur:
                cur.pop()
            elif seats:
                print(f"좌석 {seats.pop()['id']} 삭제")
        elif k == ord("s"):
            close_poly()
            h, w = frame.shape[:2]
            data = {"frame_size": [w, h], "seats": seats}
            if out_path.exists():   # rules 등 다른 설정은 유지
                old = json.loads(out_path.read_text(encoding="utf-8"))
                data = {**old, **data}
            out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"{out_path}에 좌석 {len(seats)}개 저장")
            break
        elif k in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
