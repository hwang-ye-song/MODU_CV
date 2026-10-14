"""간단한 박스 라벨링 도구 (OpenCV). 결과는 YOLO 형식 txt 로 저장된다.

    python src/label.py              # 라벨 없는 사진부터 시작
    python src/label.py --all        # 처음 사진부터

조작:
    마우스 드래그   현재 클래스로 박스 그리기
    1 / 2          현재 클래스 = normal / defect
    우클릭          커서 아래 박스의 클래스 바꾸기 (normal <-> defect)
    z              마지막 박스 지우기      c   이 사진 박스 전부 지우기
    d 또는 space   저장 후 다음 사진       a   저장 후 이전 사진
    q              저장 후 종료

YOLO 라벨 한 줄 = "클래스번호 중심x 중심y 너비 높이" (0~1 로 정규화)
박스가 하나도 없는 사진은 빈 txt 로 저장된다 (= 물품이 없는 배경 사진).
"""
import argparse

import cv2

from common import CLASSES, COLORS, RAW_IMG_DIR, label_path, list_images

MAX_W, MAX_H = 1280, 800   # 화면에 맞게 축소해서 보여준다 (저장은 원본 비율 기준이라 상관없음)


def load_boxes(img_path, w, h):
    """txt → [(cls, x1, y1, x2, y2)] (원본 픽셀 좌표)"""
    p = label_path(img_path)
    boxes = []
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                c, cx, cy, bw, bh = map(float, line.split()[:5])
                boxes.append((int(c), (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return boxes


def save_boxes(img_path, boxes, w, h):
    lines = []
    for c, x1, y1, x2, y2 in boxes:
        x1, x2 = sorted((max(0, x1), min(w, x2)))
        y1, y2 = sorted((max(0, y1), min(h, y2)))
        lines.append(f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    label_path(img_path).write_text("\n".join(lines) + ("\n" if lines else ""))


class Labeler:
    def __init__(self, images, start):
        self.images, self.idx = images, start
        self.cls = 1  # 기본은 defect (불량을 표시하는 경우가 더 많다고 가정)
        self.drag_start = self.cursor = None
        cv2.namedWindow("label")
        cv2.setMouseCallback("label", self.on_mouse)

    def load(self):
        self.img = cv2.imread(str(self.images[self.idx]))
        self.h, self.w = self.img.shape[:2]
        self.scale = min(MAX_W / self.w, MAX_H / self.h, 1.0)
        self.boxes = load_boxes(self.images[self.idx], self.w, self.h)

    def save(self):
        save_boxes(self.images[self.idx], self.boxes, self.w, self.h)

    def on_mouse(self, event, x, y, flags, _):
        ox, oy = x / self.scale, y / self.scale   # 화면 좌표 → 원본 좌표
        self.cursor = (ox, oy)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (ox, oy)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start:
            (x1, y1), (x2, y2) = self.drag_start, (ox, oy)
            if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:   # 실수로 클릭만 한 경우 무시
                self.boxes.append((self.cls, min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
            self.drag_start = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            # 커서를 포함하는 박스 중 가장 작은 것의 클래스를 뒤집는다
            hits = [i for i, (c, x1, y1, x2, y2) in enumerate(self.boxes) if x1 <= ox <= x2 and y1 <= oy <= y2]
            if hits:
                i = min(hits, key=lambda i: (self.boxes[i][3] - self.boxes[i][1]) * (self.boxes[i][4] - self.boxes[i][2]))
                c, *xy = self.boxes[i]
                self.boxes[i] = (1 - c, *xy)

    def draw(self):
        s = self.scale
        view = cv2.resize(self.img, (int(self.w * s), int(self.h * s)))
        for c, x1, y1, x2, y2 in self.boxes:
            color = COLORS[CLASSES[c]]
            cv2.rectangle(view, (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s)), color, 2)
            cv2.putText(view, CLASSES[c], (int(x1 * s), max(15, int(y1 * s) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if self.drag_start and self.cursor:
            (x1, y1), (x2, y2) = self.drag_start, self.cursor
            cv2.rectangle(view, (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s)), COLORS[CLASSES[self.cls]], 1)
        n_done = sum(label_path(p).exists() for p in self.images)
        info = (f"[{self.idx + 1}/{len(self.images)}] {self.images[self.idx].name} | labeled {n_done} | "
                f"class: {CLASSES[self.cls]} (1/2)")
        cv2.rectangle(view, (0, 0), (view.shape[1], 28), (40, 40, 40), -1)
        cv2.putText(view, info, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS[CLASSES[self.cls]], 1)
        cv2.imshow("label", view)

    def run(self):
        self.load()
        while True:
            self.draw()
            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                if cv2.getWindowProperty("label", cv2.WND_PROP_VISIBLE) < 1:   # 창 X 버튼
                    self.save()
                    break
                continue
            if key in (ord("1"), ord("2")):
                self.cls = key - ord("1")
            elif key == ord("z") and self.boxes:
                self.boxes.pop()
            elif key == ord("c"):
                self.boxes.clear()
            elif key in (ord("d"), ord(" ")):
                self.save()
                if self.idx < len(self.images) - 1:
                    self.idx += 1
                    self.load()
            elif key == ord("a"):
                self.save()
                if self.idx > 0:
                    self.idx -= 1
                    self.load()
            elif key == ord("q"):
                self.save()
                break
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="라벨 유무와 상관없이 첫 사진부터")
    args = ap.parse_args()

    images = list_images(RAW_IMG_DIR)
    if not images:
        raise SystemExit(f"{RAW_IMG_DIR} 에 사진이 없습니다. python src/capture.py 로 먼저 찍으세요.")
    start = 0 if args.all else next((i for i, p in enumerate(images) if not label_path(p).exists()), 0)
    Labeler(images, start).run()

    done = [p for p in images if label_path(p).exists()]
    print(f"라벨 완료 {len(done)}/{len(images)}장")


if __name__ == "__main__":
    main()
