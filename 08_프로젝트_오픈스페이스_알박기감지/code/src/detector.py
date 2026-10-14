"""사전학습 YOLO(COCO)로 사람·물건을 검출하고 좌석 ROI에 배정한다."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

PERSON = 0
# 좌석에 '맡아 둔' 물건으로 볼 COCO 클래스 (의자·책상처럼 원래 있는 가구는 제외)
ITEM_CLASSES = {
    24: "backpack", 25: "umbrella", 26: "handbag", 28: "suitcase",
    39: "bottle", 41: "cup", 63: "laptop", 64: "mouse",
    66: "keyboard", 67: "cell phone", 73: "book",
}


@dataclass
class Detection:
    cls: int
    name: str
    conf: float
    box: tuple[int, int, int, int]   # x1, y1, x2, y2

    @property
    def is_person(self) -> bool:
        return self.cls == PERSON


class SeatDetector:
    def __init__(self, weights: str = "yolo11s.pt", conf: float = 0.25,
                 person_conf: float = 0.35, imgsz: int = 960, device: str | None = None):
        self.model = YOLO(weights)
        self.conf, self.person_conf, self.imgsz, self.device = conf, person_conf, imgsz, device

    def detect(self, frame: np.ndarray) -> list[Detection]:
        res = self.model.predict(
            frame, conf=self.conf, imgsz=self.imgsz, device=self.device,
            classes=[PERSON, *ITEM_CLASSES], verbose=False,
        )[0]
        out = []
        for (x1, y1, x2, y2), c, s in zip(res.boxes.xyxy.tolist(),
                                          res.boxes.cls.tolist(), res.boxes.conf.tolist()):
            c = int(c)
            if c == PERSON and s < self.person_conf:
                continue
            name = "person" if c == PERSON else ITEM_CLASSES[c]
            out.append(Detection(c, name, float(s), (int(x1), int(y1), int(x2), int(y2))))
        return out


class SeatMap:
    """좌석 다각형을 마스크로 만들어 두고, 검출 박스가 어느 좌석에 속하는지 계산한다."""

    def __init__(self, seats: list[dict], frame_shape: tuple[int, int],
                 person_overlap: float = 0.25, item_overlap: float = 0.5):
        h, w = frame_shape[:2]
        self.ids = [s["id"] for s in seats]
        self.polys = [np.array(s["polygon"], dtype=np.int32) for s in seats]
        self.masks = []
        for poly in self.polys:
            m = np.zeros((h, w), np.uint8)
            cv2.fillPoly(m, [poly], 1)
            self.masks.append(m)
        self.person_overlap, self.item_overlap = person_overlap, item_overlap

    def assign(self, dets: list[Detection]) -> dict[str, dict]:
        """좌석별 {'person': bool, 'items': [Detection...]}.

        박스 면적 중 좌석 영역에 들어간 비율이 기준 이상이면 그 좌석 소속.
        여러 좌석에 걸치면 가장 많이 겹친 좌석 하나에만 배정한다.
        사람은 좌석 밖으로 많이 튀어나오므로 기준을 낮게(0.25) 둔다.
        """
        result = {sid: {"person": False, "items": [], "persons": []} for sid in self.ids}
        for d in dets:
            x1, y1, x2, y2 = d.box
            area = max(1, (x2 - x1) * (y2 - y1))
            ratios = [m[max(0, y1):y2, max(0, x1):x2].sum() / area for m in self.masks]
            if not ratios:
                break
            best = int(np.argmax(ratios))
            need = self.person_overlap if d.is_person else self.item_overlap
            if ratios[best] < need:
                continue
            seat = result[self.ids[best]]
            if d.is_person:
                seat["person"] = True
                seat["persons"].append(d)
            else:
                seat["items"].append(d)
        return result
