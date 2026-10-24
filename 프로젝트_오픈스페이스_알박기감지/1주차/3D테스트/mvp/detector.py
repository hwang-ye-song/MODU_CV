"""사전학습 YOLO(COCO)로 테이블과 사람을 검출하고, 테이블 박스를 넓힌 감지 영역 안에 사람이 있는지 판단한다.

감지 영역을 손으로 그리지 않는다.
  1) 'dining table'을 검출해 가장 큰 박스(= 카메라와 가장 가까운 이 룸의 테이블)를 고르고
  2) 의자에 앉은 사람까지 들어오도록 가로·세로를 30% 넓힌 뒤
  3) 사람(person) 박스가 이 영역과 충분히 겹치면 '재석'으로 본다.
테이블이 잠깐 안 잡히거나 사람에게 가려져도 영역이 흔들리지 않도록, 최근 영역을 이동 평균으로 유지한다.
테이블을 아직 못 찾았으면 화면 전체를 감지 영역으로 쓰고, 영역을 직접 정하고 싶으면 --zone 으로 지정한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

PERSON, TABLE = 0, 60   # COCO 클래스 번호


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = 1.0

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def expand(self, ratio: float, w: int, h: int) -> "Box":
        """가로·세로를 ratio만큼 넓힌다 (0.3 → 각각 30% 넓어짐, 양쪽에 15%씩)."""
        dx, dy = (self.x2 - self.x1) * ratio / 2, (self.y2 - self.y1) * ratio / 2
        return Box(max(0, self.x1 - dx), max(0, self.y1 - dy),
                   min(w, self.x2 + dx), min(h, self.y2 + dy), self.conf)

    def overlap_of(self, other: "Box") -> float:
        """other 면적 중 self 안에 들어온 비율."""
        ix = max(0.0, min(self.x2, other.x2) - max(self.x1, other.x1))
        iy = max(0.0, min(self.y2, other.y2) - max(self.y1, other.y1))
        return ix * iy / other.area if other.area else 0.0

    def as_int(self) -> tuple[int, int, int, int]:
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)


class RoomDetector:
    def __init__(self, weights: str = "yolo11s.pt", imgsz: int = 960, person_conf: float = 0.35,
                 table_conf: float = 0.25, expand: float = 0.30, overlap: float = 0.25,
                 fixed_zone: Box | None = None, smooth: float = 0.2, device: str | None = None):
        self.model = YOLO(weights)
        self.imgsz, self.device = imgsz, device
        self.person_conf, self.table_conf = person_conf, table_conf
        self.expand, self.overlap, self.smooth = expand, overlap, smooth
        self.fixed_zone = fixed_zone
        self.table: Box | None = None   # 이동 평균한 테이블 박스

    def detect(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        res = self.model.predict(frame, imgsz=self.imgsz, device=self.device, verbose=False,
                                 classes=[PERSON, TABLE], conf=min(self.person_conf, self.table_conf))[0]
        persons, tables = [], []
        for (x1, y1, x2, y2), c, s in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(),
                                          res.boxes.conf.tolist()):
            b = Box(x1, y1, x2, y2, s)
            if int(c) == PERSON and s >= self.person_conf:
                persons.append(b)
            elif int(c) == TABLE and s >= self.table_conf:
                tables.append(b)

        if self.fixed_zone is not None:
            zone = self.fixed_zone
        else:
            if tables:
                biggest = max(tables, key=lambda b: b.area)
                self.table = biggest if self.table is None else self._blend(self.table, biggest)
            # 테이블을 아직 못 찾았으면 화면 전체를 쓴다 (카메라가 룸 안에 있으므로 화면 = 룸)
            zone = self.table.expand(self.expand, w, h) if self.table else Box(0, 0, w, h, 0.0)

        inside = [p for p in persons if zone is not None and zone.overlap_of(p) >= self.overlap]
        return zone, persons, inside

    def _blend(self, old: Box, new: Box) -> Box:
        a = self.smooth
        return Box(*(o * (1 - a) + n * a for o, n in
                     zip((old.x1, old.y1, old.x2, old.y2), (new.x1, new.y1, new.x2, new.y2))), new.conf)
