"""좌석 상태를 영상 위에 그린다 (한글은 PIL + 맑은 고딕으로 출력)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from seat_logic import SeatState, SeatTracker

# BGR
COLORS = {
    SeatState.EMPTY: (80, 200, 80),       # 초록: 이용 가능
    SeatState.OCCUPIED: (220, 140, 40),   # 파랑: 사용중
    SeatState.AWAY: (40, 200, 230),       # 노랑: 자리비움
    SeatState.WARNING: (30, 130, 255),    # 주황: 반납예정
    SeatState.RECLAIM: (60, 60, 230),     # 빨강: 반납대상(이용 가능)
}

_FONT_CANDIDATES = [
    "C:/Windows/Fonts/malgunbd.ttf", "C:/Windows/Fonts/malgun.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]


def _font(size: int):
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def fmt_sec(sec: float) -> str:
    sec = int(round(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def blur_boxes(frame: np.ndarray, boxes) -> None:
    """사람 영역을 모자이크해 화면에 개인이 드러나지 않게 한다."""
    h, w = frame.shape[:2]
    for x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        roi = frame[y1:y2, x1:x2]
        small = cv2.resize(roi, (max(1, (x2 - x1) // 16), max(1, (y2 - y1) // 16)))
        frame[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)


def draw(frame: np.ndarray, seat_map, trackers: dict[str, SeatTracker], assigned: dict,
         now: float, time_scale: float = 1.0, privacy: bool = True,
         clock_text: str = "") -> np.ndarray:
    out = frame.copy()
    if privacy:
        blur_boxes(out, [d.box for s in assigned.values() for d in s["persons"]])

    overlay = out.copy()
    for sid, poly in zip(seat_map.ids, seat_map.polys):
        cv2.fillPoly(overlay, [poly], COLORS[trackers[sid].state])
    out = cv2.addWeighted(overlay, 0.28, out, 0.72, 0)
    for sid, poly in zip(seat_map.ids, seat_map.polys):
        cv2.polylines(out, [poly], True, COLORS[trackers[sid].state], 2)
    for s in assigned.values():
        for d in s["items"]:
            x1, y1, x2, y2 = d.box
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 1)

    # 한글 텍스트
    h, w = out.shape[:2]
    fs = max(14, w // 60)
    font, small = _font(fs), _font(int(fs * 0.8))
    img = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(img)

    # 상단 요약 바 (좌석 라벨이 가려지지 않도록 먼저 그리고 라벨을 그 아래에 둔다)
    states = [t.state for t in trackers.values()]
    avail = sum(s.available for s in states)
    summary = (f"이용 가능 {avail}/{len(states)}석  |  "
               + "  ".join(f"{st.value} {states.count(st)}" for st in SeatState))
    if clock_text:
        summary += f"  |  {clock_text}"
    tb = dr.textbbox((8, 6), summary, font=font)
    dr.rectangle((0, 0, w, tb[3] + 6), fill=(20, 20, 20))
    dr.text((8, 6), summary, font=font, fill=(255, 255, 255))
    bar_h = tb[3] + 6

    for sid, poly in zip(seat_map.ids, seat_map.polys):
        tr = trackers[sid]
        x, y = poly[:, 0].min() + 4, max(poly[:, 1].min(), bar_h) + 4
        lines = [f"{sid} {tr.state.value}"]
        if tr.session_start is not None:
            lines.append(f"재석 {tr.presence_ratio(now) * 100:.0f}% / "
                         f"{fmt_sec(tr.observed_sec(now) * time_scale)}")
        if (left := tr.grace_left(now)) is not None:
            lines.append(f"유예 {fmt_sec(left * time_scale)}")
        items = assigned[sid]["items"]
        if items:
            lines.append(", ".join(sorted({d.name for d in items})))
        b, g, r = COLORS[tr.state]
        for i, text in enumerate(lines):
            f = font if i == 0 else small
            ty = y + i * int(fs * 1.15)
            tb = dr.textbbox((x, ty), text, font=f)
            dr.rectangle((tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2), fill=(0, 0, 0))
            dr.text((x, ty), text, font=f, fill=(r, g, b) if i == 0 else (255, 255, 255))

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
