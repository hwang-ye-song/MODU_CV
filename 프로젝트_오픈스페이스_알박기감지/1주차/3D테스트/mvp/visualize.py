"""영상 위에 감지 영역과 3색 표시등을 그린다 (한글은 PIL + 맑은 고딕)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from room_logic import RoomState, RoomTracker

RGB = {"red": (230, 57, 70), "yellow": (250, 190, 40), "green": (46, 196, 102), "off": (70, 70, 70)}
_FONTS = ["C:/Windows/Fonts/malgunbd.ttf", "C:/Windows/Fonts/malgun.ttf",
          "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"]


def _font(size: int):
    for p in _FONTS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def fmt(sec: float) -> str:
    m, s = divmod(int(round(sec)), 60)
    return f"{m:02d}:{s:02d}"


def blur(frame: np.ndarray, boxes) -> None:
    """사람 영역 모자이크 (화면에 개인이 드러나지 않게)."""
    h, w = frame.shape[:2]
    for b in boxes:
        x1, y1, x2, y2 = b.as_int()
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        roi = frame[y1:y2, x1:x2]
        small = cv2.resize(roi, (max(1, (x2 - x1) // 14), max(1, (y2 - y1) // 14)))
        frame[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)


def draw(frame, zone, persons, inside, tracker: RoomTracker, now: float, scale: float,
         table=None, privacy: bool = True) -> np.ndarray:
    out = frame.copy()
    if privacy:
        blur(out, persons)
    color = RGB[tracker.state.light]
    bgr = color[::-1]
    if table is not None:
        x1, y1, x2, y2 = table.as_int()
        cv2.rectangle(out, (x1, y1), (x2, y2), (200, 200, 200), 1)
    if zone is not None:
        x1, y1, x2, y2 = zone.as_int()
        ov = out.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), bgr, -1)
        out = cv2.addWeighted(ov, 0.18, out, 0.82, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), bgr, 3)
    for p in persons:
        x1, y1, x2, y2 = p.as_int()
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255) if p in inside else (120, 120, 120), 2)

    h, w = out.shape[:2]
    fs = max(16, w // 45)
    img = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(img)

    # 오른쪽 위 신호등
    r = fs
    pw, ph = r * 2 + 24, r * 6 + 40
    px, py = w - pw - 14, 14
    dr.rounded_rectangle((px, py, px + pw, py + ph), radius=14, fill=(25, 25, 25))
    for i, name in enumerate(["red", "yellow", "green"]):
        cx, cy = px + pw // 2, py + 12 + r + i * (2 * r + 8)
        on = tracker.state.light == name
        dr.ellipse((cx - r, cy - r, cx + r, cy + r), fill=RGB[name] if on else RGB["off"])

    # 왼쪽 위 상태 안내
    f, fsm = _font(fs), _font(int(fs * 0.75))
    lines = [(f"{tracker.room_id}  {tracker.state.value}", f, color)]
    if tracker.state is RoomState.AWAY:
        lines.append((f"자리 비움 {fmt(tracker.away_sec(now) * scale)}  ·  초록불까지 {fmt(tracker.left_sec(now) * scale)}",
                      fsm, (255, 255, 255)))
    elif tracker.state is RoomState.FREE:
        lines.append((f"30분 이상 비어 있음 ({fmt(tracker.away_sec(now) * scale)})  ·  짐은 바구니로 옮기고 사용",
                      fsm, (255, 255, 255)))
    else:
        lines.append((f"감지 영역 안 사람 {len(inside)}명", fsm, (255, 255, 255)))
    full = zone is not None and zone.as_int() == (0, 0, w, h)
    if table is None and full:
        lines.append(("테이블 미검출 → 화면 전체를 감지 영역으로 사용", fsm, (255, 200, 120)))
    y = 12
    for text, ft, col in lines:
        tb = dr.textbbox((14, y), text, font=ft)
        dr.rectangle((tb[0] - 6, tb[1] - 4, tb[2] + 6, tb[3] + 4), fill=(0, 0, 0))
        dr.text((14, y), text, font=ft, fill=col)
        y = tb[3] + 10
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
