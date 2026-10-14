"""틀린그림찾기 문제 이미지(두 판이 한 장에 붙은 것)를 A·B 그림으로 나눠 저장한다.

    python spotdiff/src/prepare_pairs.py            # original/ 의 모든 이미지
    python spotdiff/src/prepare_pairs.py cafe_01    # 하나만
    python spotdiff/src/prepare_pairs.py --no-watermark   # Gemini가 아닌 그림

순서
1. 두 판의 경계 찾기: 가운데 근처에서 위아래(또는 좌우)로 균일한 선
2. 글자 띠 잘라내기: 제목("틀린 그림 찾기!"), 판 이름("장면 A: 원본"), 설명 문구는
   한 가지 배경색에 글자만 조금 있는 줄이라서, 그런 줄을 빼고 그림 줄만 남긴다
0. 워터마크 지우기: Gemini 워터마크(원본 오른쪽 아래 반투명 별) 자리를 LaMa로 지운다
3. A와 B 맞추기: 두 판이 몇 픽셀 어긋나 있으면 맞춘 뒤 겹치는 부분만 남긴다

결과: dataset_ai/A/<이름>.png, dataset_ai/B/<이름>.png, build/prepare_check/<이름>.jpg (확인용)
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "dataset_ai"
CHECK = ROOT.parent / "build" / "prepare_check"


def load(path):
    return cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)


def flatness(img, axis):
    """줄마다 '가장 흔한 색에 가까운 픽셀 비율'. 글자 띠·테두리는 1에 가깝고 그림은 낮다."""
    q = (img // 16).astype(np.int32)
    key = q[..., 0] * 256 + q[..., 1] * 16 + q[..., 2]
    if axis == 1:                       # 가로줄(행)마다
        key = key
    else:                               # 세로줄(열)마다
        key = key.T
    out = np.empty(key.shape[0])
    for i, row in enumerate(key):
        out[i] = np.bincount(row).max() / row.size
    return out


def longest_run(mask):
    """True가 가장 길게 이어진 구간 [start, end)."""
    best, cur, start = (0, 0), 0, 0
    for i, v in enumerate(list(mask) + [False]):
        if v:
            if cur == 0:
                start = i
            cur += 1
        else:
            if cur > best[1] - best[0]:
                best = (start, i)
            cur = 0
    return best


def picture_box(panel):
    """판 안에서 글자 띠·테두리를 뺀 그림 영역 (x0, y0, x1, y1)."""
    h, w = panel.shape[:2]
    rows = cv2.blur(flatness(panel, 1).reshape(-1, 1), (1, 5)).ravel() < 0.55
    y0, y1 = longest_run(rows)
    cols = cv2.blur(flatness(panel[y0:y1], 0).reshape(-1, 1), (1, 5)).ravel() < 0.55
    x0, x1 = longest_run(cols)
    m = 3                                   # 경계의 흐린 줄·테두리 조금 더 깎기
    return x0 + m, y0 + m, x1 - m, y1 - m


def split(img):
    """두 판으로 나눈다. 좌우로 붙었는지 위아래로 붙었는지 둘 다 보고 더 뚜렷한 쪽을 쓴다."""
    h, w = img.shape[:2]
    best = None
    for axis in ("x", "y"):
        n = w if axis == "x" else h
        lo, hi = int(n * 0.35), int(n * 0.65)
        strip = img[int(h * .2):int(h * .8)] if axis == "x" else img[:, int(w * .2):int(w * .8)]
        g = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY).astype(np.float32)
        std = g.std(axis=0) if axis == "x" else g.std(axis=1)
        c = lo + int(np.argmin(std[lo:hi]))
        if best is None or std[c] < best[0]:
            best = (std[c], axis, c)
    _, axis, c = best
    # 경계선 두께만큼 양옆으로 넓히기
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    prof = g.std(axis=0) if axis == "x" else g.std(axis=1)
    a = c
    while a > 0 and prof[a - 1] < prof[c] + 8:
        a -= 1
    b = c
    while b < len(prof) - 1 and prof[b + 1] < prof[c] + 8:
        b += 1
    if axis == "x":
        return img[:, :a], img[:, b + 1:], axis, (b + 1, 0)
    return img[:a], img[b + 1:], axis, (0, b + 1)


def align(A, B):
    """B가 A보다 몇 픽셀 어긋났는지 구해서, 겹치는 부분만 잘라 같은 크기로 만든다."""
    h, w = min(A.shape[0], B.shape[0]), min(A.shape[1], B.shape[1])
    A, B = A[:h, :w], B[:h, :w]
    ga = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY).astype(np.float32)
    (dx, dy), _ = cv2.phaseCorrelate(ga, gb)
    dx, dy = int(round(dx)), int(round(dy))
    ax, bx = max(0, -dx), max(0, dx)
    ay, by = max(0, -dy), max(0, dy)
    ww, hh = w - abs(dx), h - abs(dy)
    return A[ay:ay + hh, ax:ax + ww], B[by:by + hh, bx:bx + ww], (dx, dy), (bx, by)


def star(size):
    """Gemini 워터마크 모양(가운데가 오목한 네 갈래 별) 틀."""
    t = np.linspace(-1, 1, size)
    x, y = np.meshgrid(t, t)
    return ((np.abs(x) ** 0.6 + np.abs(y) ** 0.6) <= 1).astype(np.float32)


def remove_watermark(img):
    """원본 한 장의 오른쪽 아래에 찍힌 반투명 별(Gemini 워터마크)을 LaMa로 지운다.

    시행착오
    1) 'A에는 없고 B에서만 밝아진 곳'으로 찾기 → 흰 운동화·잡지처럼 실제로 바뀐 물체를
       워터마크로 잡아 지워버림
    2) 별 모양 틀 맞추기 → 워터마크가 옅어서 밝은 바닥·글자 위에서는 엉뚱한 곳을 찾음
    3) 위치 공식 사용 (지금 방식) → 받은 5장 모두 별의 중심이 오른쪽·아래 가장자리에서
       0.115 × √(가로×세로) 만큼 안쪽에 있었다 (2000×1091 → 170px, 2000×1493 → 199px).
       크기도 같은 비율(약 0.05 × √(가로×세로))이라 그 자리를 별 모양으로 지운다.
    """
    H, W = img.shape[:2]
    g = np.sqrt(W * H)
    cx, cy, size = int(W - 0.115 * g), int(H - 0.115 * g), int(0.058 * g)
    mask = np.zeros((H, W), np.uint8)
    x, y = cx - size // 2, cy - size // 2
    mask[y:y + size, x:x + size] = star(size).astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=3)
    from simple_lama_inpainting import SimpleLama
    res = SimpleLama()(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), Image.fromarray(mask))
    out = cv2.cvtColor(np.array(res), cv2.COLOR_RGB2BGR)[:H, :W]
    return out, (cx, cy, size)


def prepare(path):
    img, wm = load(path), None
    if not NO_WATERMARK:
        img, wm = remove_watermark(img)
    P1, P2, axis, (ox, oy) = split(img)
    boxes, crops = [], []
    for P in (P1, P2):
        x0, y0, x1, y1 = picture_box(P)
        boxes.append(P[y0:y1, x0:x1])
        crops.append((x0, y0))
    A, B, shift, _ = align(*boxes)
    name = path.stem
    (DS / "A").mkdir(parents=True, exist_ok=True)
    (DS / "B").mkdir(parents=True, exist_ok=True)
    CHECK.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(DS / "A" / f"{name}.png"), A)
    cv2.imwrite(str(DS / "B" / f"{name}.png"), B)
    sep = np.full((A.shape[0], 12, 3), 255, np.uint8)
    cv2.imwrite(str(CHECK / f"{name}.jpg"), cv2.hconcat([A, sep, B]))
    print(f"{name}: 원본 {img.shape[1]}x{img.shape[0]} → 판 {A.shape[1]}x{A.shape[0]} "
          f"({'좌우' if axis == 'x' else '위아래'}), 어긋남 {shift}, "
          f"워터마크 {'지움 (중심 x, y, 크기) ' + str(wm) if wm else '안 지움'}")


NO_WATERMARK = "--no-watermark" in sys.argv     # Gemini가 아닌 그림이면 워터마크 지우기를 끈다


def main():
    names = [a for a in sys.argv[1:] if not a.startswith("--")]
    files = sorted(p for p in (DS / "original").iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
    for p in files:
        if not names or p.stem in names:
            prepare(p)


if __name__ == "__main__":
    main()
