"""틀린그림찾기 도안에서 다른 곳을 찾아 표시한다 (1단계: 고전 영상처리).

    python spotdiff/src/find_diff.py --image spotdiff/data/raw/Spotthedifference_1.jpg
    python spotdiff/src/find_diff.py --all --topk 10

흐름: 그림 두 장 분리 → 정렬 → 차이 계산 → 영역 묶기 → 표시
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "results"


def find_panels(img, margin=6):
    """그림 판 두 개를 찾아 잘라낸다. 제목 글자나 페이지 여백이 섞이지 않게 하는 것이 목적이다.

    선(잉크)을 굵게 부풀려 점선 테두리를 이어 붙인 뒤, 페이지의 8% 이상을 차지하는
    비슷한 크기의 사각형 두 개를 판으로 본다.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ink = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)[1]
    closed = cv2.dilate(ink, np.ones((7, 7), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    page = img.shape[0] * img.shape[1]
    rects = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= page * 0.08 and w > 40 and h > 40:
            rects.append((x, y, w, h))
    rects.sort(key=lambda r: -r[2] * r[3])
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            (x1, y1, w1, h1), (x2, y2, w2, h2) = rects[i], rects[j]
            if min(w1 * h1, w2 * h2) / max(w1 * h1, w2 * h2) < 0.75:   # 크기가 비슷해야 한 쌍
                continue
            first, second = sorted([rects[i], rects[j]], key=lambda r: (r[1], r[0]))
            how = "좌우" if abs(first[0] - second[0]) > abs(first[1] - second[1]) else "상하"
            crops = [img[y + margin:y + h - margin, x + margin:x + w - margin] for x, y, w, h in (first, second)]
            if all(c.size for c in crops):
                return crops[0], crops[1], how
    return None


def split_pair(img):
    """도안 한 장을 그림 두 개로 나눈다. 가운데 여백이 더 뚜렷한 방향으로 자른다."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ink = (gray < 200).astype(np.float32)          # 선이 그려진 픽셀
    h, w = ink.shape

    def best_gap(profile, length):
        """가운데 1/3 구간에서 잉크가 가장 적은 위치와 그 값"""
        lo, hi = int(length * 0.35), int(length * 0.65)
        band = profile[lo:hi]
        i = int(band.argmin())
        return lo + i, float(band.min())

    x, vx = best_gap(ink.sum(axis=0) / h, w)       # 세로선 방향(좌우 분리)
    y, vy = best_gap(ink.sum(axis=1) / w, h)       # 가로선 방향(상하 분리)
    if vx <= vy:
        return img[:, :x], img[:, x:], "좌우"
    return img[:y], img[y:], "상하"


def _score(a, b):
    """두 그림이 얼마나 안 맞는지 (작을수록 잘 맞음)"""
    ga, gb = (cv2.GaussianBlur(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), (5, 5), 0) for x in (a, b))
    return float(cv2.absdiff(ga, gb).mean())


def _warp(b, M, shape):
    return cv2.warpAffine(b, M, (shape[1], shape[0]), borderValue=(255, 255, 255))


def align(a, b):
    """b를 a에 겹치도록 맞춘다.

    세 가지 방법(크기만 맞추기 / 특징점 매칭 / 위상 상관으로 평행이동)을 모두 해 보고
    가장 잘 맞는 것을 고른다. 도안마다 인쇄·스캔 상태가 달라서 한 방법만으로는 부족하다.
    """
    b = cv2.resize(b, (a.shape[1], a.shape[0]))
    ga, gb = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (a, b))
    cands = [("크기만", b)]

    orb = cv2.ORB_create(4000)
    ka, da = orb.detectAndCompute(ga, None)
    kb, db = orb.detectAndCompute(gb, None)
    if da is not None and db is not None and len(ka) >= 10 and len(kb) >= 10:
        matches = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(db, da), key=lambda m: m.distance)[:300]
        if len(matches) >= 12:
            src = np.float32([kb[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst = np.float32([ka[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3)
            if M is not None:
                cands.append(("특징점", _warp(b, M, a.shape)))

    (dx, dy), _ = cv2.phaseCorrelate(np.float32(ga) / 255, np.float32(gb) / 255)
    cands.append(("평행이동", _warp(b, np.float32([[1, 0, -dx], [0, 1, -dy]]), a.shape)))

    name, best = min(cands, key=lambda c: _score(a, c[1]))
    return best, name, _score(a, best)


def diff_boxes(a, b, topk=10, min_area=60):
    """두 그림의 차이 영역을 큰 순서대로 topk 개 돌려준다."""
    ga, gb = (cv2.GaussianBlur(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), (5, 5), 0) for x in (a, b))
    d = cv2.absdiff(ga, gb)
    _, mask = cv2.threshold(d, 40, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))   # 끊긴 조각 잇기
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))    # 점 노이즈 제거
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if w * h >= min_area:
            boxes.append((x, y, w, h, area))
    boxes.sort(key=lambda b_: -b_[4])
    return [b_[:4] for b_ in boxes[:topk]], mask


def draw(a, b, boxes):
    """두 그림에 같은 위치로 동그라미를 친다."""
    out = []
    for im in (a.copy(), b.copy()):
        for i, (x, y, w, h) in enumerate(boxes, 1):
            cx, cy = x + w // 2, y + h // 2
            r = max(w, h) // 2 + 10
            cv2.circle(im, (cx, cy), r, (0, 0, 255), 2)
            cv2.putText(im, str(i), (cx - r, cy - r - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        out.append(im)
    h = max(o.shape[0] for o in out)
    out = [cv2.copyMakeBorder(o, 0, h - o.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for o in out]
    return cv2.hconcat(out)


def run(path: Path, topk: int, save_mask: bool = False, variant: str = "full"):
    """variant: 보고서에 단계별 결과를 남기기 위한 옵션
    - naive : 페이지를 반으로 자르기만 (1차 시도)
    - panel : 그림 판만 찾아서 자르기 (정렬 없음)
    - full  : 판 찾기 + 정렬 + 테두리 여백 제거 (현재 방식)
    """
    img = cv2.imread(str(path))
    if variant == "naive":
        a, b, how = split_pair(img)
        how += " (페이지 반 자르기)"
    else:
        found = find_panels(img)
        a, b, how = found if found else (*split_pair(img)[:2], split_pair(img)[2] + " (판 인식 실패)")

    if variant == "full":
        b, how_align, score = align(a, b)
        m = int(min(a.shape[:2]) * 0.03)   # 테두리 선이 차이로 잡히지 않게 가장자리를 조금 잘라낸다
        a, b = a[m:-m, m:-m], b[m:-m, m:-m]
    else:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
        how_align, score = "없음", _score(a, b)

    boxes, mask = diff_boxes(a, b, topk=topk)
    out_dir = OUT if variant == "full" else OUT / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{path.stem}_found.jpg"), draw(a, b, boxes))
    if save_mask:
        cv2.imwrite(str(out_dir / f"{path.stem}_mask.jpg"), mask)
    print(f"[{variant}] {path.name}: {how} | 정렬 {how_align}(어긋남 {score:.1f}) | 찾은 차이 {len(boxes)}개")
    return boxes


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--mask", action="store_true", help="차이 마스크도 저장")
    ap.add_argument("--variant", default="full", choices=["naive", "panel", "full"],
                    help="보고서용: 단계별 결과 비교")
    args = ap.parse_args()
    targets = sorted(p for p in RAW.iterdir() if p.suffix.lower() in {".jpg", ".png"}) if args.all else [Path(args.image)]
    for p in targets:
        run(p, args.topk, args.mask, args.variant)
