"""COCO 도시 사진으로 틀린그림찾기 데이터셋을 만든다 (문제 사진 A/B + 정답).

    python spotdiff/src/make_city_dataset.py --min 5 --max 7 --hard 10

자연스럽게 보이도록 "작게, 여러 개" 바꾼다.
- 사라짐  : 사진의 3% 이하인 작은 물체만 LaMa 인페인팅으로 지운다 (큰 구멍은 티가 난다)
- 색 바뀜 : 물체 모양은 그대로 두고 색(색상)만 돌린다
- 좌우 반전: 물체를 제자리에서 뒤집는다
- 추가/이동: 같은 사진 안에서만, 바닥 색이 비슷한 자리에만 놓는다 (다른 사진 것은 붙이지 않는다)
물체 모양은 YOLOv8 분할 모델(yolov8n-seg)로 따고, 못 따면 GrabCut 을 쓴다.

결과: spotdiff/dataset_city/
    A/, B/          문제 사진 한 쌍
    answer/         정답 표시 이미지 (A | B)
    answers.json    정답 좌표
    정답표.md        사람이 보는 정답표
    answer_sheet.jpg
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
COCO = ROOT / "data" / "coco128"
OUT = ROOT / "dataset_city"

CITY_CLASSES = {2, 3, 5, 7, 9, 10, 11, 12}       # car motorcycle bus truck traffic-light hydrant stop-sign meter
KIND_KO = {"remove": "사라짐", "recolor": "색 바뀜", "flip": "좌우 반전", "copy": "추가", "move": "이동"}
FLIPPABLE = {2, 3, 5, 7, 1, 11, 13}               # 뒤집어도 말이 되는 물체 (차, 오토바이, 버스, 트럭, 자전거, 표지판, 벤치)


# ---------------------------------------------------------------- 도구
def load_boxes(img_path, w, h):
    lbl = COCO / "labels" / "train2017" / (img_path.stem + ".txt")
    out = []
    for line in lbl.read_text().splitlines() if lbl.exists() else []:
        c, cx, cy, bw, bh = map(float, line.split()[:5])
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        out.append((int(c), max(0, x1), max(0, y1), min(w, x1 + int(bw * w)), min(h, y1 + int(bh * h))))
    return out


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0


def grabcut_mask(img, box):
    x1, y1, x2, y2 = box
    mask = np.zeros(img.shape[:2], np.uint8)
    try:
        bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(img, mask, (x1, y1, x2 - x1, y2 - y1), bg, fg, 4, cv2.GC_INIT_WITH_RECT)
        m = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        return None
    ratio = m[y1:y2, x1:x2].mean() / 255
    return m if 0.2 <= ratio <= 0.95 else None


class Editor:
    def __init__(self):
        from ultralytics import YOLO
        from simple_lama_inpainting import SimpleLama
        self.seg = YOLO("yolov8n-seg.pt")
        self.lama = SimpleLama()
        self.names = self.seg.names

    def masks_for(self, img, boxes):
        """정답 박스마다 물체 모양 마스크를 만든다. 분할 모델 결과와 겹치면 그것을, 아니면 GrabCut."""
        r = self.seg.predict(img, conf=0.25, device=0, retina_masks=True, verbose=False)[0]
        seg_boxes = r.boxes.xyxy.cpu().numpy().astype(int) if r.masks is not None else []
        seg_masks = (r.masks.data.cpu().numpy() * 255).astype(np.uint8) if r.masks is not None else []
        out = []
        for c, x1, y1, x2, y2 in boxes:
            m = None
            for sb, sm in zip(seg_boxes, seg_masks):
                if iou((x1, y1, x2, y2), tuple(sb)) > 0.5:
                    m = sm.copy()
                    m[:y1, :] = 0; m[y2:, :] = 0; m[:, :x1] = 0; m[:, x2:] = 0
                    break
            if m is None:
                m = grabcut_mask(img, (x1, y1, x2, y2))
            out.append(m)
        return out

    def remove(self, img, mask, grow=7):
        h, w = img.shape[:2]
        m = cv2.dilate(mask, np.ones((grow, grow), np.uint8), iterations=2)
        res = self.lama(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), Image.fromarray(m))
        return cv2.cvtColor(np.array(res), cv2.COLOR_RGB2BGR)[:h, :w]


def alpha_paste(img, patch, mask, xy):
    out = img.copy()
    ph, pw = patch.shape[:2]
    x, y = xy
    a = cv2.GaussianBlur(mask.astype(np.float32) / 255, (5, 5), 0)[..., None]
    roi = out[y:y + ph, x:x + pw].astype(np.float32)
    out[y:y + ph, x:x + pw] = (a * patch + (1 - a) * roi).astype(np.uint8)
    return out


def recolor(img, mask, rng):
    """물체 색상(hue)만 돌린다. 색이 거의 없는(회색) 물체는 밝기를 바꾼다."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
    sel = mask > 0
    sat = hsv[..., 1][sel].mean()
    if sat > 60:
        hsv[..., 0][sel] = (hsv[..., 0][sel] + rng.choice([45, 60, 90, 120, 135])) % 180
        how = "색상"
    else:
        f = rng.choice([0.55, 1.5])
        hsv[..., 2][sel] = np.clip(hsv[..., 2][sel] * f, 0, 255)
        how = "밝기"
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR), how


def flip_in_place(img, mask, box):
    x1, y1, x2, y2 = box
    patch = cv2.flip(img[y1:y2, x1:x2], 1)
    m = cv2.flip(mask[y1:y2, x1:x2], 1)
    return alpha_paste(img, patch, m, (x1, y1))


def ring_color(img, box, pad=8):
    """박스 주변 띠의 평균 색 (바닥 색이 비슷한지 볼 때 쓴다)"""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    outer = img[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)].reshape(-1, 3).astype(np.float32)
    inner = img[y1:y2, x1:x2].reshape(-1, 3).astype(np.float32)
    s_o, s_i = outer.sum(0), inner.sum(0)
    n = len(outer) - len(inner)
    return (s_o - s_i) / max(n, 1)


def find_spot(img, box, taken, rng, max_shift=None, tries=80):
    """같은 높이에서 옆으로 옮길 자리. 다른 물체와 겹치지 않고 주변 바닥 색이 원래 자리와 비슷해야 한다."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw = x2 - x1
    ref = ring_color(img, box)
    max_shift = max_shift or w
    for _ in range(tries):
        dx = rng.randint(int(bw * 1.2), int(max(bw * 1.2 + 1, max_shift)))
        nx = x1 + dx * rng.choice([-1, 1])
        if nx < 3 or nx + bw > w - 3:
            continue
        cand = (nx, y1, nx + bw, y2)
        if any(iou(cand, t) > 0 for t in taken):
            continue
        if np.abs(ring_color(img, cand) - ref).mean() < 22:
            return cand
    return None


# ---------------------------------------------------------------- 문제 만들기
def make_pair(img_path, ed, rng, n_min, n_max, hard):
    a = cv2.imread(str(img_path))
    h, w = a.shape[:2]
    objs = load_boxes(img_path, w, h)
    if not any(o[0] in CITY_CLASSES and (o[3] - o[1]) * (o[4] - o[2]) >= w * h * 0.01 for o in objs):
        return None                                        # 도시 장면이 아니다

    cands = []
    for o in objs:
        c, x1, y1, x2, y2 = o
        area = (x2 - x1) * (y2 - y1) / (w * h)
        if not (0.0006 <= area <= 0.08) or (x2 - x1) < 10 or (y2 - y1) < 10:
            continue
        if x1 < 2 or y1 < 2 or x2 > w - 2 or y2 > h - 2:
            continue
        if any(iou(o[1:], p[1:]) > 0.15 for p in objs if p is not o):
            continue
        cands.append(o)
    target = rng.randint(n_min, n_max) if not hard else rng.randint(n_max, hard)
    if len(cands) < n_min:
        return None
    rng.shuffle(cands)
    masks = ed.masks_for(a, cands)

    b = a.copy()
    taken = [o[1:] for o in objs]
    diffs = []
    for o, m in zip(cands, masks):
        if len(diffs) >= target:
            break
        c, x1, y1, x2, y2 = o
        box = (x1, y1, x2, y2)
        area = (x2 - x1) * (y2 - y1) / (w * h)
        name = ed.names[c]
        if m is None or m[y1:y2, x1:x2].mean() < 255 * 0.15:
            continue
        options = []
        if area <= 0.03:
            options += ["remove"] * 3
        options += ["recolor"] * 3
        if c in FLIPPABLE and (x2 - x1) > 24:
            options += ["flip"]
        if area <= 0.03:
            options += ["copy", "move"]
        kind = rng.choice(options)

        if kind == "remove":
            b = ed.remove(b, m)
            diffs.append({"kind": kind, "class": name, "box_a": box, "box_b": None})
        elif kind == "recolor":
            b, how = recolor(b, m, rng)
            diffs.append({"kind": kind, "class": name, "box_a": box, "box_b": box, "note": how})
        elif kind == "flip":
            b = flip_in_place(b, m, box)
            diffs.append({"kind": kind, "class": name, "box_a": box, "box_b": box})
        else:
            spot = find_spot(a, box, taken, rng, max_shift=int(w * 0.35) if kind == "move" else None)
            if spot is None:
                b, how = recolor(b, m, rng)                # 자리가 없으면 색만 바꾼다
                diffs.append({"kind": "recolor", "class": name, "box_a": box, "box_b": box, "note": how})
                continue
            if kind == "move":
                b = ed.remove(b, m)
            b = alpha_paste(b, a[y1:y2, x1:x2], m[y1:y2, x1:x2], spot[:2])
            taken.append(spot)
            diffs.append({"kind": kind, "class": name, "box_a": box if kind == "move" else None, "box_b": spot})
    if len(diffs) < n_min:
        return None
    return a, b, diffs


def draw_answer(a, b, diffs):
    va, vb = a.copy(), b.copy()
    for i, d in enumerate(diffs, 1):
        for im, box, thick in ((va, d["box_a"], 2), (vb, d["box_b"], 2),
                               (vb, d["box_a"] if d["kind"] == "remove" else None, 1)):
            if box is None:
                continue
            x1, y1, x2, y2 = box
            cv2.rectangle(im, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (0, 0, 255), thick)
            cv2.putText(im, str(i), (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    gap = np.full((a.shape[0], 12, 3), 255, np.uint8)
    return cv2.hconcat([va, gap, vb])


def where(box, w, h):
    cx, cy = (box[0] + box[2]) / 2 / w, (box[1] + box[3]) / 2 / h
    return ["위", "가운데", "아래"][min(2, int(cy * 3))] + " " + ["왼쪽", "가운데", "오른쪽"][min(2, int(cx * 3))]


def describe(d, w, h):
    k = d["kind"]
    if k == "remove":
        return f"A의 {where(d['box_a'], w, h)}에 있던 것이 B에서 없어짐"
    if k == "move":
        return f"A의 {where(d['box_a'], w, h)} → B의 {where(d['box_b'], w, h)}"
    if k == "copy":
        return f"B의 {where(d['box_b'], w, h)}에 하나 더 생김"
    if k == "recolor":
        return f"{where(d['box_a'], w, h)} — {d.get('note', '색')}이 바뀜"
    return f"{where(d['box_a'], w, h)} — 좌우가 뒤집힘"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=int, default=5)
    ap.add_argument("--max", type=int, default=7)
    ap.add_argument("--hard", type=int, default=10, help="어려운 문제(3문제마다 1개)의 최대 차이 수")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    ed = Editor()

    for sub in ("A", "B", "answer"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    answers, thumbs = {}, []
    for p in sorted((COCO / "images" / "train2017").glob("*.jpg")):
        hard = args.hard if (len(answers) % 3 == 2) else 0
        res = make_pair(p, ed, rng, args.min, args.max, hard)
        if res is None:
            continue
        a, b, diffs = res
        qid = f"city_{len(answers) + 1:02d}"
        cv2.imwrite(str(OUT / "A" / f"{qid}.jpg"), a, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(OUT / "B" / f"{qid}.jpg"), b, [cv2.IMWRITE_JPEG_QUALITY, 95])
        ans = draw_answer(a, b, diffs)
        cv2.imwrite(str(OUT / "answer" / f"{qid}.jpg"), ans)
        answers[qid] = {"source": p.name, "size": [a.shape[1], a.shape[0]], "hard": bool(hard),
                        "differences": [{"no": i, **d} for i, d in enumerate(diffs, 1)]}
        t = cv2.resize(ans, (640, int(ans.shape[0] * 640 / ans.shape[1])))
        cv2.putText(t, f"{qid} ({len(diffs)})", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        thumbs.append(t)
        print(f"{qid}: {p.name} | 차이 {len(diffs)}개 " + ("(어려움)" if hard else ""))

    (OUT / "answers.json").write_text(json.dumps(answers, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = ["# 도시 틀린그림찾기 정답표", "",
             "사진 A(원본)와 B(바뀐 사진)를 비교했을 때의 정답이다. `answer/` 폴더 이미지에 같은 번호로 표시돼 있다.",
             "위치는 사진을 가로·세로 3칸씩 나눴을 때 어느 칸인지로 적었다.", ""]
    for qid, v in answers.items():
        w, h = v["size"]
        lines += [f"## {qid} (원본 {v['source']}, 차이 {len(v['differences'])}개" + (", 어려움)" if v["hard"] else ")"),
                  "", "| 번호 | 종류 | 물체 | 위치 |", "|---|---|---|---|"]
        lines += [f"| {d['no']} | {KIND_KO[d['kind']]} | {d['class']} | {describe(d, w, h)} |" for d in v["differences"]]
        lines.append("")
    (OUT / "정답표.md").write_text("\n".join(lines), encoding="utf-8")

    rows = []
    for i in range(0, len(thumbs), 2):
        pair = thumbs[i:i + 2]
        hmax = max(t.shape[0] for t in pair)
        pair = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0] + 8, 4, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in pair]
        if len(pair) == 1:
            pair.append(np.full_like(pair[0], 255))
        rows.append(cv2.hconcat(pair))
    if rows:
        cv2.imwrite(str(OUT / "answer_sheet.jpg"), cv2.vconcat(rows))
    kinds = [d["kind"] for v in answers.values() for d in v["differences"]]
    print(f"문제 {len(answers)}개 | 차이 {len(kinds)}개 | " + ", ".join(f"{KIND_KO[k]} {kinds.count(k)}" for k in KIND_KO))


if __name__ == "__main__":
    main()
