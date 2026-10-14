"""YOLO로 틀린그림찾기: 두 그림의 물체를 찾아 비교하고, 정답과 맞춰 점수를 매긴다.

    python spotdiff/src/detect_diff.py            # 5문제 전부 + 점수
    python spotdiff/src/detect_diff.py --dump     # 후보마다 수치 출력 (기준값 정할 때)

순서
1. 탐지: YOLO11x로 A와 B의 물체를 전부 찾는다
2. 짝짓기: 위치가 겹치는(IoU ≥ 0.5) 박스끼리 A-B 짝을 짓는다
   - 짝이 없는 박스 → 한쪽에만 있는 물체 (사라짐·추가·바뀜 후보)
   - 짝이 있는데 클래스가 다름 → 다른 물체로 바뀜 후보
   - 짝도 있고 클래스도 같음 → 내용만 바뀌었을 수 있음 (색, 화면 등)
3. 확인: 모든 후보 자리에서 A와 B를 잘라 내용을 비교한다
   - 픽셀: 두 조각을 몇 픽셀 맞춘 뒤, 색이 크게 다른 픽셀의 비율
   - 색 분포: HSV 히스토그램 거리
   - (참고용) 모양: YOLO 분류 모델(yolo11n-cls) 특징 벡터 거리 — 차이를 거의 못 느껴서 판단에는 안 씀
   픽셀이나 색 분포가 기준보다 크면 '틀린 곳'. YOLO가 한쪽에서만 못 찾은 경우(내용은 같음)는 여기서 걸러진다.
4. 정리: 같은 종류가 붙어 있으면 묶고(자판기 병들), 겹치면 작은 박스만 남긴다
5. 채점: 정답 박스와 IoU ≥ 0.1 이거나, 정답 영역 안의 일부를 가리키면 찾은 것으로 본다
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "dataset_ai"
OUT = ROOT / "results_ai"

DET_CONF = 0.10        # 탐지 신뢰도 기준
PAIR_IOU = 0.5         # 짝으로 보는 겹침 기준
PIX_TH = 0.25          # 맞춘 뒤 색이 크게 다른 픽셀 비율 기준
COLOR_TH = 0.45        # 색 분포 거리 기준 (바타차리야 거리)
MAX_AREA = 0.25        # 그림의 25%보다 큰 박스는 틀린 곳으로 내지 않음 (식탁·사람 전체 등)
HIT_IOU = 0.1          # 채점: 정답과 이만큼 겹치면 찾은 것


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter + 1e-9)


class Finder:
    def __init__(self):
        self.det = YOLO("yolo11x.pt")
        self.cls = YOLO("yolo11n-cls.pt")

    def detect(self, img):
        r = self.det.predict(img, conf=DET_CONF, device=0, verbose=False)[0]
        return [dict(box=[int(v) for v in b], cls=r.names[int(c)], conf=float(p))
                for b, c, p in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]

    def pair(self, da, db):
        """IoU가 큰 순서대로 짝짓기 (클래스는 따지지 않음)."""
        cand = sorted(((iou(a["box"], b["box"]), i, j) for i, a in enumerate(da) for j, b in enumerate(db)), reverse=True)
        used_a, used_b, pairs = set(), set(), []
        for v, i, j in cand:
            if v < PAIR_IOU:
                break
            if i in used_a or j in used_b:
                continue
            used_a.add(i), used_b.add(j)
            pairs.append((i, j))
        return pairs, [i for i in range(len(da)) if i not in used_a], [j for j in range(len(db)) if j not in used_b]

    def compare(self, A, B, box):
        """같은 자리를 A·B에서 잘라 모양 거리와 색 거리를 잰다."""
        x1, y1, x2, y2 = box
        ca, cb = A[y1:y2, x1:x2], B[y1:y2, x1:x2]
        ea, eb = [e.cpu().numpy() for e in self.cls.embed([ca, cb], verbose=False)]
        shape = 1 - float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb)))
        hist = lambda c: cv2.normalize(cv2.calcHist([cv2.cvtColor(c, cv2.COLOR_BGR2HSV)], [0, 1], None, [18, 8], [0, 180, 0, 256]), None).flatten()
        color = float(cv2.compareHist(hist(ca), hist(cb), cv2.HISTCMP_BHATTACHARYYA))
        return shape, color, pixel_change(ca, cb)

    def find(self, A, B):
        da, db = self.detect(A), self.detect(B)
        pairs, only_a, only_b = self.pair(da, db)
        cands = []
        for i, j in pairs:
            a, b = da[i], db[j]
            box = [min(a["box"][0], b["box"][0]), min(a["box"][1], b["box"][1]),
                   max(a["box"][2], b["box"][2]), max(a["box"][3], b["box"][3])]
            why = "클래스 다름" if a["cls"] != b["cls"] else "내용 비교"
            cands.append(dict(box=box, label=f"{a['cls']}→{b['cls']}" if a["cls"] != b["cls"] else a["cls"], why=why))
        for i in only_a:
            cands.append(dict(box=da[i]["box"], label=da[i]["cls"], why="A에만 있음"))
        for j in only_b:
            cands.append(dict(box=db[j]["box"], label=db[j]["cls"], why="B에만 있음"))

        h, w = A.shape[:2]
        for c in cands:
            c["shape"], c["color"], c["pix"] = self.compare(A, B, c["box"])
            area = (c["box"][2] - c["box"][0]) * (c["box"][3] - c["box"][1]) / (w * h)
            c["diff"] = (c["pix"] > PIX_TH or c["color"] > COLOR_TH) and area <= MAX_AREA
        found = group([c for c in cands if c["diff"]])
        # 겹치는 것 정리: 작은 박스를 먼저 남기고, 그 박스와 많이 겹치는 큰 박스는 버린다
        found.sort(key=lambda c: (c["box"][2] - c["box"][0]) * (c["box"][3] - c["box"][1]))
        keep = []
        for c in found:
            if all(inside(k["box"], c["box"]) < 0.6 for k in keep):
                keep.append(c)
        return keep, cands, da, db


def group(found, gap=15):
    """같은 종류가 붙어 있으면 하나로 묶는다 (자판기 병 12개 → 병 묶음 1개)."""
    found = [dict(c) for c in found]
    merged = True
    while merged:
        merged = False
        for i in range(len(found)):
            for j in range(i + 1, len(found)):
                a, b = found[i], found[j]
                if a["label"] != b["label"]:
                    continue
                near = (a["box"][0] - gap < b["box"][2] and b["box"][0] - gap < a["box"][2]
                        and a["box"][1] - gap < b["box"][3] and b["box"][1] - gap < a["box"][3])
                if near:
                    a["box"] = [min(a["box"][0], b["box"][0]), min(a["box"][1], b["box"][1]),
                                max(a["box"][2], b["box"][2]), max(a["box"][3], b["box"][3])]
                    found.pop(j)
                    merged = True
                    break
            if merged:
                break
    return found


def pixel_change(ca, cb):
    """두 조각을 몇 픽셀 맞춘 뒤, 색이 크게 다른 픽셀의 비율.

    Gemini는 B를 새로 그리면서 선이 몇 픽셀씩 어긋나므로, 조각끼리 먼저 맞추고
    살짝 흐리게 해서 선 어긋남을 줄인 다음 비교한다.
    """
    ga = cv2.cvtColor(ca, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(cb, cv2.COLOR_BGR2GRAY).astype(np.float32)
    (dx, dy), _ = cv2.phaseCorrelate(ga, gb)
    dx, dy = float(np.clip(dx, -10, 10)), float(np.clip(dy, -10, 10))
    cb = cv2.warpAffine(cb, np.float32([[1, 0, -dx], [0, 1, -dy]]), (cb.shape[1], cb.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    la = cv2.cvtColor(cv2.GaussianBlur(ca, (0, 0), 3), cv2.COLOR_BGR2LAB).astype(np.float32)
    lb = cv2.cvtColor(cv2.GaussianBlur(cb, (0, 0), 3), cv2.COLOR_BGR2LAB).astype(np.float32)
    de = np.linalg.norm(la - lb, axis=2)
    return float((de > 25).mean())


def inside(small, big):
    """small 박스가 big 박스 안에 얼마나 들어가 있는지 (0~1)."""
    x1, y1 = max(small[0], big[0]), max(small[1], big[1])
    x2, y2 = min(small[2], big[2]), min(small[3], big[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((small[2] - small[0]) * (small[3] - small[1]) + 1e-9)


def hits(box, gt):
    """정답과 충분히 겹치거나(IoU ≥ 0.1), 정답 영역 안의 일부를 가리키면 찾은 것으로 본다."""
    return any(iou(box, g) >= HIT_IOU or inside(box, g) >= 0.6 for g in (gt["box_a"], gt.get("box_b", gt["box_a"])))


def draw(A, B, found, gts, path):
    A, B = A.copy(), B.copy()
    for g in gts:                                   # 정답: 초록
        ok = any(hits(f["box"], g) for f in found)
        for im, b in ((A, g["box_a"]), (B, g.get("box_b", g["box_a"]))):
            cv2.rectangle(im, b[:2], b[2:], (0, 200, 0) if ok else (0, 200, 255), 2)
            cv2.putText(im, str(g["no"]), (b[0] + 3, b[3] - 6), 0, 0.7, (0, 200, 0) if ok else (0, 200, 255), 2)
    for f in found:                                 # 탐지기가 찾은 곳: 빨강
        tp = any(hits(f["box"], g) for g in gts)
        for im in (A, B):
            cv2.rectangle(im, f["box"][:2], f["box"][2:], (0, 0, 255) if tp else (255, 0, 255), 3)
            cv2.putText(im, f["label"], (f["box"][0] + 3, f["box"][1] + 20), 0, 0.6, (0, 0, 255) if tp else (255, 0, 255), 2)
    sep = np.full((A.shape[0], 12, 3), 255, np.uint8)
    cv2.imwrite(str(path), cv2.hconcat([A, sep, B]), [cv2.IMWRITE_JPEG_QUALITY, 85])


def plot(cands, path):
    """후보마다 (픽셀 변화, 색 분포 거리)를 찍어서 기준선이 정답/오답을 어떻게 가르는지 보여 준다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=130)
    for flag, color, lab in ((False, "#8c959f", "정답과 무관한 후보"), (True, "#cf222e", "정답 자리 후보")):
        pts = [(p, c) for p, c, g in cands if g == flag]
        ax.scatter([p for p, _ in pts], [c for _, c in pts], s=18, c=color, alpha=0.75, label=lab)
    ax.axvline(PIX_TH, ls="--", c="#0969da"); ax.axhline(COLOR_TH, ls="--", c="#0969da")
    ax.set_xlabel("픽셀 변화 비율"); ax.set_ylabel("색 분포 거리")
    ax.set_title("후보별 내용 차이 (파란 점선 밖이면 '틀린 곳')")
    ax.legend(loc="upper left"); fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main():
    dump = "--dump" in sys.argv
    answers = {k: v for k, v in json.loads((DS / "answers.json").read_text(encoding="utf-8")).items() if not k.startswith("_")}
    OUT.mkdir(exist_ok=True)
    finder = Finder()
    total = dict(gt=0, found=0, pred=0, tp=0)
    rows, all_cands = [], []
    for name, gts in answers.items():
        A = cv2.imread(str(DS / "A" / f"{name}.png"))
        B = cv2.imread(str(DS / "B" / f"{name}.png"))
        found, cands, da, db = finder.find(A, B)
        all_cands += [(c["pix"], c["color"], any(hits(c["box"], g) for g in gts)) for c in cands]
        if dump:
            print(f"\n== {name}  (A {len(da)}개, B {len(db)}개 탐지)")
            for c in sorted(cands, key=lambda c: -max(c["pix"] / PIX_TH, c["color"] / COLOR_TH)):
                g = [x["no"] for x in gts if hits(c["box"], x)]
                print(f"  {'정답' + str(g) if g else '     '} {c['why']:6s} {c['label']:14s} 모양 {c['shape']:.3f} 색 {c['color']:.3f} 픽셀 {c['pix']:.3f} {'← 틀린 곳' if c['diff'] else ''}")
        n_found = sum(any(hits(f["box"], g) for f in found) for g in gts)
        n_tp = sum(any(hits(f["box"], g) for g in gts) for f in found)
        total["gt"] += len(gts); total["found"] += n_found; total["pred"] += len(found); total["tp"] += n_tp
        missed = [g["what"] for g in gts if not any(hits(f["box"], g) for f in found)]
        rows.append((name, len(gts), n_found, len(found), n_tp, missed))
        draw(A, B, found, gts, OUT / f"{name}.jpg")

    print("\n문제          정답  찾음  탐지  맞음")
    for name, g, f, p, t, _ in rows:
        print(f"{name:13s} {g:4d} {f:5d} {p:5d} {t:5d}")
    rec = total["found"] / total["gt"]
    prec = total["tp"] / max(1, total["pred"])
    print(f"합계          {total['gt']:4d} {total['found']:5d} {total['pred']:5d} {total['tp']:5d}")
    print(f"재현율(정답 중 찾은 비율) {rec:.1%}  정밀도(탐지 중 맞은 비율) {prec:.1%}")
    for name, *_, missed in rows:
        print(f"  {name} 놓침: {', '.join(missed) if missed else '없음'}")
    plot(all_cands, OUT / "threshold.png")
    (OUT / "score.json").write_text(json.dumps(dict(total=total, recall=rec, precision=prec,
        per_problem=[dict(name=r[0], gt=r[1], found=r[2], pred=r[3], tp=r[4], missed=r[5]) for r in rows]),
        ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
