"""3D 트윈 천장 카메라 렌더에서 YOLO11s 가 사람·테이블을 얼마나 잡는지 측정한다.

1) 수집: yolo_server.py 를 --capture-dir 와 함께 띄운 뒤
       python eval_yolo.py --collect --trials 8
   → 헤드리스 Edge 로 http://127.0.0.1:8766/?eval=1 을 열어 룸 14개 × trials 장면을 렌더·저장
2) 분석: python eval_yolo.py   (수집 폴더의 jpg + 정답 json 으로 표를 출력)

정답 박스는 3D 모델 꼭짓점을 카메라에 투영한 것이다 (가려짐은 반영하지 않음).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent))
from detector import PERSON, TABLE, RoomDetector  # noqa: E402

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
POSE_KO = {"study": "앉아서 공부", "sleep": "엎드려 잠", "walk": "걷는 중"}


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def ioa(a, b):
    """a 면적 중 b 안에 들어간 비율."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / area if area > 0 else 0.0


def imread(path: Path):
    """한글 경로에서도 읽히도록."""
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def collect(args):
    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*"):
        f.unlink()
    prof = tempfile.mkdtemp(prefix="edge_eval_")
    url = f"http://127.0.0.1:{args.port}/?eval=1&trials={args.trials}{args.extra}"
    cmd = [EDGE, "--headless=new", f"--user-data-dir={prof}", "--no-first-run", "--disable-extensions",
           "--ignore-gpu-blocklist", "--enable-unsafe-swiftshader", "--window-size=1400,900", url]
    print("헤드리스 Edge:", url)
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    try:
        while not (out / "done.json").exists():
            if time.time() - t0 > args.timeout:
                print("시간 초과: 저장된 장면", len(list(out.glob("*.jpg"))))
                break
            time.sleep(1)
    finally:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
    print(f"수집 완료: {len(list(out.glob('*.jpg')))}장, {time.time() - t0:.0f}초")


def analyze(args):
    d = Path(args.dir)
    files = sorted(d.glob("t*_r*.json"))
    if not files:
        sys.exit(f"{d} 에 측정 장면이 없습니다. --collect 먼저 실행하세요.")
    base = RoomDetector(weights=str(ROOT / "yolo11s.pt"), imgsz=args.imgsz, person_conf=args.conf, table_conf=args.table_conf)
    dets: dict[str, RoomDetector] = {}
    pose = defaultdict(lambda: {"n": 0, "hit": 0, "conf": []})
    tab = {"n": 0, "seen": 0, "iou_ok": 0, "conf": []}
    room_cm = defaultdict(int)       # (정답 재석, 예측 재석) → 개수
    full_zone = 0
    fp_total = 0
    per_img = []
    for jf in files:
        gt = json.loads(jf.read_text("utf-8"))
        img = imread(jf.with_suffix(".jpg"))
        res = base.model.predict(img, imgsz=args.imgsz, verbose=False, classes=[PERSON, TABLE],
                                 conf=min(args.conf, args.table_conf))[0]
        persons, tables = [], []
        for b, c, s in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
            if int(c) == PERSON and s >= args.conf:
                persons.append((b, s))
            elif int(c) == TABLE and s >= args.table_conf:
                tables.append((b, s))
        # 사람: 화면에 절반 이상 들어온 정답만 평가.
        # 정답 박스는 테이블에 가린 다리까지 포함하므로, 검출 박스의 중심이 정답 박스 안에 있고
        # 검출 박스의 60% 이상이 정답 박스 안이면 같은 사람으로 본다 (IoU 큰 순으로 1:1).
        cand = []
        for gi, g in enumerate(gt["persons"]):
            if not g or g.get("vis", 0) < 0.5:
                continue
            for bi, (b, s) in enumerate(persons):
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                if not (g["box"][0] <= cx <= g["box"][2] and g["box"][1] <= cy <= g["box"][3]):
                    continue
                if ioa(b, g["box"]) >= 0.6 or iou(b, g["box"]) >= args.iou:
                    cand.append((iou(g["box"], b), gi, bi))
        used_g, used_b, matched = set(), set(), {}
        for v, gi, bi in sorted(cand, reverse=True):
            if gi in used_g or bi in used_b:
                continue
            used_g.add(gi); used_b.add(bi); matched[gi] = persons[bi][1]
        for gi, g in enumerate(gt["persons"]):
            if not g or g.get("vis", 0) < 0.5:
                continue
            keys = [g["pose"]]
            if "faceCam" in g and g["pose"] in ("study", "sleep"):
                keys.append((g["pose"], g["faceCam"]))
            for k in keys:
                st = pose[k]
                st["n"] += 1
                if gi in matched:
                    st["hit"] += 1
                    st["conf"].append(matched[gi])
        fp_total += len(persons) - len(used_b)
        # 테이블
        if gt.get("table"):
            tab["n"] += 1
            if tables:
                tab["seen"] += 1
                bt, s = max(tables, key=lambda t: (t[0][2] - t[0][0]) * (t[0][3] - t[0][1]))
                tab["conf"].append(s)
                if iou(gt["table"]["box"], bt) >= 0.3:
                    tab["iou_ok"] += 1
        # 룸 판정: 서버와 같은 RoomDetector (룸마다 따로, 장면 순서대로)
        rid = gt["room"]
        if rid not in dets:
            dets[rid] = copy.copy(base)
            dets[rid].table = None
        zone, ps, inside = dets[rid].detect(img)
        if dets[rid].table is None:
            full_zone += 1
        truth = any(p and p["pose"] in ("study", "sleep") for p in gt["persons"])
        pred = len(inside) > 0
        room_cm[(truth, pred)] += 1
        per_img.append((jf.stem, rid, len([p for p in gt["persons"] if p]), len(persons), len(inside)))

    n_img = len(files)
    print(f"\n측정 장면 {n_img}장 (960×720, imgsz {args.imgsz}, person_conf {args.conf}, 매칭 IoU ≥ {args.iou})\n")
    print("| 자세 | 사람 수 | 검출 | 검출률 | 평균 신뢰도 |")
    print("|---|---:|---:|---:|---:|")
    tot = {"n": 0, "hit": 0, "conf": []}
    for k in ("study", "sleep", "walk"):
        st = pose[k]
        tot["n"] += st["n"]; tot["hit"] += st["hit"]; tot["conf"] += st["conf"]
        rate = st["hit"] / st["n"] * 100 if st["n"] else 0
        mc = sum(st["conf"]) / len(st["conf"]) if st["conf"] else 0
        print(f"| {POSE_KO[k]} | {st['n']} | {st['hit']} | {rate:.1f}% | {mc:.2f} |")
    print(f"| 전체 | {tot['n']} | {tot['hit']} | {tot['hit'] / max(1, tot['n']) * 100:.1f}% | "
          f"{sum(tot['conf']) / max(1, len(tot['conf'])):.2f} |")
    for k in ("study", "sleep"):
        for f, name in ((True, "얼굴이 카메라 쪽"), (False, "등이 카메라 쪽")):
            st = pose[(k, f)]
            if st["n"]:
                mc = sum(st["conf"]) / len(st["conf"]) if st["conf"] else 0
                print(f"| └ {POSE_KO[k]} · {name} | {st['n']} | {st['hit']} | {st['hit'] / st['n'] * 100:.1f}% | {mc:.2f} |")
    print(f"\n정답과 안 맞는 사람 박스(옆 룸 사람·오검출): {fp_total}개")
    tc = sum(tab["conf"]) / len(tab["conf"]) if tab["conf"] else 0
    print(f"테이블(dining table ≥{args.table_conf}) 검출: {tab['seen']}/{tab['n']} ({tab['seen'] / max(1, tab['n']) * 100:.1f}%), "
          f"정답 위치와 일치(IoU≥0.3) {tab['iou_ok']}/{tab['n']}, 평균 신뢰도 {tc:.2f}")
    print(f"테이블을 한 번도 못 찾아 화면 전체를 감지 영역으로 쓴 장면: {full_zone}/{n_img}")
    tp, fn = room_cm[(True, True)], room_cm[(True, False)]
    tn, fp = room_cm[(False, False)], room_cm[(False, True)]
    print(f"\n룸 판정(영역 안 사람 ≥1): 사람 있는 룸 {tp}/{tp + fn} 맞힘, 사람 없는 룸 {tn}/{tn + fp} 맞힘 "
          f"→ 정확도 {(tp + tn) / max(1, n_img) * 100:.1f}%")
    if args.verbose:
        for row in per_img:
            print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--extra", default="", help="페이지에 덧붙일 쿼리 (예: &cam=end)")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--dir", default=str(HERE / "_captures"))
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--table-conf", type=float, default=0.15)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.collect:
        collect(args)
    analyze(args)


if __name__ == "__main__":
    main()
