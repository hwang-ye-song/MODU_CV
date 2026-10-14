"""오픈스페이스 자리 알박기 감지 실행.

    # 웹캠 실시간 (실제 기준: 2시간 / 10% / 유예 15분)
    python src/run.py --source 0 --seats seats.json

    # 녹화 영상으로 시연: 시간을 60배 압축 (2시간 → 2분, 유예 15분 → 15초)
    python src/run.py --source demo.mp4 --seats seats.json --time-scale 60 --save out.mp4

영상은 저장하지 않는 것이 기본이다. 남는 기록은 logs/ 아래의
  presence.csv  (시각, 좌석, 사람 있음, 물건 있음, 상태, 재석 비율)
  events.csv    (상태가 바뀐 순간)
  status.json   (현재 좌석 현황, 입구 화면/웹페이지용)
뿐이며, 사람이 누구인지는 어디에도 남지 않는다.
--save 는 발표 시연용 결과 영상이 필요할 때만 쓰고, 사람은 모자이크 처리된다.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2

from detector import SeatDetector, SeatMap
from seat_logic import Rules, SeatTracker
from visualize import draw, fmt_sec


def parse_args():
    ap = argparse.ArgumentParser(description="오픈스페이스 자리 알박기 감지")
    ap.add_argument("--source", required=True, help="웹캠 번호(0) 또는 영상 경로")
    ap.add_argument("--seats", default="seats.json", help="roi_editor.py로 만든 좌석 파일")
    ap.add_argument("--weights", default="yolo11s.pt", help="사전학습 YOLO 가중치 (COCO)")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25, help="물건 검출 신뢰도 기준")
    ap.add_argument("--person-conf", type=float, default=0.35, help="사람 검출 신뢰도 기준")
    ap.add_argument("--fps", type=float, default=2.0, help="초당 분석 프레임 수")
    ap.add_argument("--time-scale", type=float, default=1.0,
                    help="판정 시간 압축 배율 (시연용, 60이면 1분 규칙이 1초)")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--log-every", type=float, default=10.0, help="presence.csv 기록 간격(초)")
    ap.add_argument("--save", default=None, help="결과 영상 저장 경로 (시연용, 기본은 저장 안 함)")
    ap.add_argument("--no-show", action="store_true", help="화면 창을 띄우지 않음")
    ap.add_argument("--no-privacy", action="store_true", help="화면에서 사람 모자이크를 끔")
    ap.add_argument("--device", default=None, help="cuda:0 / cpu (기본 자동)")
    return ap.parse_args()


def main():
    args = parse_args()
    seats_path = Path(args.seats)
    if not seats_path.exists():
        raise SystemExit(f"좌석 파일 {seats_path}이(가) 없습니다. 먼저 좌석 영역을 지정하세요.\n"
                         f"  python src/roi_editor.py --source {args.source} --out {seats_path}")
    cfg = json.loads(seats_path.read_text(encoding="utf-8"))
    if not cfg.get("seats"):
        raise SystemExit(f"{seats_path}에 좌석이 없습니다. roi_editor.py로 좌석을 1개 이상 그리세요.")
    rules = Rules.from_dict(cfg.get("rules")).scaled(args.time_scale)

    is_cam = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source) if is_cam else args.source)
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없습니다: {args.source}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("첫 프레임을 읽을 수 없습니다.")

    # 좌석 좌표를 지정할 때와 해상도가 다르면 비율대로 맞춘다
    h, w = frame.shape[:2]
    fw, fh = cfg.get("frame_size", [w, h])
    if (fw, fh) != (w, h):
        sx, sy = w / fw, h / fh
        for s in cfg["seats"]:
            s["polygon"] = [[round(x * sx), round(y * sy)] for x, y in s["polygon"]]

    detector = SeatDetector(args.weights, args.conf, args.person_conf, args.imgsz, args.device)
    seat_map = SeatMap(cfg["seats"], frame.shape,
                       cfg.get("person_overlap", 0.25), cfg.get("item_overlap", 0.5))
    trackers = {sid: SeatTracker(sid, rules) for sid in seat_map.ids}

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    pres_f = open(log_dir / "presence.csv", "a", newline="", encoding="utf-8-sig")
    ev_f = open(log_dir / "events.csv", "a", newline="", encoding="utf-8-sig")
    pres_w, ev_w = csv.writer(pres_f), csv.writer(ev_f)
    if pres_f.tell() == 0:
        pres_w.writerow(["시각", "좌석", "사람", "물건", "상태", "재석비율"])
    if ev_f.tell() == 0:
        ev_w.writerow(["시각", "좌석", "이전상태", "새상태", "재석비율", "관찰시간"])

    writer = None
    if args.save:
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                 max(1.0, args.fps), (w, h))

    # 시각: 웹캠은 실제 시각, 영상은 영상 안의 재생 시각을 쓴다
    wall_start = datetime.now()
    t0 = time.monotonic()
    step = max(1, round(src_fps / args.fps))
    frame_idx, last_log = 0, -1e9
    print(f"좌석 {len(trackers)}개, 규칙: 기간 {fmt_sec(rules.window_sec)}, "
          f"기준 {rules.min_ratio:.0%}, 유예 {fmt_sec(rules.grace_sec)}  (q로 종료)")

    try:
        while ok:
            now = (time.monotonic() - t0) if is_cam else frame_idx / src_fps
            stamp = (wall_start + timedelta(seconds=now * args.time_scale)).strftime("%Y-%m-%d %H:%M:%S")

            assigned = seat_map.assign(detector.detect(frame))
            for sid, tr in trackers.items():
                prev = tr.state
                # 빈자리가 되면 기록이 초기화되므로, 판정 근거는 갱신 직전 값을 남긴다
                before = (tr.presence_ratio(now), tr.observed_sec(now))
                a = assigned[sid]
                tr.update(a["person"], bool(a["items"]), now)
                if tr.state is not prev:
                    ratio, observed = (before if tr.session_start is None
                                       else (tr.presence_ratio(now), tr.observed_sec(now)))
                    ev_w.writerow([stamp, sid, prev.value, tr.state.value, f"{ratio:.3f}",
                                   fmt_sec(observed * args.time_scale)])
                    print(f"[{stamp}] {sid}: {prev.value} → {tr.state.value}")
            ev_f.flush()

            if now - last_log >= args.log_every / args.time_scale:
                last_log = now
                for sid, tr in trackers.items():
                    a = assigned[sid]
                    pres_w.writerow([stamp, sid, int(a["person"]), int(bool(a["items"])),
                                     tr.state.value, f"{tr.presence_ratio(now):.3f}"])
                pres_f.flush()
                write_status(log_dir / "status.json", stamp, trackers, now, args.time_scale)

            if not args.no_show or writer:
                vis = draw(frame, seat_map, trackers, assigned, now, args.time_scale,
                           privacy=not args.no_privacy, clock_text=stamp)
                if writer:
                    writer.write(vis)
                if not args.no_show:
                    cv2.imshow("seat monitor", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            # 다음 분석 프레임으로 이동 (중간 프레임은 디코딩만 건너뜀)
            if is_cam:
                ok, frame = cap.read()
            else:
                for _ in range(step - 1):
                    cap.grab()
                ok, frame = cap.read()
                frame_idx += step
    finally:
        cap.release()
        if writer:
            writer.release()
        pres_f.close()
        ev_f.close()
        cv2.destroyAllWindows()


def write_status(path: Path, stamp: str, trackers: dict[str, SeatTracker], now: float, scale: float):
    seats = []
    for sid, tr in trackers.items():
        left = tr.grace_left(now)
        seats.append({
            "id": sid,
            "state": tr.state.value,
            "available": tr.state.available,
            "presence_ratio": round(tr.presence_ratio(now), 3),
            "grace_left_sec": None if left is None else round(left * scale),
        })
    data = {"updated": stamp, "available": sum(s["available"] for s in seats),
            "total": len(seats), "seats": seats}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
