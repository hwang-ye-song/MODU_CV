"""스터디룸 알박기 감지 MVP 실행.

    # 웹캠 실시간 (실제 기준 30분)
    python mvp/run.py --source 0

    # 녹화 영상으로 시연: 30분 규칙을 30초로 압축(60배), 결과 영상 저장
    python mvp/run.py --source demo.mp4 --time-scale 60 --save 결과.mp4

남는 것 (logs/ 아래)
  데이터     presence.csv  일정 간격으로 룸, 시각, 사람 있음/없음, 상태      → 계속 남김
             events.csv    상태가 바뀐 순간과 그때까지의 부재 시간           → 계속 남김
             status.json   현재 상태 (현황판·표시등이 읽어 감)
  탐지 결과  snapshots/    상태가 바뀐 순간의 화면 1장 (사람 모자이크)       → 오류 검수용, 1일 뒤 자동 삭제
영상 자체는 저장하지 않는다. --save 결과 영상은 발표 시연용이며 여기서도 사람은 모자이크된다.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2

from detector import Box, RoomDetector
from room_logic import RoomTracker, Rules
from visualize import draw

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    ap = argparse.ArgumentParser(description="스터디룸 알박기 감지 MVP")
    ap.add_argument("--source", required=True, help="웹캠 번호(0) 또는 영상 경로")
    ap.add_argument("--room", default="3층-1", help="룸 이름")
    ap.add_argument("--limit-min", type=float, default=30, help="이만큼 비면 초록불 (분)")
    ap.add_argument("--time-scale", type=float, default=1.0, help="시연용 시간 압축 (60이면 30분 → 30초)")
    ap.add_argument("--weights", default=str(ROOT / "yolo11s.pt"))
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--table-conf", type=float, default=0.25, help="테이블 검출 신뢰도 기준")
    ap.add_argument("--expand", type=float, default=0.30, help="테이블 박스를 넓히는 비율")
    ap.add_argument("--zone", default=None, help="테이블이 안 잡힐 때 직접 지정: x1,y1,x2,y2")
    ap.add_argument("--fps", type=float, default=3.0, help="초당 분석 프레임 수")
    ap.add_argument("--log-every", type=float, default=10.0, help="presence.csv 기록 간격 (실제 초)")
    ap.add_argument("--keep-hours", type=float, default=24.0, help="탐지 결과(스냅샷) 보관 시간")
    ap.add_argument("--save", default=None, help="결과 영상 저장 경로 (시연용)")
    ap.add_argument("--log-dir", default=str(ROOT / "logs"))
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--no-privacy", action="store_true", help="모자이크 끄기 (디버깅용)")
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def purge_old(folder: Path, keep_hours: float) -> int:
    """보관 시간이 지난 탐지 결과를 지운다."""
    limit = time.time() - keep_hours * 3600
    n = 0
    for f in folder.glob("*.jpg"):
        if f.stat().st_mtime < limit:
            f.unlink(missing_ok=True)
            n += 1
    return n


def open_csv(path: Path, header: list[str]):
    f = open(path, "a", newline="", encoding="utf-8-sig")
    w = csv.writer(f)
    if f.tell() == 0:
        w.writerow(header)
    return f, w


def main():
    a = parse_args()
    rules = Rules(limit_sec=a.limit_min * 60 / a.time_scale)   # 가림 보정(on/off)은 실제 시간 그대로
    zone = Box(*map(float, a.zone.split(","))) if a.zone else None
    det = RoomDetector(a.weights, a.imgsz, table_conf=a.table_conf, expand=a.expand, fixed_zone=zone, device=a.device)

    is_cam = a.source.isdigit()
    cap = cv2.VideoCapture(int(a.source) if is_cam else a.source)
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없습니다: {a.source}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(src_fps / a.fps))
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("첫 프레임을 읽을 수 없습니다.")
    h, w = frame.shape[:2]

    tracker = RoomTracker(a.room, rules, start=0.0)
    log_dir = Path(a.log_dir); log_dir.mkdir(parents=True, exist_ok=True)
    snap_dir = log_dir / "snapshots"; snap_dir.mkdir(exist_ok=True)
    removed = purge_old(snap_dir, a.keep_hours)
    ev_f, ev = open_csv(log_dir / "events.csv", ["시각", "룸", "이전상태", "새상태", "부재시간(초)", "스냅샷"])
    pr_f, pr = open_csv(log_dir / "presence.csv", ["시각", "룸", "사람", "상태"])
    writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h)) if a.save else None

    wall0, t0, idx = datetime.now(), time.monotonic(), 0
    last_log, last_purge = -1e9, time.monotonic()
    print(f"{a.room}: {a.limit_min:g}분 규칙 (시연 배율 {a.time_scale:g}배), "
          f"탐지 결과 {a.keep_hours:g}시간 보관 (지난 것 {removed}장 삭제)  q로 종료")
    try:
        while ok:
            now = (time.monotonic() - t0) if is_cam else idx / src_fps
            when = wall0 + timedelta(seconds=now * a.time_scale)
            stamp = when.strftime("%Y-%m-%d %H:%M:%S")
            zone_box, persons, inside = det.detect(frame)
            prev = tracker.state
            state = tracker.update(bool(inside), now)
            changed = state is not prev
            need_vis = changed or writer is not None or not a.no_show
            vis = draw(frame, zone_box, persons, inside, tracker, now, a.time_scale,
                       table=det.table, privacy=not a.no_privacy) if need_vis else None

            if changed:
                snap = snap_dir / f"{when:%Y%m%d_%H%M%S}_{a.room}_{state.value}.jpg"
                cv2.imencode(".jpg", vis)[1].tofile(str(snap))   # 한글 경로 대응
                ev.writerow([stamp, a.room, prev.value, state.value,
                             round(tracker.away_sec(now) * a.time_scale), snap.name])
                ev_f.flush()
                print(f"[{stamp}] {a.room}: {prev.value} → {state.value}  (스냅샷 {snap.name})")
            if now - last_log >= a.log_every / a.time_scale:
                last_log = now
                pr.writerow([stamp, a.room, int(bool(inside)), state.value]); pr_f.flush()
            (log_dir / "status.json").write_text(json.dumps({
                "updated": stamp, "room": a.room, "state": state.value, "light": state.light,
                "away_sec": round(tracker.away_sec(now) * a.time_scale)}, ensure_ascii=False), encoding="utf-8")
            if time.monotonic() - last_purge > 600:          # 10분마다 오래된 탐지 결과 정리
                purge_old(snap_dir, a.keep_hours); last_purge = time.monotonic()

            if writer:
                writer.write(vis)
            if not a.no_show:
                cv2.imshow("study room", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if is_cam:
                ok, frame = cap.read()
            else:
                for _ in range(step - 1):
                    cap.grab()
                ok, frame = cap.read()
                idx += step
    finally:
        cap.release()
        if writer:
            writer.release()
        ev_f.close(); pr_f.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
