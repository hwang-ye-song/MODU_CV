"""3D 디지털 트윈용 YOLO 서버 (표준 라이브러리 http.server + ultralytics).

- 같은 폴더(twin)의 정적 파일을 서빙한다 → http://127.0.0.1:8766/
- POST /detect?room=3층-1[&person_conf=0.35]
    본문: JPEG 바이트 그대로, 또는 JSON {"image": "<base64 또는 data URL>", "room": ..., "person_conf": ...}
    응답: 테이블 박스, 30% 감지 영역, 사람 박스(신뢰도·영역 안 여부), 영역 안 사람 수
    판정 로직은 mvp/detector.py 의 RoomDetector 를 그대로 쓴다. 룸마다 인스턴스를 따로 둔다(테이블 이동 평균 때문).
- GET /health → 서버 상태
- (측정용, --capture-dir 를 줄 때만) POST /debug/capture, POST /debug/done

실행: .venv\\Scripts\\python mvp\\twin\\yolo_server.py [--port 8766] [--person-conf 0.35]
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent          # mvp/twin
MVP = HERE.parent                               # mvp
ROOT = MVP.parent                               # CV프로젝트
sys.path.insert(0, str(MVP))
from detector import Box, RoomDetector  # noqa: E402  (mvp/detector.py 재사용)


def box_json(b: Box | None):
    return None if b is None else [round(b.x1, 1), round(b.y1, 1), round(b.x2, 1), round(b.y2, 1), round(b.conf, 3)]


class Engine:
    """YOLO 모델 하나를 여러 룸의 RoomDetector 가 함께 쓴다 (GPU 추론은 잠금으로 한 번에 하나씩)."""

    def __init__(self, weights: str, imgsz: int, person_conf: float, table_conf: float, device: str | None):
        self.base = RoomDetector(weights=weights, imgsz=imgsz, person_conf=person_conf, table_conf=table_conf,
                                 device=device)
        self.base.detect(np.zeros((720, 960, 3), np.uint8))   # 워밍업
        self.base.table = None
        self.rooms: dict[str, RoomDetector] = {}
        self.lock = threading.Lock()
        self.count = 0

    def room(self, rid: str) -> RoomDetector:
        if rid not in self.rooms:
            d = copy.copy(self.base)       # 모델은 공유, 테이블 이동 평균은 룸마다 따로
            d.table = None
            self.rooms[rid] = d
        return self.rooms[rid]

    def detect(self, rid: str, frame: np.ndarray, person_conf: float | None):
        h, w = frame.shape[:2]
        with self.lock:
            det = self.room(rid)
            if person_conf is not None:
                det.person_conf = person_conf
            prev = det.table
            t0 = time.perf_counter()
            zone, persons, inside = det.detect(frame)
            ms = (time.perf_counter() - t0) * 1000
            self.count += 1
            table_seen = det.table is not prev        # 이번 프레임에서 테이블을 새로 찾았는가
        ins = {id(p) for p in inside}
        return {
            "room": rid, "w": w, "h": h, "ms": round(ms, 1),
            "table": box_json(det.table), "table_seen": table_seen,
            "zone": box_json(zone), "zone_source": "table" if det.table is not None else "full",
            "persons": [{"box": box_json(p), "conf": round(p.conf, 3), "inside": id(p) in ins,
                         "overlap": round(zone.overlap_of(p), 3)} for p in persons],
            "inside_count": len(inside), "person_count": len(persons),
            "person_conf": det.person_conf,
        }


def decode_image(raw: bytes) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def b64_bytes(s: str) -> bytes:
    if s.startswith("data:"):
        s = s.split(",", 1)[1]
    return base64.b64decode(s)


def make_handler(engine: Engine, args):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(HERE), **k)

        def log_message(self, fmt, *a):      # 요청마다 찍히는 로그는 끈다
            if args.verbose:
                super().log_message(fmt, *a)

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def send_json(self, obj, code=200):
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if urlparse(self.path).path == "/health":
                return self.send_json({"ok": True, "model": Path(args.weights).name, "imgsz": args.imgsz,
                                       "person_conf": args.person_conf, "rooms": len(engine.rooms),
                                       "frames": engine.count})
            return super().do_GET()

        def body(self) -> bytes:
            return self.rfile.read(int(self.headers.get("Content-Length", 0)))

        def do_POST(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path == "/detect":
                    raw = self.body()
                    if "json" in (self.headers.get("Content-Type") or ""):
                        j = json.loads(raw)
                        raw = b64_bytes(j["image"])
                        q.setdefault("room", j.get("room", "room"))
                        if j.get("person_conf") is not None:
                            q.setdefault("person_conf", str(j["person_conf"]))
                    frame = decode_image(raw)
                    if frame is None:
                        return self.send_json({"error": "이미지를 읽을 수 없습니다"}, 400)
                    conf = float(q["person_conf"]) if q.get("person_conf") else args.person_conf
                    return self.send_json(engine.detect(unquote(q.get("room", "room")), frame, conf))

                if url.path.startswith("/debug/") and args.capture_dir:
                    out = Path(args.capture_dir)
                    out.mkdir(parents=True, exist_ok=True)
                    if url.path == "/debug/capture":
                        j = json.loads(self.body())
                        name = Path(j["name"]).name
                        (out / f"{name}.jpg").write_bytes(b64_bytes(j["image"]))
                        (out / f"{name}.json").write_text(json.dumps(j["gt"], ensure_ascii=False), "utf-8")
                        return self.send_json({"ok": True})
                    if url.path == "/debug/done":
                        (out / "done.json").write_text(self.body().decode("utf-8") or "{}", "utf-8")
                        return self.send_json({"ok": True})
                return self.send_json({"error": "없는 주소입니다"}, 404)
            except Exception as e:  # noqa: BLE001
                return self.send_json({"error": str(e)}, 500)

    return Handler


def main():
    ap = argparse.ArgumentParser(description="3D 디지털 트윈용 YOLO 서버")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--weights", default=str(ROOT / "yolo11s.pt"))
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--person-conf", type=float, default=0.35, help="사람 신뢰도 기준 (기본 0.35)")
    ap.add_argument("--table-conf", type=float, default=0.15,
                    help="테이블 신뢰도 기준 (3D 렌더 실측으로 0.15, MVP 기본은 0.25)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--capture-dir", default=None, help="측정용: 렌더 이미지와 정답을 저장할 폴더")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--open", action="store_true", help="모델을 불러온 뒤 브라우저로 페이지 열기")
    args = ap.parse_args()

    print(f"YOLO 모델 불러오는 중: {args.weights}")
    engine = Engine(args.weights, args.imgsz, args.person_conf, args.table_conf, args.device)
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(engine, args))
    url = f"http://{args.host}:{args.port}/"
    print(f"YOLO 서버 실행 중: {url}  (person_conf={args.person_conf}, table_conf={args.table_conf}, 끄려면 Ctrl+C)")
    if args.open:
        import webbrowser
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
