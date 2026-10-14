"""노트북 출력을 가볍고 단순하게 만든다 (GitHub 웹에서 노트북이 열리도록).

    python tools/shrink_notebook.py notebooks/yolov8_manufacturing_practice.ipynb

- 학습 로그처럼 수천 조각으로 나뉜 출력을 셀마다 하나로 합치고,
  진행 막대가 '\\r' 로 덮어쓴 줄은 Jupyter 화면처럼 마지막 상태만 남기고, 색상 코드는 지운다
- 가로가 MAX_W 보다 큰 이미지는 비율을 유지해 줄인다
- PNG·JPEG 를 JPEG(품질 75)로 바꿨을 때 더 작아지는 경우에만 바꾼다 (사진이 들어간 그림)
코드와 결과 내용은 바꾸지 않는다.
"""
import base64
import io
import json
import re
import sys
from pathlib import Path

from PIL import Image

MAX_W = 1000
QUALITY = 75


def shrink(b64: str) -> tuple[str, str]:
    """(mime, base64) 를 돌려준다."""
    im = Image.open(io.BytesIO(base64.b64decode(b64)))
    if im.width > MAX_W:
        im = im.resize((MAX_W, round(im.height * MAX_W / im.width)), Image.LANCZOS)
    png = io.BytesIO(); im.save(png, "PNG", optimize=True)
    jpg = io.BytesIO(); im.convert("RGB").save(jpg, "JPEG", quality=QUALITY, optimize=True)
    best_mime, best = ("image/jpeg", jpg) if jpg.tell() < png.tell() * 0.8 else ("image/png", png)
    return best_mime, base64.b64encode(best.getvalue()).decode()


ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def render_text(text: str) -> str:
    """터미널처럼 해석한다: 색상 코드 제거, '\\r' 로 덮어쓴 줄은 마지막 내용만 남김."""
    lines = []
    for line in ANSI.sub("", text).split("\n"):
        lines.append(line.rstrip("\r").split("\r")[-1])
    return "\n".join(lines)


def merge_streams(outputs: list) -> list:
    """같은 종류(stdout/stderr)의 연속된 출력 조각을 하나로 합친다."""
    merged = []
    for out in outputs:
        if out["output_type"] == "stream":
            text = "".join(out["text"]) if isinstance(out["text"], list) else out["text"]
            if merged and merged[-1]["output_type"] == "stream" and merged[-1]["name"] == out["name"]:
                merged[-1]["text"] += text
                continue
            out = {**out, "text": text}
        merged.append(out)
    for out in merged:
        if out["output_type"] == "stream":
            out["text"] = render_text(out["text"]).splitlines(True)
    return merged


def main(path: str):
    p = Path(path)
    before = p.stat().st_size
    nb = json.loads(p.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = merge_streams(cell["outputs"])
        for out in cell.get("outputs", []):
            data = out.get("data", {})
            for key in ("image/png", "image/jpeg"):
                if key in data:
                    mime, b64 = shrink("".join(data.pop(key)))
                    data[mime] = b64
                    out.get("metadata", {}).pop(key, None)
                    break
    p.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{before / 1e6:.1f} MB → {p.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main(sys.argv[1])
