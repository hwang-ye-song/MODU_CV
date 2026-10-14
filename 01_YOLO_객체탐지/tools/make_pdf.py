"""실습 노트북(+ 선택적으로 보고서)을 제출용 PDF 한 파일로 묶는다.

    python tools/make_pdf.py                 # 실습 노트북만
    python tools/make_pdf.py --with-report   # report/REPORT.md 도 포함

순서: 표지 → (보고서) → NOTEBOOKS 에 적힌 노트북 (실행된 것만)
각 부분을 HTML 로 만든 뒤 Edge(headless) 로 PDF 인쇄 → pypdf 로 병합 + 책갈피.
결과: CV-YOLO_제출본.pdf
"""
import argparse
import datetime as dt
import html
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import markdown
import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import Preprocessor
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "CV-YOLO_제출본.pdf"
REPO_URL = "https://github.com/hwang-ye-song/CV_YOLO"
AUTHOR = "hwang-ye-song"
TITLE = "YOLOv8 기반 제조 데이터<br>객체 탐지 실습"
SUBTITLE = "사전학습 모델 이해 → stamp 데이터 학습·평가 → 모델·조건별 성능 실험"
NOTEBOOKS = ["yolov8_manufacturing_practice.ipynb"]
MAX_STREAM_LINES = 30   # 학습 로그처럼 긴 출력은 앞뒤만 남긴다
MAX_STREAM_CHARS = 2400
EDGE_CANDIDATES = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]

PRINT_CSS = """
<style>
  @page { size: A4; margin: 14mm 12mm; }
  body { font-family: 'Malgun Gothic', 'Segoe UI', sans-serif; font-size: 10.5pt; line-height: 1.55; color: #1f2328; }
  pre, code { font-family: Consolas, 'D2Coding', monospace; font-size: 8.5pt; white-space: pre-wrap !important; word-break: break-all; }
  img { max-width: 100% !important; height: auto; page-break-inside: avoid; }
  table { border-collapse: collapse; margin: 8px 0; font-size: 9.5pt; }
  th, td { border: 1px solid #d0d7de; padding: 4px 8px; }
  th { background: #f6f8fa; }
  h1, h2, h3 { page-break-after: avoid; }
  .jp-OutputArea-output { overflow: visible !important; }
  .jp-OutputArea-output table.dataframe { font-size: 7pt; }
  .jp-OutputArea-output table.dataframe th, .jp-OutputArea-output table.dataframe td { padding: 2px 4px; }
  .jp-Cell { page-break-inside: auto; }
  .jp-InputArea, .jp-OutputArea-child { page-break-inside: avoid; }
</style>
"""


def shorten(lines: list[str]) -> list[str]:
    """앞뒤 일부만 남긴다. 줄 수(MAX_STREAM_LINES)와 글자 수(MAX_STREAM_CHARS) 둘 다 기준으로 자른다."""
    half_chars = MAX_STREAM_CHARS // 2
    head, n = [], 0
    for l in lines[:MAX_STREAM_LINES // 2]:
        if n + len(l) > half_chars:
            break
        head.append(l)
        n += len(l)
    tail, n = [], 0
    for l in reversed(lines[len(head):][-(MAX_STREAM_LINES // 2):]):
        if n + len(l) > half_chars:
            break
        tail.insert(0, l)
        n += len(l)
    skipped = len(lines) - len(head) - len(tail)
    return lines if skipped <= 0 else head + [f"... ({skipped}줄 생략) ..."] + tail


class DropNoise(Preprocessor):
    """stderr(진행바) 출력은 빼고, 긴 stdout 은 앞뒤만 남긴다."""

    def preprocess_cell(self, cell, resources, index):
        if cell.cell_type != "code":
            return cell, resources
        # 학습 로그는 여러 stream 조각으로 나뉘어 나오므로 연속된 stdout 을 먼저 합친다
        merged = []
        for o in cell.outputs:
            if o.output_type == "stream" and o.name == "stderr":
                continue
            if merged and o.output_type == "stream" and merged[-1].output_type == "stream":
                merged[-1].text += o.text
            else:
                merged.append(o)
        for o in merged:
            # 객체 repr 처럼 거대한 텍스트 결과도 앞뒤만 남긴다 (그림·표는 그대로)
            if o.output_type in ("execute_result", "display_data") and "text/plain" in o.get("data", {}) \
                    and len(o["data"]) == 1:
                o["data"]["text/plain"] = "\n".join(shorten(o["data"]["text/plain"].splitlines()))
            if o.output_type != "stream":
                continue
            lines = [l for l in o.text.splitlines() if "it/s" not in l and "s/it" not in l]
            o.text = "\n".join(shorten(lines)) + "\n"
        cell.outputs = merged
        return cell, resources


def cover_html(parts: list[str]) -> str:
    items = "".join(f"<li>{html.escape(p)}</li>" for p in parts)
    today = dt.date.today().isoformat()
    return f"""<!doctype html><html><head><meta charset="utf-8">{PRINT_CSS}</head><body>
<div style="margin-top:60mm;text-align:center">
  <div style="font-size:12pt;color:#57606a">컴퓨터 비전 프로젝트</div>
  <h1 style="font-size:26pt;margin:10px 0 6px">{TITLE}</h1>
  <div style="font-size:12pt;color:#57606a">{SUBTITLE}</div>
</div>
<div style="margin:40mm auto 0;width:70%">
  <table style="width:100%">
    <tr><th style="width:30%">작성자</th><td>{AUTHOR}</td></tr>
    <tr><th>레포지토리</th><td>{REPO_URL}</td></tr>
    <tr><th>작성일</th><td>{today}</td></tr>
    <tr><th>환경</th><td>Windows 11 · Python 3.11 · PyTorch 2.11 (CUDA 12.8) · Ultralytics 8.4 · RTX 5060 Laptop</td></tr>
  </table>
  <h3 style="margin-top:24px">목차</h3>
  <ol>{items}</ol>
</div>
</body></html>"""


def report_html(md_path: Path) -> str:
    text = md_path.read_text(encoding="utf-8")
    # 아직 만들어지지 않은 그래프는 깨진 이미지 대신 빼 둔다
    def keep_img(m):
        return m.group(0) if (md_path.parent / m.group(1)).exists() else "*(실험 후 추가될 그림)*"
    text = re.sub(r"!\[[^\]]*\]\(([^)]+)\)", keep_img, text)
    body = markdown.markdown(text, extensions=["tables", "fenced_code"])
    base = md_path.parent.as_uri() + "/"
    return f'<!doctype html><html><head><meta charset="utf-8"><base href="{base}">{PRINT_CSS}</head><body>{body}</body></html>'


def notebook_html(nb_path: Path) -> str:
    nb = nbformat.read(nb_path, as_version=4)
    exporter = HTMLExporter(template_name="lab")
    exporter.register_preprocessor(DropNoise, enabled=True)
    body, _ = exporter.from_notebook_node(nb)
    return body.replace("</head>", PRINT_CSS + "</head>", 1)


def is_executed(nb_path: Path) -> bool:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    return any(c.get("execution_count") for c in nb["cells"] if c["cell_type"] == "code")


def print_pdf(edge: Path, html_file: Path, pdf_file: Path, timeout: float = 180):
    subprocess.run([str(edge), "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--user-data-dir={html_file.parent / 'edge-profile'}",
                    f"--print-to-pdf={pdf_file}", html_file.as_uri()],
                   capture_output=True, timeout=timeout)
    # msedge.exe 는 인쇄가 끝나기 전에 반환될 수 있어서, 파일 크기가 멈출 때까지 기다린다
    deadline, last = time.time() + timeout, -1
    while time.time() < deadline:
        size = pdf_file.stat().st_size if pdf_file.exists() else -1
        if size > 0 and size == last:
            return
        last = size
        time.sleep(1.5)
    raise RuntimeError(f"PDF 생성 실패: {html_file}")


def main():
    edge = next((p for p in EDGE_CANDIDATES if p.exists()), None)
    if edge is None:
        raise SystemExit("Microsoft Edge 를 찾을 수 없습니다.")

    ap = argparse.ArgumentParser()
    ap.add_argument("--with-report", action="store_true")
    args = ap.parse_args()

    sections = [("프로젝트 보고서", report_html(ROOT / "report" / "REPORT.md"))] if args.with_report else []
    for nb in (ROOT / "notebooks" / n for n in NOTEBOOKS):
        if is_executed(nb):
            sections.append((f"실습 노트북 — {nb.name}", notebook_html(nb)))
        else:
            print(f"건너뜀 (아직 실행 안 됨): {nb.name}")

    # 시스템 Temp 는 앱 가상화로 Edge 와 경로가 어긋날 수 있어 프로젝트 안에서 작업한다
    tmp = ROOT / "build" / "pdf"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        pages = [("표지", cover_html([t for t, _ in sections]))] + sections
        writer = PdfWriter()
        for i, (title, doc) in enumerate(pages):
            h, p = tmp / f"{i:02d}.html", tmp / f"{i:02d}.pdf"
            h.write_text(doc, encoding="utf-8")
            print_pdf(edge, h, p)
            start = len(writer.pages)
            writer.append(PdfReader(p))
            writer.add_outline_item(title, start)
            print(f"{title}: {len(writer.pages) - start}쪽")
        writer.add_metadata({"/Title": TITLE.replace("<br>", " "), "/Author": AUTHOR})
        with open(OUT, "wb") as f:
            writer.write(f)
        print(f"완료 → {OUT} (총 {len(writer.pages)}쪽)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
