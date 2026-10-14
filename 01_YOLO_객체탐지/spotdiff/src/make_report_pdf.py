"""작업기록.md를 PDF로 만든다.

    python spotdiff/src/make_report_pdf.py

결과: spotdiff/report/작업기록.pdf
"""
import shutil
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "report" / "작업기록.md"
OUT = ROOT / "report" / "작업기록.pdf"
sys.path.insert(0, str(ROOT.parent / "tools"))
from make_pdf import EDGE_CANDIDATES, print_pdf  # noqa: E402

CSS = """
<style>
  @page { size: A4; margin: 16mm 15mm; }
  body { font-family: 'Malgun Gothic', sans-serif; color: #1f2328; font-size: 10.5pt; line-height: 1.65; }
  h1 { font-size: 20pt; border-bottom: 2px solid #1f2328; padding-bottom: 4px; margin-top: 0; }
  h2 { font-size: 15pt; border-bottom: 1px solid #d0d7de; padding-bottom: 3px; margin-top: 22px; page-break-after: avoid; }
  h3 { font-size: 12pt; margin-top: 16px; page-break-after: avoid; }
  img { display: block; max-width: 100%; max-height: 120mm; margin: 8px auto; page-break-inside: avoid; }
  table { border-collapse: collapse; margin: 8px 0; page-break-inside: avoid; }
  th, td { border: 1px solid #d0d7de; padding: 3px 9px; }
  th { background: #f6f8fa; }
  blockquote { margin: 6px 0; padding: 2px 12px; color: #57606a; border-left: 3px solid #d0d7de; }
  code { font-family: Consolas, monospace; background: #f6f8fa; padding: 0 3px; border-radius: 3px; font-size: 9.5pt; }
  hr { border: none; border-top: 1px solid #d0d7de; margin: 18px 0; }
  li { margin: 2px 0; }
</style>
"""


def main():
    body = markdown.markdown(MD.read_text(encoding="utf-8"), extensions=["tables"])
    doc = (f'<!doctype html><html><head><meta charset="utf-8"><base href="{MD.parent.as_uri()}/">'
           f"{CSS}</head><body>{body}</body></html>")
    edge = next(p for p in EDGE_CANDIDATES if p.exists())
    tmp = ROOT.parent / "build" / "report_pdf"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    (tmp / "doc.html").write_text(doc, encoding="utf-8")
    print_pdf(edge, tmp / "doc.html", tmp / "doc.pdf")
    shutil.copy2(tmp / "doc.pdf", OUT)
    print("완료 →", OUT)


if __name__ == "__main__":
    main()
