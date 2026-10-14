# -*- coding: utf-8 -*-
"""tools/content.md 하나로 노트북·정리본(md)·PDF·Notion 본문을 만든다.

사용법: python tools/build_all.py            (프로젝트 최상위에서)
        python tools/build_all.py --notion ids.json   (Notion 업로드 ID로 notion.md 생성)
"""
import base64, contextlib, io, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "tools" / "content.md").read_text(encoding="utf-8")
IMG = re.compile(r"^!\[\[([^|\]]+)\|([^\]]+)\]\]\s*$")
NB_NAME, MD_NAME, PDF_NAME = "Vision_AI_특강_정리.ipynb", "특강_정리본.md", "Vision_AI_특강_정리.pdf"


def blocks(text):
    """('md', str) / ('img', (file, caption)) / ('run', code) 순서 목록"""
    out, buf, lines, i = [], [], text.split("\n"), 0
    def flush():
        if "".join(buf).strip(): out.append(("md", "\n".join(buf).strip("\n")))
        buf.clear()
    while i < len(lines):
        line = lines[i]
        m = IMG.match(line)
        if m:
            flush(); out.append(("img", (m.group(1), m.group(2)))); i += 1; continue
        if line.strip() == "```python run":
            flush(); j = i + 1
            while lines[j].strip() != "```": j += 1
            out.append(("run", "\n".join(lines[i + 1:j]))); i = j + 1; continue
        if line.startswith("## ") and buf:
            flush()
        buf.append(line); i += 1
    flush()
    return out


def run_code(code, glb):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(code, glb)
    return buf.getvalue()


def build_notebook(items):
    cells, n, glb = [], 0, {}
    for kind, val in items:
        if kind == "md":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": val.splitlines(keepends=True)})
        elif kind == "img":
            n += 1; name, cap = val
            data = base64.b64encode((ROOT / "images" / name).read_bytes()).decode()
            cells.append({"cell_type": "code", "execution_count": n, "metadata": {},
                          "source": [f"# {cap}\n", "from IPython.display import Image\n", f'Image("images/{name}")'],
                          "outputs": [{"output_type": "display_data", "metadata": {},
                                       "data": {"image/png": data, "text/plain": [f"<{cap}>"]}}]})
        else:
            n += 1; out = run_code(val, glb)
            cells.append({"cell_type": "code", "execution_count": n, "metadata": {},
                          "source": val.splitlines(keepends=True),
                          "outputs": [{"name": "stdout", "output_type": "stream", "text": out.splitlines(keepends=True)}]})
    for i, c in enumerate(cells): c["id"] = f"cell-{i:02d}"
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}}
    (ROOT / NB_NAME).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return cells


def build_markdown(items, cells):
    parts, k = [], 0
    for (kind, val), cell in zip(items, cells):
        if kind == "md": parts.append(val)
        elif kind == "img": parts.append(f"![{val[1]}](images/{val[0]})\n\n*{val[1]}*")
        else:
            out = "".join(cell["outputs"][0]["text"]).rstrip()
            parts.append(f"```python\n{val}\n```\n\n**실행 결과 (이 정리본을 만든 PC)**\n\n```text\n{out}\n```")
    doc = "\n\n".join(parts) + "\n"
    (ROOT / MD_NAME).write_text(doc, encoding="utf-8")
    return doc


CSS = """
@page { size: A4; margin: 16mm 14mm; }
body { font-family: 'Malgun Gothic', 'Segoe UI Emoji', sans-serif; font-size: 10.5pt; line-height: 1.6; color: #1f2328; }
h1 { font-size: 21pt; border-bottom: 3px solid #1a73e8; padding-bottom: 6px; }
h2 { font-size: 15.5pt; color: #1a73e8; border-bottom: 1px solid #d0d7de; padding-bottom: 4px; margin-top: 26px; page-break-after: avoid; }
h3 { font-size: 12.5pt; margin-top: 16px; page-break-after: avoid; }
h4 { font-size: 11pt; page-break-after: avoid; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 9.3pt; page-break-inside: avoid; }
th, td { border: 1px solid #d0d7de; padding: 4px 7px; vertical-align: top; }
th { background: #f0f5ff; }
code { font-family: Consolas, 'Malgun Gothic', monospace; background: #f3f4f6; padding: 1px 4px; border-radius: 3px; font-size: 8.8pt; }
pre { background: #0d1117; color: #e6edf3; padding: 10px 12px; border-radius: 6px; white-space: pre-wrap; word-break: break-all; font-size: 8.5pt; line-height: 1.45; page-break-inside: avoid; }
pre code { background: none; color: inherit; padding: 0; font-family: Consolas, 'Malgun Gothic', monospace; }
blockquote { border-left: 4px solid #1a73e8; background: #f6f9ff; margin: 8px 0; padding: 6px 12px; }
img { max-width: 100%; display: block; margin: 8px auto; page-break-inside: avoid; }
"""


def build_pdf(doc):
    import markdown
    from playwright.sync_api import sync_playwright
    lines, out, in_code = doc.split("\n"), [], False
    for i, line in enumerate(lines):
        if line.strip().startswith("```"): in_code = not in_code
        if not in_code and i and re.match(r"^(\s*[-*] |\s*\d+\. |\|)", line) and lines[i - 1].strip() \
                and not re.match(r"^(\s*[-*] |\s*\d+\. |\||>)", lines[i - 1]) and not lines[i - 1].strip().startswith("```"):
            out.append("")
        out.append(line.replace("- [ ]", "- ☐"))
    body = markdown.markdown("\n".join(out), extensions=["tables", "fenced_code", "sane_lists"])
    html = f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"
    tmp = ROOT / "_print.html"; tmp.write_text(html, encoding="utf-8")
    with sync_playwright() as p:
        b = p.chromium.launch(); page = b.new_page(); page.goto(tmp.as_uri()); page.wait_for_load_state("networkidle")
        page.pdf(path=str(ROOT / PDF_NAME), format="A4", print_background=True, display_header_footer=True,
                 header_template="<span></span>",
                 footer_template="<div style='font-size:8px;width:100%;text-align:center;color:#888'><span class='pageNumber'></span> / <span class='totalPages'></span></div>",
                 margin={"top": "16mm", "bottom": "16mm", "left": "14mm", "right": "14mm"})
        b.close()
    tmp.unlink()


def to_notion(items, ids):
    """Notion-flavored Markdown 변환: 표 → <table>, 인용 → 콜아웃, 이미지 → file-upload"""
    out = []
    for kind, val in items:
        if kind == "img":
            out.append(f"![{val[1]}](file-upload://{ids[val[0]]})"); continue
        if kind == "run":
            out.append(f"```python\n{val}\n```"); continue
        lines, i = val.split("\n"), 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("# "): i += 1; continue
            if line.startswith("## 📌 목차"):
                out.append("<table_of_contents/>"); i += 1
                while i < len(lines) and not lines[i].startswith("## "): i += 1
                continue
            if line.startswith("```"):
                j = i + 1
                while not lines[j].startswith("```"): j += 1
                out.append("\n".join(lines[i:j + 1])); i = j + 1; continue
            if line.startswith("|"):
                rows = []
                while i < len(lines) and lines[i].startswith("|"):
                    if not re.match(r"^\|[\s:|-]+\|$", lines[i]):
                        rows.append([c.strip() for c in lines[i].strip().strip("|").split(" | ")])
                    i += 1
                t = ['<table header-row="true">']
                for r in rows:
                    t.append("\t<tr>"); t += [f"\t\t<td>{c}</td>" for c in r]; t.append("\t</tr>")
                t.append("</table>"); out.append("\n".join(t)); continue
            if line.startswith(">"):
                q = []
                while i < len(lines) and lines[i].startswith(">"):
                    q.append(lines[i][1:].strip()); i += 1
                first = q[0]
                m = re.match(r"^([⚠️💡🗣️]+)\s*", first)
                icon = m.group(1) if m else "💡"
                if m: q[0] = first[m.end():]
                color = "red_bg" if icon.startswith("⚠") else "gray_bg" if icon.startswith("🗣") else "blue_bg"
                out.append(f'<callout icon="{icon}" color="{color}">\n' + "\n".join("\t" + x for x in q if x) + "\n</callout>")
                continue
            if line.strip(): out.append(line)
            i += 1
    return "\n".join(out)


if __name__ == "__main__":
    items = blocks(SRC)
    if "--notion" in sys.argv:
        ids = json.loads(Path(sys.argv[sys.argv.index("--notion") + 1]).read_text(encoding="utf-8"))
        Path(sys.argv[sys.argv.index("--notion") + 2]).write_text(to_notion(items, ids), encoding="utf-8")
        print("notion ok"); sys.exit()
    cells = build_notebook(items)
    doc = build_markdown(items, cells)
    build_pdf(doc)
    print("built", len(cells), "cells")
