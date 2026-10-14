"""최종 확인 노트북: 실습 직접 해보기(1부) + 틀린그림찾기 노트북(2부)을 하나로 묶고 PDF도 만든다.

    python tools/make_final_notebook.py

결과
- 최종확인_노트북.ipynb : 두 노트북을 합친 것 (GitHub에서 열리도록 이미지 압축)
- 최종확인_노트북.pdf   : 표지 + 1부 + 2부 + 개인 회고

각 부의 실행 결과는 원래 노트북을 실행한 결과를 그대로 옮긴 것이다.
(1부는 학습이 들어 있어 다시 실행하면 시간이 걸린다.)
"""
import copy
import shutil
import sys
from pathlib import Path

import nbformat
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import make_pdf  # noqa: E402
import shrink_notebook  # noqa: E402

PRACTICE = ROOT / "notebooks" / "실습_직접해보기.ipynb"
SPOTDIFF = ROOT / "spotdiff" / "spotdiff_yolo.ipynb"
SUBMISSION = ROOT / "notebooks" / "yolov8_manufacturing_practice_original.ipynb"   # 개인 실험 1을 가져올 곳
RETRO_MD = ROOT / "report" / "개인회고.md"          # 맨 끝 개인 회고
OUT_NB = ROOT / "최종확인_노트북.ipynb"
OUT_PDF = ROOT / "최종확인_노트북.pdf"
PART1 = "1부. YOLOv8 제조 데이터 객체 탐지 실습 — 직접 해보기"
PART2 = "2부. YOLO로 틀린그림찾기"
RETRO = "개인 회고"

INTRO = f"""# 최종 확인 노트북

과제 실습과 직접 만든 틀린그림찾기를 **한 노트북에 모은 최종 확인용 노트북**이다.

| 부 | 내용 | 원래 노트북 |
|---|---|---|
| {PART1} | 과제 원본 코드(버스 이미지 추론, stamp 데이터 학습·평가)를 11개 흐름으로 나눠 직접 실행하고, 흐름마다 이해한 내용을 **✍️ 내 코멘트** 칸에 직접 적었다. 흐름 8 뒤에는 **🔬 개인 실험 1**(신뢰도 임계값, 추가 샘플 zidane으로 모델 비교, IoU·NMS)을 넣었다 | `notebooks/실습_직접해보기.ipynb` |
| {PART2} | Gemini로 문제 그림 5장을 만들고, YOLO로 두 그림의 물체를 찾아 비교하는 탐지기를 만들어 채점했다 (재현율 65.7%, 정밀도 100%) | `spotdiff/spotdiff_yolo.ipynb` |

각 부의 실행 결과는 원래 노트북을 실행한 결과를 그대로 옮긴 것이다. 1부는 학습이 들어 있어 처음부터 다시 실행하면 시간이 걸린다.
각 부 앞의 작업 폴더 이동 칸은 이 노트북을 레포 맨 위에서 다시 실행할 때를 위한 것이다.
틀린그림찾기를 만들며 겪은 시행착오는 `spotdiff/report/작업기록.md`에 따로 정리했다.
개인 회고(막혔던 부분과 해결, KPT, AAR)와 실습 전체 정리는 맨 끝에 한 번에 적었다.
"""


def part_cells(src: Path, title: str, workdir: str):
    nb = nbformat.read(src, as_version=4)
    cells = [copy.deepcopy(c) for c in nb.cells]
    # 원래 제목 줄을 'n부' 제목으로 바꾼다
    first = cells[0]
    lines = first.source.splitlines()
    first.source = "\n".join([f"# {title}"] + lines[1:])
    chdir = nbformat.v4.new_code_cell(
        f"# 이 부의 코드는 {workdir}/ 폴더 기준 경로를 쓴다\n"
        "import os\n"
        "from pathlib import Path\n"
        "REPO = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'spotdiff').is_dir())   # 레포 맨 위\n"
        f"os.chdir(REPO / '{workdir}')")
    if workdir == "spotdiff":   # 최종 노트북은 레포 맨 위에 있으므로 링크 경로를 맞춘다
        for c in cells:
            if c.cell_type == "markdown":
                c.source = c.source.replace("](report/", "](spotdiff/report/")
    return [first, chdir] + cells[1:]


def renumber(nb):
    """두 노트북의 셀 id가 겹치지 않게 새로 붙인다."""
    for i, c in enumerate(nb.cells):
        c["id"] = f"cell-{i:03d}"


def exp1_cells():
    """제출용 실습 노트북의 '🔬 개인 실험 1' 부분(신뢰도, 모델 비교, IoU·NMS)을 그대로 가져온다."""
    nb = nbformat.read(SUBMISSION, as_version=4)
    is_title = lambda c, t: c.cell_type == "markdown" and any(l.lstrip("# ").startswith(t) for l in c.source.splitlines() if l.startswith("#"))
    start = next(i for i, c in enumerate(nb.cells) if is_title(c, "🔬 개인 실험 1"))       # 목차 글자가 아닌 제목 줄로 찾는다
    end = next(i for i, c in enumerate(nb.cells) if i > start and is_title(c, "Part 2"))
    return [copy.deepcopy(c) for c in nb.cells[start:end]]


def retro_cells():
    """맨 끝의 개인 회고(막혔던 부분과 해결, 실습 전체 정리, KPT, AAR)를 report/개인회고.md에서 가져온다."""
    text = RETRO_MD.read_text(encoding="utf-8").strip()
    parts = text.split("\n### ")
    return [nbformat.v4.new_markdown_cell(parts[0])] + [nbformat.v4.new_markdown_cell("### " + p) for p in parts[1:]]


def build():
    p1 = part_cells(PRACTICE, PART1, "notebooks")
    # 개인 실험 1은 사전학습 모델(model)과 버스 사진(img_np)을 쓰므로, 학습(흐름 9) 전에 넣는다
    at = next(i for i, c in enumerate(p1) if c.cell_type == "markdown" and c.source.startswith("## 흐름 9"))
    p1 = p1[:at] + exp1_cells() + p1[at:]
    # 직접 쓴 '전체 정리'는 report/개인회고.md에 옮겨 두었으므로 1부에서는 빼고, 빈 칸도 뺀다
    p1 = [c for c in p1 if not (c.cell_type == "markdown" and "## 전체 정리" in c.source) and c.source.strip()]
    p2 = part_cells(SPOTDIFF, PART2, "spotdiff")
    base = nbformat.read(SPOTDIFF, as_version=4)
    full = nbformat.v4.new_notebook(metadata=base.metadata)
    ending = retro_cells()
    full.cells = [nbformat.v4.new_markdown_cell(INTRO)] + p1 + p2 + ending
    renumber(full)
    return full, p1, p2, ending


def make_pdf_file(p1, p2, ending):
    tmp = ROOT / "build" / "final_pdf"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    make_pdf.TITLE = "최종 확인 노트북<br>YOLO 실습 + 틀린그림찾기"
    make_pdf.SUBTITLE = "과제 실습(제조 데이터 탐지·개인 실험·회고) → 직접 만든 YOLO 틀린그림찾기 탐지기"
    sections = []
    for i, (title, cells) in enumerate([(PART1, [nbformat.v4.new_markdown_cell(INTRO)] + p1), (PART2, p2), (RETRO, ending)]):
        nb = nbformat.v4.new_notebook(cells=[copy.deepcopy(c) for c in cells])
        renumber(nb)
        path = tmp / f"part{i + 1}.ipynb"
        nbformat.write(nb, path)
        sections.append((title, make_pdf.notebook_html(path)))
    edge = next(p for p in make_pdf.EDGE_CANDIDATES if p.exists())
    pages = [("표지", make_pdf.cover_html([t for t, _ in sections]))] + sections
    writer = PdfWriter()
    for i, (title, doc) in enumerate(pages):
        h, p = tmp / f"{i:02d}.html", tmp / f"{i:02d}.pdf"
        h.write_text(doc, encoding="utf-8")
        make_pdf.print_pdf(edge, h, p)
        start = len(writer.pages)
        writer.append(PdfReader(p))
        writer.add_outline_item(title, start)
        print(f"{title}: {len(writer.pages) - start}쪽")
    writer.add_metadata({"/Title": "최종 확인 노트북 — YOLO 실습 + 틀린그림찾기", "/Author": make_pdf.AUTHOR})
    with open(OUT_PDF, "wb") as f:
        writer.write(f)
    print(f"PDF → {OUT_PDF} (총 {len(writer.pages)}쪽)")


def main():
    full, p1, p2, ending = build()
    make_pdf_file(p1, p2, ending)
    nbformat.write(full, OUT_NB)
    # GitHub 웹에서 열리도록 그림을 줄인다 (두 노트북 분량이라 기존보다 더 작게)
    shrink_notebook.MAX_W, shrink_notebook.QUALITY = 720, 60
    shrink_notebook.main(str(OUT_NB))
    print(f"노트북 → {OUT_NB}")


if __name__ == "__main__":
    main()
