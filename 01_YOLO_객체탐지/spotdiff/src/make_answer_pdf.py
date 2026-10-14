"""도시 틀린그림찾기 데이터셋을 보기 편한 PDF로 만든다 (문제 페이지 → 정답 페이지 순서).

    python spotdiff/src/make_answer_pdf.py

결과: spotdiff/dataset_city/도시_틀린그림찾기_문제와정답.pdf
"""
import html
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "dataset_city"
OUT = DS / "도시_틀린그림찾기_문제와정답.pdf"
sys.path.insert(0, str(ROOT.parent / "tools"))
from make_pdf import EDGE_CANDIDATES, print_pdf  # noqa: E402

KIND_KO = {"remove": "사라짐", "move": "이동", "copy": "추가"}
CSS = """
<style>
  @page { size: A4 landscape; margin: 10mm; }
  body { font-family: 'Malgun Gothic', sans-serif; color: #1f2328; margin: 0; }
  .page { page-break-after: always; height: 188mm; display: flex; flex-direction: column; }
  .page:last-child { page-break-after: auto; }
  h1 { font-size: 20pt; margin: 0 0 4mm; }
  h2 { font-size: 15pt; margin: 0 0 3mm; }
  .sub { color: #57606a; font-size: 10pt; margin-bottom: 4mm; }
  .pair { display: flex; gap: 6mm; flex: 1; min-height: 0; }
  .pair figure { flex: 1; margin: 0; display: flex; flex-direction: column; min-height: 0; }
  .pair img { width: 100%; height: 100%; object-fit: contain; min-height: 0; }
  figcaption { font-weight: bold; text-align: center; margin-bottom: 2mm; }
  .ans { flex: 1; min-height: 0; display: flex; }
  .ans img { width: 100%; height: 100%; object-fit: contain; }
  table { border-collapse: collapse; font-size: 10.5pt; margin-top: 3mm; width: 100%; }
  th, td { border: 1px solid #d0d7de; padding: 3px 8px; text-align: left; }
  th { background: #f6f8fa; }
  .tag { display: inline-block; padding: 1px 8px; border-radius: 10px; color: #fff; font-size: 9.5pt; }
  .remove { background: #cf222e; } .move { background: #0969da; } .copy { background: #1a7f37; }
  .cover td, .cover th { font-size: 11pt; }
</style>
"""


def where(box, w, h):
    cx, cy = (box[0] + box[2]) / 2 / w, (box[1] + box[3]) / 2 / h
    return ["위", "가운데", "아래"][min(2, int(cy * 3))] + " " + ["왼쪽", "가운데", "오른쪽"][min(2, int(cx * 3))]


def main():
    answers = json.loads((DS / "answers.json").read_text(encoding="utf-8"))
    kinds = [d["kind"] for v in answers.values() for d in v["differences"]]
    base = DS.as_uri() + "/"

    rows = "".join(
        f"<tr><td>{q}</td><td>{len(v['differences'])}개</td><td>"
        + ", ".join(f"{KIND_KO[d['kind']]}({d['class']})" for d in v["differences"]) + "</td></tr>"
        for q, v in answers.items())
    pages = [f"""<div class="page cover">
<h1>도시 틀린그림찾기 — 문제와 정답</h1>
<div class="sub">COCO 도시 사진을 편집해 만든 틀린그림찾기 {len(answers)}문제 · 차이 총 {len(kinds)}개
(사라짐 {kinds.count('remove')}, 이동 {kinds.count('move')}, 추가 {kinds.count('copy')})</div>
<p>각 문제는 <b>문제 페이지</b>(사진 A와 B)와 바로 다음 <b>정답 페이지</b>로 되어 있다.
정답 페이지에는 바뀐 곳에 빨간 박스와 번호가 표시돼 있고, 아래 표에 종류·물체·위치를 적었다.</p>
<p><span class="tag remove">사라짐</span> A에 있던 물체가 B에서 없어짐 &nbsp;
<span class="tag move">이동</span> 물체가 다른 자리로 옮겨짐 &nbsp;
<span class="tag copy">추가</span> B에 없던 물체가 새로 생김</p>
<table><tr><th>문제</th><th>차이 수</th><th>정답 요약 (먼저 보고 싶지 않으면 건너뛰기)</th></tr>{rows}</table>
</div>"""]

    for q, v in answers.items():
        w, h = v["size"]
        pages.append(f"""<div class="page"><h2>{q} — 문제 ({len(v['differences'])}곳이 다릅니다)</h2>
<div class="pair"><figure><figcaption>A (원본)</figcaption><img src="A/{q}.jpg"></figure>
<figure><figcaption>B</figcaption><img src="B/{q}.jpg"></figure></div></div>""")
        trs = []
        for d in v["differences"]:
            if d["kind"] == "remove":
                pos = f"A의 {where(d['box_a'], w, h)}에 있던 것이 B에서 없어짐"
            elif d["kind"] == "move":
                pos = f"A의 {where(d['box_a'], w, h)} → B의 {where(d['box_b'], w, h)}"
            else:
                pos = f"B의 {where(d['box_b'], w, h)}에 새로 생김"
            trs.append(f"<tr><td>{d['no']}</td><td><span class='tag {d['kind']}'>{KIND_KO[d['kind']]}</span></td>"
                       f"<td>{html.escape(d['class'])}</td><td>{pos}</td></tr>")
        pages.append(f"""<div class="page"><h2>{q} — 정답</h2>
<div class="ans"><img src="answer/{q}.jpg"></div>
<table><tr><th>번호</th><th>종류</th><th>물체</th><th>위치</th></tr>{''.join(trs)}</table></div>""")

    doc = f'<!doctype html><html><head><meta charset="utf-8"><base href="{base}">{CSS}</head><body>{"".join(pages)}</body></html>'
    edge = next(p for p in EDGE_CANDIDATES if p.exists())
    tmp = ROOT.parent / "build" / "spot_pdf"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    (tmp / "doc.html").write_text(doc, encoding="utf-8")
    print_pdf(edge, tmp / "doc.html", tmp / "doc.pdf")
    shutil.copy2(tmp / "doc.pdf", OUT)
    print("완료 →", OUT)


if __name__ == "__main__":
    main()
