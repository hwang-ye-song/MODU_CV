"""코랩용 학습 노트북(알박기_yolo26_ml_학습.ipynb)을 만든다."""

import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1])


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


cells = [
    md("""
# 알박기 감지: YOLO26 m·l 파인튜닝 (코랩)

노트북(로컬)에서 yolo11s·yolo26s로 학습한 것과 **같은 데이터, 같은 설정**으로 yolo26m, yolo26l을 학습합니다.

- 데이터: `내 드라이브/알박기_학습/dataset_v2_colab.zip` (학습 637장, 검증 90장)
- 설정: epochs 25, imgsz 960, batch 8, AdamW, lr0 0.001 (로컬과 동일)
- 결과 저장: `내 드라이브/알박기_학습/runs/` (에포크마다 드라이브에 저장되므로 연결이 끊겨도 남아 있음)

**사용법**
1. 메뉴 `런타임 → 런타임 유형 변경`에서 GPU를 **A100**으로 고르기
2. `런타임 → 모두 실행`
3. 중간에 연결이 끊기면 다시 `모두 실행`만 누르면 **끊긴 에포크부터 이어서** 학습합니다
"""),
    md("## 1. GPU 확인"),
    code("!nvidia-smi --query-gpu=name,memory.total --format=csv"),
    md("## 2. 구글 드라이브 연결"),
    code("""
from google.colab import drive
drive.mount('/content/drive')

import os, glob
DRIVE_DIR = '/content/drive/MyDrive/알박기_학습'
# zip 파일 하나로 올렸으면 그걸 쓰고, 11조각으로 올렸으면 조각을 합친다
ZIP = f'{DRIVE_DIR}/dataset_v2_colab.zip'
PARTS = [ZIP] if os.path.exists(ZIP) else sorted(glob.glob(f'{ZIP}.part*'))
assert PARTS == [ZIP] or len(PARTS) == 11, '드라이브 알박기_학습 폴더에 dataset_v2_colab.zip 이 없습니다'
print('OK', PARTS[0] if len(PARTS) == 1 else f'{len(PARTS)}조각')
"""),
    md("## 3. 라이브러리 설치 (로컬과 같은 버전)"),
    code("!pip install -q ultralytics==8.4.176"),
    md("## 4. 데이터셋 풀기\n\n드라이브에서 바로 읽으면 느려서 코랩 디스크(`/content`)로 복사하고, 원본과 같은지 md5로 확인한 뒤 풉니다."),
    code("""
import os, hashlib
if not os.path.exists('/content/dataset_v2/data.yaml'):
    with open('/content/dataset_v2_colab.zip', 'wb') as out:
        for p in PARTS:
            out.write(open(p, 'rb').read())
    md5 = hashlib.md5(open('/content/dataset_v2_colab.zip', 'rb').read()).hexdigest()
    assert md5 == 'b63b0d083a986eeb6b3833a84028f861', f'합친 파일이 원본과 다릅니다 ({md5}). 조각을 다시 올려 주세요'
    print('md5 OK')
    !unzip -q /content/dataset_v2_colab.zip -d /content/
!ls /content/dataset_v2/images/train | wc -l
!ls /content/dataset_v2/images/val | wc -l
!head -3 /content/dataset_v2/data.yaml
"""),
    md("""
## 5. 학습 (yolo26m → yolo26l 순서)

- 이미 끝난 모델은 건너뜁니다 (`DONE` 파일로 확인).
- 중간에 끊긴 모델은 드라이브에 남은 `last.pt`에서 이어서 학습합니다.
"""),
    code("""
import os
from ultralytics import YOLO

PROJECT = f'{DRIVE_DIR}/runs'
SETTINGS = dict(data='/content/dataset_v2/data.yaml', epochs=25, imgsz=960, batch=8, workers=2,
                optimizer='AdamW', lr0=0.001, seed=0, project=PROJECT, exist_ok=True)

for size in ['m', 'l']:
    name = f'v2_yolo26{size}'
    run_dir = f'{PROJECT}/{name}'
    last = f'{run_dir}/weights/last.pt'
    if os.path.exists(f'{run_dir}/DONE'):
        print(f'{name}: 이미 끝남, 건너뜀')
        continue
    if os.path.exists(last):
        print(f'{name}: 끊긴 지점부터 이어서 학습')
        try:
            YOLO(last).train(resume=True)
        except AssertionError as e:
            # 마지막 에포크까지 끝났는데 DONE을 쓰기 전에 끊긴 경우
            if 'finished' not in str(e):
                raise
            print(f'{name}: 학습은 이미 끝나 있었음')
    else:
        print(f'{name}: 처음부터 학습')
        YOLO(f'yolo26{size}.pt').train(name=name, **SETTINGS)
    open(f'{run_dir}/DONE', 'w').write('done')
    print(f'{name}: 끝')
"""),
    md("## 6. 결과 확인 (검증 데이터 mAP)"),
    code("""
from ultralytics import YOLO

for size in ['m', 'l']:
    w = f'{PROJECT}/v2_yolo26{size}/weights/best.pt'
    r = YOLO(w).val(data='/content/dataset_v2/data.yaml', imgsz=960, batch=8, verbose=False, plots=False)
    names = list(r.box.ap_class_index)
    table = r.box.ap50[names.index(60)] if 60 in names else float('nan')
    person = r.box.ap50[names.index(0)] if 0 in names else float('nan')
    print(f'yolo26{size}: 전체 mAP50 {r.box.map50:.3f} | 사람 {person:.3f} | 테이블 {table:.3f}')
"""),
    md("""
## 7. 완성된 모델 파일 모으기

`내 드라이브/알박기_학습/모델/` 폴더에 이름을 붙여 복사합니다.
로컬 노트북의 `G:\\내 드라이브\\알박기_학습\\모델\\` 에 자동으로 동기화되므로, 그 파일을 평가에 쓰면 됩니다.
"""),
    code("""
import os, shutil
os.makedirs(f'{DRIVE_DIR}/모델', exist_ok=True)
for size in ['m', 'l']:
    src = f'{PROJECT}/v2_yolo26{size}/weights/best.pt'
    dst = f'{DRIVE_DIR}/모델/알박기_yolo26{size}_v2.pt'
    shutil.copy(src, dst)
    print(dst, round(os.path.getsize(dst) / 1e6, 1), 'MB')
"""),
]

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("saved", OUT)
