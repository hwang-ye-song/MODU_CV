"""프로젝트 공통 설정: 경로, 클래스, 디바이스."""
import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent

# 촬영한 사진과 라벨이 모이는 곳
#   data/raw/images/*.jpg   ← capture.py 로 찍거나 직접 복사
#   data/raw/labels/*.txt   ← label.py 로 박스를 그리면 생성 (YOLO 형식)
# (환경변수 CVYOLO_DATA 로 다른 폴더를 지정할 수 있음)
DATA_DIR = Path(os.environ.get("CVYOLO_DATA", ROOT / "data"))
RAW_IMG_DIR = DATA_DIR / "raw" / "images"
RAW_LBL_DIR = DATA_DIR / "raw" / "labels"
DATASET_DIR = DATA_DIR / "dataset"          # split.py 가 train/val 로 나눠 만드는 폴더
DATA_YAML = DATASET_DIR / "data.yaml"

RUNS_DIR = ROOT / "runs"
RESULTS_DIR = ROOT / "results"
REPORT_IMG_DIR = ROOT / "report" / "images"

# 클래스 번호 = 리스트 순서. 라벨 txt 의 첫 숫자가 이 번호다.
CLASSES = ["normal", "defect"]
# OpenCV 는 BGR 순서: normal=초록, defect=빨강
COLORS = {"normal": (0, 200, 0), "defect": (0, 0, 255)}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

RUN_NAME = "defect-det"
BEST_WEIGHTS = RUNS_DIR / "detect" / RUN_NAME / "weights" / "best.pt"

for d in (RAW_IMG_DIR, RAW_LBL_DIR, RESULTS_DIR, REPORT_IMG_DIR):
    d.mkdir(parents=True, exist_ok=True)


def get_device() -> str:
    """GPU가 있으면 '0', 없으면 'cpu'."""
    return "0" if torch.cuda.is_available() else "cpu"


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("*") if p.suffix.lower() in IMG_EXTS)


def label_path(img: Path, label_dir: Path = RAW_LBL_DIR) -> Path:
    return label_dir / (img.stem + ".txt")
