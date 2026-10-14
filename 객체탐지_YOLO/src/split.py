"""라벨이 있는 사진만 골라 train/val 로 나누고 data.yaml 을 만든다.

    python src/split.py              # val 20%
    python src/split.py --val 0.3

만들어지는 구조 (YOLO 탐지 학습용):
    dataset/images/train/*.jpg   dataset/labels/train/*.txt
    dataset/images/val/*.jpg     dataset/labels/val/*.txt
    dataset/data.yaml
"""
import argparse
import random
import shutil
from collections import Counter

from common import CLASSES, DATA_YAML, DATASET_DIR, RAW_IMG_DIR, label_path, list_images


def count_boxes(label_files):
    cnt = Counter()
    for f in label_files:
        for line in f.read_text().splitlines():
            if line.strip():
                cnt[CLASSES[int(line.split()[0])]] += 1
    return cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=float, default=0.2, help="검증용 비율")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    images = [p for p in list_images(RAW_IMG_DIR) if label_path(p).exists()]
    if len(images) < 5:
        raise SystemExit(f"라벨된 사진이 {len(images)}장뿐입니다. python src/label.py 로 먼저 라벨링하세요.")

    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)  # 매번 raw 에서 새로 만든다

    random.Random(args.seed).shuffle(images)
    n_val = max(1, round(len(images) * args.val))
    for split, files in (("val", images[:n_val]), ("train", images[n_val:])):
        (DATASET_DIR / "images" / split).mkdir(parents=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True)
        for img in files:
            shutil.copy2(img, DATASET_DIR / "images" / split / img.name)
            shutil.copy2(label_path(img), DATASET_DIR / "labels" / split / label_path(img).name)
        boxes = count_boxes([DATASET_DIR / "labels" / split / label_path(i).name for i in files])
        print(f"{split:>5}: 사진 {len(files):4d}장 | 박스 {dict(boxes)}")

    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    DATA_YAML.write_text(
        f"path: {DATASET_DIR.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n",
        encoding="utf-8",
    )
    print(f"완료 → {DATA_YAML}")


if __name__ == "__main__":
    main()
