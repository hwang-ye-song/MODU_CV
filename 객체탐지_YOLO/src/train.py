"""COCO 사전학습 YOLO11n 을 정상/불량 박스 데이터로 파인튜닝한다.

    python src/train.py
    python src/train.py --epochs 100 --model yolo11s.pt
"""
import argparse

from ultralytics import YOLO

from common import DATA_YAML, RUN_NAME, RUNS_DIR, get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt", help="시작 가중치 (n < s < m 순으로 크고 정확)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="8GB GPU 기준 16")
    ap.add_argument("--name", default=RUN_NAME)
    args = ap.parse_args()

    if not DATA_YAML.exists():
        raise SystemExit("data/dataset/data.yaml 이 없습니다. 먼저 python src/split.py 를 실행하세요.")

    model = YOLO(args.model)
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=get_device(),
        project=str(RUNS_DIR / "detect"),
        name=args.name,
        exist_ok=True,
        workers=2,      # Windows 에서는 작게
        patience=15,    # 15 에폭 동안 좋아지지 않으면 조기 종료
        seed=42,
        plots=True,
    )
    print(f"학습 완료 → {RUNS_DIR / 'detect' / args.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":  # Windows 멀티프로세싱 때문에 반드시 필요
    main()
