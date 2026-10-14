"""학습한 모델 평가.

    python src/evaluate.py test1     # 기본 성능: 박스 지표(mAP) + 사진 단위 OK/NG 판정 정확도
    python src/evaluate.py test2     # 현장 조건 변화(어두움/밝음/흐림/노이즈)에서도 버티는지

결과: results/*.csv, *.json  /  그래프·샘플: report/images/*.png
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from ultralytics import YOLO

from common import (BEST_WEIGHTS, CLASSES, COLORS, DATA_DIR, DATA_YAML, DATASET_DIR,
                    REPORT_IMG_DIR, RESULTS_DIR, RUNS_DIR, get_device, list_images)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFECT_ID = CLASSES.index("defect")
VERDICT_CONF = 0.5   # live.py 기본값과 맞춘다


def box_metrics(model, data_yaml) -> dict:
    """Ultralytics 공식 검증: 클래스별 Precision / Recall / mAP."""
    m = model.val(data=str(data_yaml), device=get_device(), plots=False, verbose=False,
                  project=str(RUNS_DIR / "val"), name="eval", exist_ok=True)
    per_class = {}
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        per_class[CLASSES[int(c)]] = {"precision": round(p, 4), "recall": round(r, 4),
                                      "mAP50": round(ap50, 4), "mAP50-95": round(ap, 4)}
    return {
        "precision": round(float(m.box.mp), 4), "recall": round(float(m.box.mr), 4),
        "mAP50": round(float(m.box.map50), 4), "mAP50-95": round(float(m.box.map), 4),
        "per_class": per_class,
        "ms_per_img": round(sum(m.speed.values()), 2),
    }


def verdicts(model, img_dir: Path, lbl_dir: Path) -> pd.DataFrame:
    """사진 한 장 = 제품 한 판. 불량 박스가 하나라도 있으면 NG.

    정답(라벨)과 예측의 OK/NG 가 같은지, 불량/정상 개수가 같은지 비교한다.
    """
    rows = []
    imgs = list_images(img_dir)
    for r in model.predict([str(p) for p in imgs], conf=VERDICT_CONF, device=get_device(),
                           stream=True, verbose=False):
        name = Path(r.path).name
        lines = [l for l in (lbl_dir / (Path(name).stem + ".txt")).read_text().splitlines() if l.strip()]
        gt = [int(l.split()[0]) for l in lines]
        pred = [int(c) for c in r.boxes.cls.tolist()]
        gt_ng, pred_ng = gt.count(DEFECT_ID), pred.count(DEFECT_ID)
        rows.append({
            "image": name,
            "gt_normal": len(gt) - gt_ng, "gt_defect": gt_ng,
            "pred_normal": len(pred) - pred_ng, "pred_defect": pred_ng,
            "gt_verdict": "NG" if gt_ng else "OK",
            "pred_verdict": "NG" if pred_ng else "OK",
        })
    df = pd.DataFrame(rows)
    df["verdict_correct"] = df.gt_verdict == df.pred_verdict
    df["count_correct"] = (df.gt_normal == df.pred_normal) & (df.gt_defect == df.pred_defect)
    return df


def verdict_summary(df: pd.DataFrame) -> dict:
    ng = df[df.gt_verdict == "NG"]
    ok = df[df.gt_verdict == "OK"]
    return {
        "images": len(df),
        "verdict_acc": round(df.verdict_correct.mean(), 4),
        "count_acc": round(df.count_correct.mean(), 4),
        # 제조에서 가장 치명적: 불량인데 OK 로 통과 (미검출, escape)
        "missed_NG": int((ng.pred_verdict == "OK").sum()), "gt_NG": len(ng),
        # 비용 문제: 정상인데 NG 로 버림 (과검출, overkill)
        "false_NG": int((ok.pred_verdict == "NG").sum()), "gt_OK": len(ok),
    }


def save_samples(model, img_dir: Path, lbl_dir: Path, out: Path, n: int = 6):
    """정답(왼쪽) vs 예측(오른쪽) 샘플 그리드."""
    imgs = list_images(img_dir)[:n]
    fig, axes = plt.subplots(len(imgs), 2, figsize=(10, 3.4 * len(imgs)), squeeze=False)
    for i, p in enumerate(imgs):
        gt = cv2.imread(str(p))
        h, w = gt.shape[:2]
        for line in (lbl_dir / (p.stem + ".txt")).read_text().splitlines():
            if line.strip():
                c, cx, cy, bw, bh = map(float, line.split()[:5])
                color = COLORS[CLASSES[int(c)]]
                cv2.rectangle(gt, (int((cx - bw / 2) * w), int((cy - bh / 2) * h)),
                              (int((cx + bw / 2) * w), int((cy + bh / 2) * h)), color, 3)
        pred = model.predict(str(p), conf=VERDICT_CONF, device=get_device(), verbose=False)[0].plot()
        for j, (title, im) in enumerate((("ground truth", gt), ("prediction", pred))):
            axes[i, j].imshow(im[:, :, ::-1])
            axes[i, j].set_title(f"{title} | {p.name}", fontsize=8)
            axes[i, j].axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    plt.close()


def run_test1():
    model = YOLO(str(BEST_WEIGHTS))
    metrics = box_metrics(model, DATA_YAML)
    img_dir, lbl_dir = DATASET_DIR / "images" / "val", DATASET_DIR / "labels" / "val"
    df = verdicts(model, img_dir, lbl_dir)
    df.to_csv(RESULTS_DIR / "test1_per_image.csv", index=False)
    metrics["verdict"] = verdict_summary(df)
    (RESULTS_DIR / "test1_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    save_samples(model, img_dir, lbl_dir, REPORT_IMG_DIR / "test1_samples.png")

    # 학습 곡선 / 혼동행렬은 학습 폴더에 이미 있으므로 보고서 폴더로 복사
    run_dir = BEST_WEIGHTS.parent.parent
    for f in ("results.png", "confusion_matrix.png", "BoxPR_curve.png"):
        if (run_dir / f).exists():
            shutil.copy2(run_dir / f, REPORT_IMG_DIR / f"train_{f}")

    print("\n[test 1] 기본 성능 (val)")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


# ---- test 2: 현장 조건 변화 --------------------------------------------------
def _dark(im):   return cv2.convertScaleAbs(im, alpha=0.45, beta=0)
def _bright(im): return cv2.convertScaleAbs(im, alpha=1.3, beta=60)
def _blur(im):   return cv2.GaussianBlur(im, (15, 15), 0)          # 초점 흐림 / 움직임


def _noise(im):
    n = np.random.default_rng(0).normal(0, 25, im.shape)             # 저조도 센서 노이즈
    return np.clip(im.astype(np.float32) + n, 0, 255).astype(np.uint8)


CONDITIONS = {"original": None, "dark": _dark, "bright": _bright, "blur": _blur, "noise": _noise}


def run_test2():
    model = YOLO(str(BEST_WEIGHTS))
    src_img, src_lbl = DATASET_DIR / "images" / "val", DATASET_DIR / "labels" / "val"
    root = DATA_DIR / "robust"
    rows = []
    for cond, fn in CONDITIONS.items():
        d = root / cond
        if d.exists():
            shutil.rmtree(d)
        (d / "images" / "val").mkdir(parents=True)
        shutil.copytree(src_lbl, d / "labels" / "val")
        for p in list_images(src_img):
            im = cv2.imread(str(p))
            cv2.imwrite(str(d / "images" / "val" / p.name), fn(im) if fn else im)
        yaml = d / "data.yaml"
        names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
        yaml.write_text(f"path: {d.as_posix()}\ntrain: images/val\nval: images/val\nnames:\n{names}\n")

        m = box_metrics(model, yaml)
        v = verdict_summary(verdicts(model, d / "images" / "val", d / "labels" / "val"))
        rows.append({"condition": cond, "mAP50": m["mAP50"], "mAP50-95": m["mAP50-95"],
                     "defect_recall": m["per_class"].get("defect", {}).get("recall"),
                     "verdict_acc": v["verdict_acc"], "missed_NG": v["missed_NG"], "false_NG": v["false_NG"]})
        if cond != "original":
            save_samples(model, d / "images" / "val", d / "labels" / "val",
                         REPORT_IMG_DIR / f"test2_{cond}.png", n=2)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "test2_robustness.csv", index=False)

    ax = df.plot.bar(x="condition", y=["mAP50", "defect_recall", "verdict_acc"], rot=0,
                     color=["#94a3b8", "#ef4444", "#3b82f6"], figsize=(8, 4))
    ax.set_ylim(0, 1.08)
    ax.set_title("Robustness under shop-floor conditions")
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.2f", fontsize=7)
    plt.tight_layout()
    plt.savefig(REPORT_IMG_DIR / "test2_robustness.png", dpi=150)
    plt.close()

    print("\n[test 2] 조건 변화별 성능")
    print(df.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["test1", "test2"])
    {"test1": run_test1, "test2": run_test2}[ap.parse_args().task]()
