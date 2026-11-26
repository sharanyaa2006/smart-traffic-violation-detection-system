"""
Phase 7 — Model evaluation.

Runs ultralytics' built-in validation on your helmet model against a
labeled test set, which automatically computes:
  - Precision, Recall, mAP50, mAP50-95 (per class and overall)
  - A confusion matrix (saved as an image)
  - Precision-Recall curves (saved as images)

This is the core evidence for your Phase 7 metrics report — don't hand-
roll this with sklearn, ultralytics' val() already does it properly and
saves plots.

REQUIRES: a data.yaml pointing to a labeled train/val/test split for your
helmet dataset (same format as any YOLO training data.yaml). If you only
have train/val (no separate test split), point --split at "val" instead.

Usage:
    python src/evaluate_model.py --data path/to/data.yaml --split test
"""

import argparse
import os
import sys

from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def evaluate(data_yaml_path, split="test"):
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"Could not find data.yaml at {data_yaml_path}")

    model = YOLO(config.HELMET_MODEL_PATH)

    print("=" * 50)
    print(f"Evaluating {config.HELMET_MODEL_PATH} on split='{split}'")
    print(f"Using dataset config: {data_yaml_path}")
    print("=" * 50)

    metrics = model.val(data=data_yaml_path, split=split)

    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")

    print("\nPer-class results:")
    for i, class_name in model.names.items():
        try:
            print(
                f"  {class_name}: "
                f"Precision={metrics.box.p[i]:.4f}  "
                f"Recall={metrics.box.r[i]:.4f}  "
                f"mAP50={metrics.box.ap50[i]:.4f}"
            )
        except (IndexError, KeyError):
            pass

    print("\nConfusion matrix and PR-curve plots saved to the 'runs/detect/val*' folder")
    print("(ultralytics auto-saves these — copy the relevant PNGs into docs/ for your report)")
    print("=" * 50)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate helmet model with precision/recall/mAP/confusion matrix")
    parser.add_argument("--data", required=True, help="Path to data.yaml for your helmet dataset")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                         help="Which split to evaluate on (default: test)")
    args = parser.parse_args()

    evaluate(args.data, args.split)