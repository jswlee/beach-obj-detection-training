"""
General-purpose utilities shared across the training and inference scripts.

Provides:
  - patch_torch_load()        – Fix for PyTorch 2.6+ weights_only default
  - select_device()           – Pick best available device (cuda > mps > cpu)
  - save_training_summary()   – Write a detailed training report to disk
  - save_test_summary()       – Write a test-set evaluation report to disk
  - collect_images()          – Gather image paths from a directory
"""

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import yaml


# ---------------------------------------------------------------------------
# PyTorch compatibility
# ---------------------------------------------------------------------------

def patch_torch_load() -> None:
    """Monkey-patch torch.load so it defaults to weights_only=False.

    Starting with PyTorch 2.6 the default changed to weights_only=True,
    which breaks loading Ultralytics .pt checkpoints that contain
    arbitrary Python objects.  This patch restores the old behaviour.
    """
    _original_load = torch.load

    def patched_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _original_load(*args, **kwargs)

    torch.load = patched_load


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def select_device() -> str:
    """Return the best available compute device string for Ultralytics.

    Priority: CUDA → MPS (Apple Silicon) → CPU.
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Image file discovery
# ---------------------------------------------------------------------------

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect_images(input_dir: Path) -> list[Path]:
    """Return a sorted list of image file paths found in *input_dir*."""
    return sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )


# ---------------------------------------------------------------------------
# Training summary
# ---------------------------------------------------------------------------

def save_training_summary(
    run_dir: Path,
    model_path: Path | str,
    start_time: datetime,
    end_time: datetime,
) -> Path:
    """Write a comprehensive training report to ``run_dir/training_summary.txt``.

    Reads the actual training config from Ultralytics' ``args.yaml`` and
    epoch-by-epoch metrics from ``results.csv``.

    Returns:
        Path to the written summary file.
    """
    summary_file = run_dir / "training_summary.txt"

    # Load training config written by Ultralytics
    args_file = run_dir / "args.yaml"
    training_config: dict = {}
    if args_file.exists():
        with open(args_file, "r") as f:
            training_config = yaml.safe_load(f)

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("YOLO TRAINING SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        # Timestamps
        f.write("TRAINING TIMESTAMPS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Start Time:    {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End Time:      {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duration:      {str(end_time - start_time).split('.')[0]}\n\n")

        # Training config
        if training_config:
            f.write("TRAINING CONFIGURATION (actual values used)\n")
            f.write("-" * 80 + "\n")
            important_params = [
                "model", "data", "epochs", "batch", "imgsz", "device",
                "optimizer", "lr0", "lrf", "momentum", "weight_decay",
                "warmup_epochs", "warmup_momentum", "warmup_bias_lr",
                "box", "cls", "dfl", "freeze",
                "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale",
                "shear", "perspective", "flipud", "fliplr",
                "mosaic", "mixup", "copy_paste",
                "patience", "close_mosaic",
            ]
            for key in important_params:
                if key in training_config:
                    f.write(f"{key:20s}: {training_config[key]}\n")
            f.write("\n")

        # Model info
        f.write("MODEL INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Model Path:          {model_path}\n")
        mp = Path(model_path)
        if mp.exists():
            f.write(f"Model Size:          {os.path.getsize(mp) / (1024**2):.2f} MB\n")
        f.write("\n")

        # Results CSV
        csv_path = run_dir / "results.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                df.columns = df.columns.str.strip()

                f.write("TRAINING METRICS SUMMARY\n")
                f.write("-" * 80 + "\n")

                if "metrics/mAP50-95(B)" in df.columns:
                    best_idx = df["metrics/mAP50-95(B)"].idxmax()
                    best = df.loc[best_idx]
                    f.write(f"Best Epoch:          {int(best['epoch'])}\n")
                    f.write(f"Best mAP50-95:       {best['metrics/mAP50-95(B)']:.4f}\n")
                    f.write(f"Best mAP50:          {best['metrics/mAP50(B)']:.4f}\n")
                    f.write(f"Best Precision:      {best['metrics/precision(B)']:.4f}\n")
                    f.write(f"Best Recall:         {best['metrics/recall(B)']:.4f}\n\n")

                final = df.iloc[-1]
                f.write(f"Final Epoch:         {int(final['epoch'])}\n")
                f.write(f"Final mAP50-95:      {final['metrics/mAP50-95(B)']:.4f}\n")
                f.write(f"Final mAP50:         {final['metrics/mAP50(B)']:.4f}\n")
                f.write(f"Final Precision:     {final['metrics/precision(B)']:.4f}\n")
                f.write(f"Final Recall:        {final['metrics/recall(B)']:.4f}\n\n")

                # Loss values
                f.write("FINAL LOSS VALUES\n")
                f.write("-" * 80 + "\n")
                for col_label, display in [
                    ("train/box_loss", "Train Box Loss"),
                    ("train/cls_loss", "Train Class Loss"),
                    ("train/dfl_loss", "Train DFL Loss"),
                    ("val/box_loss", "Val Box Loss"),
                    ("val/cls_loss", "Val Class Loss"),
                    ("val/dfl_loss", "Val DFL Loss"),
                ]:
                    if col_label in df.columns:
                        f.write(f"{display:20s}: {final[col_label]:.4f}\n")
                f.write("\n")

                # Epoch-by-epoch table
                f.write("EPOCH-BY-EPOCH RESULTS\n")
                f.write("=" * 80 + "\n\n")
                cols_to_show = [
                    "epoch", "train/box_loss", "train/cls_loss", "train/dfl_loss",
                    "metrics/precision(B)", "metrics/recall(B)",
                    "metrics/mAP50(B)", "metrics/mAP50-95(B)",
                ]
                cols_available = [c for c in cols_to_show if c in df.columns]
                header = "  ".join(f"{c:>18s}" for c in cols_available)
                f.write(header + "\n")
                f.write("-" * len(header) + "\n")
                for _, row in df.iterrows():
                    values = "  ".join(
                        f"{row[c]:>18.6f}" if pd.notna(row[c]) else f"{'N/A':>18s}"
                        for c in cols_available
                    )
                    f.write(values + "\n")
            except Exception as e:
                f.write(f"Error reading results CSV: {e}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF TRAINING SUMMARY\n")
        f.write("=" * 80 + "\n")

    print(f"✓ Training summary saved to: {summary_file}")
    return summary_file


# ---------------------------------------------------------------------------
# Test summary
# ---------------------------------------------------------------------------

def save_test_summary(
    run_dir: Path,
    model_path: Path | str,
    data_yaml: str,
    img_size: int,
    device: str,
    results,
) -> Path:
    """Write a test-set evaluation report to ``run_dir/test_summary.txt``.

    Args:
        run_dir:    Directory where test results are saved.
        model_path: Path to the evaluated model.
        data_yaml:  Dataset YAML path.
        img_size:   Inference image size.
        device:     Device used (cuda / mps / cpu).
        results:    Ultralytics Results object from ``model.val``.

    Returns:
        Path to the written summary file.
    """
    summary_file = run_dir / "test_summary.txt"
    metrics = getattr(results, "metrics", None)
    names = getattr(results, "names", None)

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("YOLO TEST SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write("TEST CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Model:             {model_path}\n")
        f.write(f"Dataset:           {data_yaml}\n")
        f.write(f"Image size:        {img_size}\n")
        f.write(f"Device:            {device}\n")
        f.write(f"Results dir:       {run_dir}\n\n")

        if metrics is not None:
            f.write("OVERALL METRICS\n")
            f.write("-" * 80 + "\n")
            try:
                f.write(f"mAP50-95 (all):   {metrics.map:0.4f}\n")
                f.write(f"mAP50 (all):      {metrics.map50:0.4f}\n")
                f.write(f"Precision (all):  {metrics.precision.mean():0.4f}\n")
                f.write(f"Recall (all):     {metrics.recall.mean():0.4f}\n")
            except Exception:
                f.write("Could not read overall metrics.\n")
            f.write("\n")

            if names is not None:
                try:
                    f.write("PER-CLASS METRICS\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"{'Class':<15s} {'Prec':>8s} {'Rec':>8s} {'mAP50':>10s} {'mAP50-95':>10s}\n")
                    f.write("-" * 80 + "\n")
                    for cls_id, cls_name in names.items():
                        map50_95 = float(metrics.ap[cls_id].mean()) if metrics.ap is not None else float("nan")
                        map50 = float(metrics.ap50[cls_id]) if getattr(metrics, "ap50", None) is not None else float("nan")
                        prec = float(metrics.precision[cls_id]) if metrics.precision is not None else float("nan")
                        rec = float(metrics.recall[cls_id]) if metrics.recall is not None else float("nan")
                        f.write(f"{cls_name:<15s} {prec:8.3f} {rec:8.3f} {map50:10.3f} {map50_95:10.3f}\n")
                    f.write("\n")
                except Exception:
                    f.write("Could not read per-class metrics.\n\n")

        f.write("=" * 80 + "\n")
        f.write("END OF TEST SUMMARY\n")
        f.write("=" * 80 + "\n")

    print(f"✓ Test summary saved to: {summary_file}")
    return summary_file
