#!/usr/bin/env python3
"""
Compare merged tiled inference metrics across all trained models.

Reads:
  - runs/detect/training_results/{run}/merged_test_inference/merged_metrics.json
  - training_results/args/{run}_args.txt

Produces one bar chart per metric saved to:
  runs/detect/training_results/model_comparison/
"""

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

RUNS_DIR = REPO_ROOT / "runs" / "detect" / "training_results"
ARGS_DIR = REPO_ROOT / "training_results" / "args"
OUT_DIR = RUNS_DIR / "model_comparison"

METRICS = [
    ("total_tp", "Total TP", False),
    ("total_fp", "Total FP", False),
    ("total_fn", "Total FN", False),
    ("precision", "Precision", True),
    ("recall", "Recall", True),
    ("f1", "F1 Score", True),
    ("ap50", "AP@50", True),
]


def parse_args_file(path: Path) -> dict:
    """Parse a key = value args file into a dict."""
    result = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            result[key.strip()] = val.strip()
    return result


def build_label(args: dict) -> str:
    """Build a short human-readable model label from parsed args."""
    model_raw = args.get("model", "unknown")
    model_name = Path(model_raw).stem if model_raw.endswith(".pt") else model_raw

    p2_val = args.get("p2", "False").strip().lower()
    arch = "p2" if p2_val == "true" else "p3"

    min_px = int(args.get("min_pixel_size", "0"))
    if min_px == 0:
        ann = "all"
    else:
        ann = f"sub{min_px}px"

    return f"{model_name}-{arch}-{ann}"


def build_label_from_folder(folder_name: str) -> str:
    """Parse a run folder name into a model label when no args file exists.

    Handles patterns like:
      yolov8n_4px_p2_beach_detection_...  -> yolov8n-p2-sub4px
      yolo26l_4x_p2_beach_detection_...   -> yolo26l-p2-sub4px  (4x treated as 4px)
      yolov8m_8px_beach_detection_...     -> yolov8m-p3-sub8px
      yolov8n_beach_detection_...         -> yolov8n-p3-all
    """
    prefix = re.split(r"_beach_detection_", folder_name, maxsplit=1)[0]
    tokens = prefix.split("_")

    model_name = tokens[0] if tokens else folder_name

    arch = "p2" if "p2" in tokens[1:] else "p3"

    ann = "all"
    for tok in tokens[1:]:
        m = re.fullmatch(r"(\d+)(?:px?|x)", tok, re.IGNORECASE)
        if m:
            ann = f"sub{m.group(1)}px"
            break

    return f"{model_name}-{arch}-{ann}"


def collect_results() -> list[dict]:
    """Walk RUNS_DIR and collect metrics + label for each model."""
    if not RUNS_DIR.exists():
        print(f"ERROR: Runs directory not found: {RUNS_DIR}")
        sys.exit(1)

    results = []
    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "model_comparison":
            continue

        metrics_file = run_dir / "merged_test_inference" / "merged_metrics.json"
        if not metrics_file.exists():
            print(f"  [skip] no merged_metrics.json in {run_dir.name}")
            continue

        with open(metrics_file) as f:
            data = json.load(f)
        summary = data.get("summary", {})

        args_file = ARGS_DIR / f"{run_dir.name}_args.txt"
        if args_file.exists():
            args = parse_args_file(args_file)
            label = build_label(args)
        else:
            label = build_label_from_folder(run_dir.name)

        results.append({"label": label, "run": run_dir.name, **summary})

    metric_keys = [k for k, _, _ in METRICS]
    results = [r for r in results if all(r.get(k, 0) != 0 for k in metric_keys)]
    results.sort(key=lambda r: r["label"])
    return results


def plot_metrics(results: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    labels = [r["label"] for r in results]
    n = len(labels)
    x = np.arange(n)
    bar_width = 0.5

    colors_count = "#4c72b0"
    colors_rate = "#55a868"

    for key, title, is_rate in METRICS:
        values = [r.get(key, 0) for r in results]

        fig, ax = plt.subplots(figsize=(max(6, n * 1.4), 5))
        color = colors_rate if is_rate else colors_count
        bars = ax.bar(x, values, width=bar_width, color=color, edgecolor="white", linewidth=0.8)

        for bar, val in zip(bars, values):
            fmt = f"{val:.4f}" if is_rate else str(int(val))
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.002 if is_rate else max(values) * 0.01),
                fmt,
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f"Model Comparison – {title}", fontsize=13, fontweight="bold")

        if is_rate:
            ax.set_ylim(0, 1.05)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        else:
            ax.set_ylim(0, max(values) * 1.15 if values else 1)

        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

        fig.tight_layout()
        out_path = OUT_DIR / f"{key}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_path.relative_to(REPO_ROOT)}")


def print_summary(results: list[dict]) -> None:
    col_w = max(len(r["label"]) for r in results) + 2
    header = f"{'Model':<{col_w}}  {'TP':>6}  {'FP':>6}  {'FN':>6}  {'P':>7}  {'R':>7}  {'F1':>7}  {'AP50':>7}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['label']:<{col_w}}"
            f"  {r.get('total_tp', 0):>6}"
            f"  {r.get('total_fp', 0):>6}"
            f"  {r.get('total_fn', 0):>6}"
            f"  {r.get('precision', 0):>7.4f}"
            f"  {r.get('recall', 0):>7.4f}"
            f"  {r.get('f1', 0):>7.4f}"
            f"  {r.get('ap50', 0):>7.4f}"
        )
    print("=" * len(header) + "\n")


def main() -> None:
    print(f"Scanning: {RUNS_DIR}")
    results = collect_results()

    if not results:
        print("No models with merged metrics found.")
        sys.exit(1)

    print(f"\nFound {len(results)} model(s).")
    print_summary(results)

    print("Generating plots...")
    plot_metrics(results)
    print(f"\nDone. Charts saved to: {OUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
