#!/usr/bin/env python3
"""
Interactive dashboard for comparing YOLO beach detection models.

Usage:
    cd /Users/jlee/Desktop/github/beach-obj-detection-training
    python dashboard/app.py

Then open http://localhost:5000 in your browser.
"""

import json
import re
import sys
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs" / "detect" / "training_results"
ARGS_DIR = REPO_ROOT / "training_results" / "args"

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
    if not path.exists():
        return result
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            result[key.strip()] = val.strip()
    return result


def parse_training_summary(path: Path) -> dict:
    """Parse training_summary.txt into a dict."""
    result = {}
    if not path.exists():
        return result
    
    content = path.read_text()
    
    # Extract key metrics using regex
    patterns = {
        "best_epoch": r"Best Epoch:\s*(\d+)",
        "best_map50_95": r"Best mAP50-95:\s*(\d+\.\d+)",
        "best_map50": r"Best mAP50:\s*(\d+\.\d+)",
        "best_precision": r"Best Precision:\s*(\d+\.\d+)",
        "best_recall": r"Best Recall:\s*(\d+\.\d+)",
        "final_epoch": r"Final Epoch:\s*(\d+)",
        "final_map50_95": r"Final mAP50-95:\s*(\d+\.\d+)",
        "final_map50": r"Final mAP50:\s*(\d+\.\d+)",
        "final_precision": r"Final Precision:\s*(\d+\.\d+)",
        "final_recall": r"Final Recall:\s*(\d+\.\d+)",
        "train_box_loss": r"Train Box Loss\s*:\s*(\d+\.\d+)",
        "train_class_loss": r"Train Class Loss\s*:\s*(\d+\.\d+)",
        "train_dfl_loss": r"Train DFL Loss\s*:\s*(\d+\.\d+)",
        "val_box_loss": r"Val Box Loss\s*:\s*(\d+\.\d+)",
        "val_class_loss": r"Val Class Loss\s*:\s*(\d+\.\d+)",
        "val_dfl_loss": r"Val DFL Loss\s*:\s*(\d+\.\d+)",
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            result[key] = float(match.group(1)) if "." in match.group(1) else int(match.group(1))
    
    return result


def build_label_from_args(args: dict) -> str:
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
    """Parse a run folder name into a model label when no args file exists."""
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


def parse_model_info(folder_name: str) -> dict:
    """Extract model type, size, yolo version from model name."""
    # Examples: yolov8n, yolo26m, yolov8s -> yolo_version: 8, size: n, yolo_type: v8
    match = re.match(r"(yolo)(v?(\d+))?([nsmlx])", folder_name, re.IGNORECASE)
    if match:
        yolo_type = match.group(2) or "v8"  # v8, v11, 26, etc
        yolo_version = match.group(3) or "8"  # 8, 11, 26
        size = match.group(4).lower()  # n, s, m, l, x
        return {
            "yolo_type": yolo_type.lower().replace("v", ""),
            "yolo_version": yolo_version,
            "size": size,
        }
    return {"yolo_type": "unknown", "yolo_version": "unknown", "size": "unknown"}


def collect_model_data() -> list[dict]:
    """Collect all model data from the runs directory."""
    if not RUNS_DIR.exists():
        return []

    results = []
    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "model_comparison":
            continue

        metrics_file = run_dir / "merged_test_inference" / "merged_metrics.json"
        if not metrics_file.exists():
            continue

        with open(metrics_file) as f:
            data = json.load(f)
        summary = data.get("summary", {})

        # Check for zero metrics
        metric_keys = [k for k, _, _ in METRICS]
        if any(summary.get(k, 0) == 0 for k in metric_keys):
            continue

        # Parse args
        args_file = ARGS_DIR / f"{run_dir.name}_args.txt"
        if args_file.exists():
            args = parse_args_file(args_file)
            label = build_label_from_args(args)
        else:
            args = {}
            label = build_label_from_folder(run_dir.name)

        # Parse training summary
        summary_file = REPO_ROOT / run_dir.name / "training_summary.txt"
        training_metrics = parse_training_summary(summary_file)

        # Parse model info from label
        model_info = parse_model_info(label)

        # Extract architecture and pixel filter from label
        arch_match = re.search(r"-(p[23])-", label)
        arch = arch_match.group(1) if arch_match else "p3"

        pixel_match = re.search(r"-sub(\d+)px$", label)
        pixel_filter = int(pixel_match.group(1)) if pixel_match else 0

        results.append({
            "id": run_dir.name,
            "label": label,
            "yolo_version": model_info["yolo_version"],
            "yolo_type": model_info["yolo_type"],
            "size": model_info["size"],
            "architecture": arch,
            "pixel_filter": pixel_filter,
            "args": args,
            "training_metrics": training_metrics,
            "inference_metrics": summary,
        })

    return results


# Global cache for model data
_model_data_cache = None
_last_scan_time = None


def get_model_data(refresh: bool = False) -> list[dict]:
    """Get model data, using cache unless refresh is requested."""
    global _model_data_cache, _last_scan_time
    
    if refresh or _model_data_cache is None:
        _model_data_cache = collect_model_data()
        _last_scan_time = datetime.now().isoformat()
    
    return _model_data_cache


@app.route("/")
def index():
    """Serve the main dashboard page."""
    return render_template("index.html")


@app.route("/api/models")
def get_models():
    """Get all model data."""
    refresh = request.args.get("refresh", "false").lower() == "true"
    data = get_model_data(refresh=refresh)
    return jsonify({
        "models": data,
        "last_scan": _last_scan_time,
        "count": len(data),
    })


@app.route("/api/filters")
def get_filters():
    """Get available filter options."""
    data = get_model_data()
    
    yolo_versions = sorted(set(m["yolo_version"] for m in data if m["yolo_version"] != "unknown"))
    sizes = sorted(set(m["size"] for m in data if m["size"] != "unknown"))
    architectures = sorted(set(m["architecture"] for m in data))
    pixel_filters = sorted(set(m["pixel_filter"] for m in data))
    
    return jsonify({
        "yolo_versions": yolo_versions,
        "sizes": sizes,
        "architectures": architectures,
        "pixel_filters": pixel_filters,
        "metrics": [{"key": k, "label": l, "is_rate": r} for k, l, r in METRICS],
    })


@app.route("/api/compare")
def compare_models():
    """Compare selected models on a specific metric."""
    model_ids = request.args.getlist("model")
    metric = request.args.get("metric", "f1")
    
    data = get_model_data()
    selected = [m for m in data if m["id"] in model_ids]
    
    return jsonify({
        "models": selected,
        "metric": metric,
        "metric_label": next((l for k, l, _ in METRICS if k == metric), metric),
    })


if __name__ == "__main__":
    print(f"Dashboard starting...")
    print(f"Scanning: {RUNS_DIR}")
    print(f"Open http://localhost:5000 in your browser")
    print(f"Press Ctrl+C to stop")
    
    # Pre-load data
    get_model_data(refresh=True)
    
    app.run(debug=True, host="0.0.0.0", port=5000)
