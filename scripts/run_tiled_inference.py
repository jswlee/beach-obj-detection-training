#!/usr/bin/env python3
"""
Standalone tiled YOLO inference on full-resolution images.

Pipeline:
  original image → determine tiling layout → optionally apply ROI mask →
  crop tiles → batched YOLO predict → map boxes to full-image coords →
  cross-tile NMS merge → save outputs (labels, visualisations, metrics).

This script is designed for apples-to-apples evaluation / visualisation
when the model was trained on 640×640 tiles.

Usage examples:
  # Basic run with visualisation
  python scripts/run_tiled_inference.py --model path/to/best.pt --save-img

  # Full metrics run with ground-truth comparison
  python scripts/run_tiled_inference.py --model path/to/best.pt --calc-metrics --save-txt

  # Live preview (press 'q' to quit)
  python scripts/run_tiled_inference.py --model path/to/best.pt --show
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Make sure the repo root is on sys.path so that `config` and `lib` resolve.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import (
    EXCLUSION_POLYGON,
    INFERENCE_DEFAULTS,
    OVERLAP_FULL,
    MIN_OVERLAP_RATIO,
    PERSON_CLASS_ID,
)
from lib.preprocessing import (
    crop_and_mask_tiles,
    get_tiles_for_image,
    is_youtube_snapshot,
    should_apply_mask,
)
from lib.metrics import (
    nms_merge,
    calculate_map_metrics,
)
from lib.utils import (
    patch_torch_load,
    select_device,
    collect_images,
)

# Apply PyTorch compatibility patch before any model loading
patch_torch_load()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Det:
    """A single detection in full-image pixel coordinates."""
    cls_id: int
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float


# ---------------------------------------------------------------------------
# Drawing / formatting helpers
# ---------------------------------------------------------------------------

def to_yolo_line(det: Det, w: int, h: int) -> str:
    """Convert a Det to a normalised YOLO label line (cls cx cy bw bh)."""
    cx = max(0.0, min(1.0, ((det.x1 + det.x2) / 2.0) / w))
    cy = max(0.0, min(1.0, ((det.y1 + det.y2) / 2.0) / h))
    bw = max(0.0, min(1.0, (det.x2 - det.x1) / w))
    bh = max(0.0, min(1.0, (det.y2 - det.y1) / h))
    return f"{det.cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def draw_dets(
    img_bgr: np.ndarray,
    dets: list[Det],
    names: dict[int, str] | None,
    gt_boxes: list[tuple[int, float, float, float, float]] | None = None,
) -> np.ndarray:
    """Draw predictions (green) and optional ground-truth (blue) on an image.

    Args:
        img_bgr:  BGR image to annotate.
        dets:     Model detections to draw.
        names:    Class-ID → name mapping (from the model).
        gt_boxes: Optional ground-truth in YOLO normalised format
                  (cls, cx, cy, bw, bh).

    Returns:
        Annotated BGR image (copy of input).
    """
    out = img_bgr.copy()
    h, w = out.shape[:2]

    # Ground truth in blue
    if gt_boxes:
        for cls_id, cx, cy, bw, bh in gt_boxes:
            x1 = max(0, min(w - 1, int((cx - bw / 2.0) * w)))
            y1 = max(0, min(h - 1, int((cy - bh / 2.0) * h)))
            x2 = max(0, min(w - 1, int((cx + bw / 2.0) * w)))
            y2 = max(0, min(h - 1, int((cy + bh / 2.0) * h)))
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Predictions in green
    for d in dets:
        x1 = max(0, min(w - 1, int(d.x1)))
        y1 = max(0, min(h - 1, int(d.y1)))
        x2 = max(0, min(w - 1, int(d.x2)))
        y2 = max(0, min(h - 1, int(d.y2)))
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = names.get(d.cls_id, str(d.cls_id)) if names else str(d.cls_id)
        cv2.putText(out, f"{label} {d.conf:.2f}",
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out


# ---------------------------------------------------------------------------
# Ground-truth loading
# ---------------------------------------------------------------------------

def _find_label_path(img_path: Path) -> Path | None:
    """Locate the YOLO .txt label file for a given image.

    Searches common directory layouts (co-located, sibling labels/ dir).
    Returns None if no label file exists.
    """
    candidates: list[Path] = [img_path.with_suffix(".txt")]
    parent = img_path.parent
    if parent.name.lower() == "images":
        candidates.append(parent.parent / "labels" / f"{img_path.stem}.txt")
        candidates.append(parent / "labels" / f"{img_path.stem}.txt")
    # Windows-style path check
    img_str = str(img_path)
    if "\\images\\" in img_str.lower():
        parts = img_str.split("\\")
        for i, p in enumerate(parts):
            if p.lower() == "images":
                parts2 = parts.copy()
                parts2[i] = "labels"
                candidates.append(Path("\\".join(parts2)).with_suffix(".txt"))
                break
    for c in candidates:
        if c.exists():
            return c
    return None


def load_gt_boxes(
    img_path: Path,
    keep_class_id: int | None = None,
) -> tuple[list[tuple[int, float, float, float, float]], bool]:
    """Load ground-truth boxes for an image from its YOLO label file.

    Args:
        img_path:       Path to the image.
        keep_class_id:  If set, only keep boxes with this class ID.

    Returns:
        (boxes, found) where boxes is a list of (cls, cx, cy, bw, bh) and
        found indicates whether a label file was located at all.
    """
    txt_path = _find_label_path(img_path)
    if txt_path is None:
        return ([], False)
    raw = txt_path.read_text().strip()
    if not raw:
        return ([], True)
    out: list[tuple[int, float, float, float, float]] = []
    for line in raw.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError:
            continue
        if keep_class_id is None or cls_id == keep_class_id:
            out.append((cls_id, cx, cy, bw, bh))
    return (out, True)


def filter_gt_to_tile_region(
    gt_boxes: list[tuple[int, float, float, float, float]],
    img_w: int,
    img_h: int,
    tile_size: int,
    img_filename: str,
) -> list[tuple[int, float, float, float, float]]:
    """Keep only GT boxes that the tiled model can actually see.

    Mirrors preprocessing logic:
      1. Spatial: box must overlap >= MIN_OVERLAP_RATIO with at least one tile.
      2. ROI mask: for masked images, drop boxes whose centre is in the
         exclusion zone.
    """
    tiles = get_tiles_for_image(img_filename, img_w, img_h, tile_size)
    apply_mask = should_apply_mask(img_filename)

    roi_mask = None
    if apply_mask:
        roi_mask = np.ones((img_h, img_w), dtype=np.uint8)
        cv2.fillPoly(roi_mask, [EXCLUSION_POLYGON], 0)

    filtered: list[tuple[int, float, float, float, float]] = []
    for cls, cx, cy, bw, bh in gt_boxes:
        # ROI check
        if roi_mask is not None:
            px = max(0, min(img_w - 1, int(cx * img_w)))
            py = max(0, min(img_h - 1, int(cy * img_h)))
            if roi_mask[py, px] == 0:
                continue

        # Spatial overlap check
        abs_cx, abs_cy = cx * img_w, cy * img_h
        abs_w, abs_h = bw * img_w, bh * img_h
        box_x1 = abs_cx - abs_w / 2.0
        box_y1 = abs_cy - abs_h / 2.0
        box_x2 = abs_cx + abs_w / 2.0
        box_y2 = abs_cy + abs_h / 2.0
        original_area = abs_w * abs_h
        if original_area <= 0:
            continue

        in_any = False
        for tx, ty in tiles:
            ix1 = max(box_x1, tx)
            iy1 = max(box_y1, ty)
            ix2 = min(box_x2, tx + tile_size)
            iy2 = min(box_y2, ty + tile_size)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            if (ix2 - ix1) * (iy2 - iy1) / original_area >= MIN_OVERLAP_RATIO:
                in_any = True
                break
        if in_any:
            filtered.append((cls, cx, cy, bw, bh))

    return filtered


# ---------------------------------------------------------------------------
# Core inference (one image)
# ---------------------------------------------------------------------------

def infer_one_image(
    model: YOLO,
    img_bgr: np.ndarray,
    conf: float,
    tile_size: int,
    iou_merge: float,
    device: str,
    img_filename: str = "",
    stage_times: dict | None = None,
) -> list[Det]:
    """Run tiled inference on a single image and return merged detections.

    Stages:
      1. Tile + mask the image.
      2. Batch-predict with the YOLO model.
      3. Map tile-local boxes to full-image coordinates.
      4. Cross-tile NMS merge.

    Args:
        model:        Loaded YOLO model.
        img_bgr:      Full-resolution BGR image.
        conf:         Confidence threshold.
        tile_size:    Tile width/height in pixels (must match training).
        iou_merge:    IoU threshold for cross-tile NMS.
        device:       Compute device string.
        img_filename: Original filename (used to pick tiling strategy).
        stage_times:  Optional dict to record per-stage latencies (ms).

    Returns:
        List of Det objects in full-image pixel coordinates.
    """
    h, w = img_bgr.shape[:2]

    # Stage 1: Tile + mask
    t0 = time.perf_counter()
    tile_imgs, offsets = crop_and_mask_tiles(img_bgr, tile_size, img_filename)
    if stage_times is not None:
        stage_times["tile_mask_ms"] = (time.perf_counter() - t0) * 1000.0
    if not tile_imgs:
        return []

    # Stage 2: Model predict (batched)
    t1 = time.perf_counter()
    results = model.predict(
        source=tile_imgs, imgsz=tile_size, conf=conf,
        device=device, verbose=False, save=False,
    )
    if stage_times is not None:
        stage_times["infer_ms"] = (time.perf_counter() - t1) * 1000.0

    # Stage 3: Map back + NMS merge
    t2 = time.perf_counter()
    all_boxes, all_scores, all_cls = [], [], []
    for res, (tx, ty) in zip(results, offsets):
        if res.boxes is None or len(res.boxes) == 0:
            continue
        xyxy = res.boxes.xyxy.cpu().numpy()
        xyxy[:, [0, 2]] += tx
        xyxy[:, [1, 3]] += ty
        all_boxes.append(xyxy)
        all_scores.append(res.boxes.conf.cpu().numpy())
        all_cls.append(res.boxes.cls.cpu().numpy().astype(np.int32))

    if not all_boxes:
        if stage_times is not None:
            stage_times["postprocess_ms"] = (time.perf_counter() - t2) * 1000.0
        return []

    boxes_np = np.concatenate(all_boxes).astype(np.float32)
    scores_np = np.concatenate(all_scores).astype(np.float32)
    cls_np = np.concatenate(all_cls)
    keep = nms_merge(boxes_np, scores_np, cls_np, iou_merge)

    dets = [
        Det(
            cls_id=int(cls_np[i]),
            conf=float(scores_np[i]),
            x1=float(max(0, boxes_np[i, 0])),
            y1=float(max(0, boxes_np[i, 1])),
            x2=float(min(w, boxes_np[i, 2])),
            y2=float(min(h, boxes_np[i, 3])),
        )
        for i in keep
    ]
    if stage_times is not None:
        stage_times["postprocess_ms"] = (time.perf_counter() - t2) * 1000.0
    return dets


# ---------------------------------------------------------------------------
# Metrics report writer
# ---------------------------------------------------------------------------

def _percentile(vals: list[float], p: float) -> float:
    """Simple percentile without scipy dependency."""
    if not vals:
        return float("nan")
    v = sorted(vals)
    k = (len(v) - 1) * (p / 100.0)
    f_idx = int(np.floor(k))
    c_idx = int(np.ceil(k))
    if f_idx == c_idx:
        return float(v[f_idx])
    return float(v[f_idx] + (k - f_idx) * (v[c_idx] - v[f_idx]))


def _write_metrics(
    out_dir: Path,
    frame_metrics: list[dict],
    args,
    model_path: Path,
    device: str,
    all_predictions: list | None = None,
    all_ground_truth: list | None = None,
    img_sizes: list | None = None,
) -> None:
    """Write JSON + human-readable metrics summary to *out_dir*."""
    if not frame_metrics:
        return

    lat_ms = [float(m["latency_ms"]) for m in frame_metrics]
    total_s = sum(lat_ms) / 1000.0
    frames = len(frame_metrics)
    avg_fps = frames / total_s if total_s > 0 else 0.0

    summary: dict = {
        "model": str(model_path),
        "device": device,
        "conf": float(args.conf),
        "metrics_conf": float(args.metrics_conf),
        "tile_size": int(args.tile_size),
        "iou_merge": float(args.iou_merge),
        "half": bool(getattr(args, "half", False)),
        "frames": frames,
        "total_time_s": total_s,
        "avg_fps": avg_fps,
        "latency_ms": {
            "min": float(min(lat_ms)),
            "mean": float(sum(lat_ms) / len(lat_ms)),
            "p50": _percentile(lat_ms, 50),
            "p90": _percentile(lat_ms, 90),
            "p95": _percentile(lat_ms, 95),
            "max": float(max(lat_ms)),
        },
    }

    # Detection metrics (mAP etc.) if requested
    if args.calc_metrics and all_predictions and all_ground_truth and img_sizes:
        print("Calculating detection metrics...")
        gt_count = sum(1 for gts in all_ground_truth if gts)
        print(f"  Images with predictions: {len(all_predictions)}")
        print(f"  Images with ground truth: {gt_count}")
        if gt_count == 0:
            summary["detection_metrics"] = {
                "map50": 0.0, "map50_95": 0.0,
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "error": "No ground truth labels found",
            }
        else:
            try:
                det_metrics = calculate_map_metrics(
                    all_predictions=all_predictions,
                    all_ground_truth=all_ground_truth,
                    img_sizes=img_sizes,
                    conf_threshold=args.metrics_conf,
                    iou_threshold=0.45,
                    num_classes=args.num_classes,
                )
                summary["detection_metrics"] = det_metrics
                for k, v in det_metrics.items():
                    print(f"  {k}: {v:.4f}")
            except Exception as e:
                print(f"  Error: {e}")
                summary["detection_metrics"] = {
                    "map50": 0.0, "map50_95": 0.0,
                    "precision": 0.0, "recall": 0.0, "f1": 0.0,
                    "error": str(e),
                }

    # Write JSON
    json_path = out_dir / "inference_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "frames": frame_metrics}, f, indent=2)

    # Write human-readable text
    txt_path = out_dir / "inference_metrics.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("INFERENCE METRICS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Model:         {summary['model']}\n")
        f.write(f"Device:        {summary['device']}\n")
        f.write(f"Half:          {summary['half']}\n")
        f.write(f"Conf:          {summary['conf']}\n")
        f.write(f"Metrics conf:  {summary['metrics_conf']}\n")
        f.write(f"Tile size:     {summary['tile_size']}\n")
        f.write(f"IoU merge:     {summary['iou_merge']}\n")
        f.write(f"Frames:        {summary['frames']}\n")
        f.write(f"Total time (s):{summary['total_time_s']:.3f}\n")
        f.write(f"Avg FPS:       {summary['avg_fps']:.3f}\n")
        f.write("\nLatency (ms):\n")
        for k in ("min", "mean", "p50", "p90", "p95", "max"):
            f.write(f"  {k}: {summary['latency_ms'][k]:.3f}\n")
        if "detection_metrics" in summary:
            f.write("\nDETECTION METRICS\n")
            f.write("-" * 80 + "\n")
            dm = summary["detection_metrics"]
            for k in ("map50", "map50_95", "precision", "recall", "f1"):
                f.write(f"  {k}: {dm[k]:.4f}\n")
            if "error" in dm:
                f.write(f"  error: {dm['error']}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    inf = INFERENCE_DEFAULTS

    parser = argparse.ArgumentParser(
        description="Tiled YOLO inference with merge-back to full image"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained YOLO .pt model")
    parser.add_argument("--input-dir", type=str,
                        default="roboflow_data/Beach Counter.v16i.yolov8/test/images",
                        help="Directory with input images")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (auto-generated if omitted)")
    parser.add_argument("--conf", type=float, default=inf["conf"],
                        help="Confidence threshold for display / saving")
    parser.add_argument("--metrics-conf", type=float, default=inf["metrics_conf"],
                        help="Confidence threshold for mAP calculation")
    parser.add_argument("--tile-size", type=int, default=640,
                        help="Tile size in pixels (must match training)")
    parser.add_argument("--iou-merge", type=float, default=inf["iou_merge"],
                        help="IoU threshold for cross-tile NMS merge")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'auto', 'cpu', or 'cuda'")
    parser.add_argument("--gt-person-class-id", type=int, default=PERSON_CLASS_ID,
                        help="Class ID for person in ground-truth labels")
    parser.add_argument("--save-img", action="store_true",
                        help="Save annotated full-size images")
    parser.add_argument("--save-txt", action="store_true",
                        help="Save YOLO-format label files in full-image coords")
    parser.add_argument("--half", action="store_true",
                        help="Use FP16 inference on CUDA")
    parser.add_argument("--show", action="store_true",
                        help="Show a live preview window (press 'q' to quit)")
    parser.add_argument("--loop", action="store_true",
                        help="Loop over input images continuously")
    parser.add_argument("--delay-ms", type=int, default=0,
                        help="Delay between frames in ms")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Stop after N frames (0 = no limit)")
    parser.add_argument("--calc-metrics", action="store_true",
                        help="Calculate mAP metrics (requires GT labels)")
    parser.add_argument("--num-classes", type=int, default=inf["num_classes"],
                        help="Number of classes in the dataset")

    args = parser.parse_args()

    # Resolve paths
    model_path = Path(args.model).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    if args.output_dir is None:
        args.output_dir = str(model_path.parent.parent / "inference_output")
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Output sub-directories
    labels_dir = out_dir / "labels"
    vis_dir = out_dir / "visualized"
    vis_gt_dir = out_dir / "visualized_gt"
    if args.save_txt:
        labels_dir.mkdir(parents=True, exist_ok=True)
    if args.save_img:
        vis_dir.mkdir(parents=True, exist_ok=True)
        vis_gt_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = args.device.lower()
    if device == "auto":
        device = select_device()

    print(f"Loading model: {model_path}")
    print(f"Device: {device}")
    model = YOLO(str(model_path))

    if args.half and device == "cuda":
        try:
            model.model.half()
            print("Using FP16 (half precision)")
        except Exception as e:
            print(f"Could not enable FP16: {e}")

    names = getattr(model, "names", None)

    # GT class → model class mapping (handles class-ID mismatch)
    gt_to_model_cls: Dict[int, int] = {}
    if names and args.calc_metrics:
        model_name_to_id = {v.lower(): k for k, v in names.items()}
        if "person" in model_name_to_id:
            gt_to_model_cls[args.gt_person_class_id] = model_name_to_id["person"]
        if gt_to_model_cls:
            print(f"GT class remap: {gt_to_model_cls}")

    images = collect_images(input_dir)
    if not images:
        print(f"No images found in: {input_dir}")
        return 0
    print(f"Found {len(images)} images")

    # Warmup (reduces first-frame overhead on GPU)
    try:
        warm_img = cv2.imread(str(images[0]))
        if warm_img is not None:
            for _ in range(3):
                infer_one_image(model, warm_img, args.conf, args.tile_size,
                                args.iou_merge, device, images[0].name)
    except Exception:
        pass

    # --- Main loop ---
    frames_total = 0
    time_total_s = 0.0
    frame_metrics: list[dict] = []
    all_predictions: List[List[Tuple[float, float, float, float, float, int]]] = []
    all_ground_truth: List[List[Tuple[int, float, float, float, float]]] = []
    img_sizes: List[Tuple[int, int]] = []

    while True:
        for img_path in images:
            t0 = time.perf_counter()
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            stage_times: dict = {}
            dets = infer_one_image(
                model, img, args.conf, args.tile_size,
                args.iou_merge, device, img_path.name, stage_times,
            )

            # Ground truth & visualisation
            vis = None
            gt_boxes: list[tuple[int, float, float, float, float]] = []
            gt_found = False
            if args.save_img or args.show or args.calc_metrics:
                gt_boxes, gt_found = load_gt_boxes(img_path, keep_class_id=args.gt_person_class_id)
                vis = draw_dets(img, dets, names=names, gt_boxes=gt_boxes)

            # Metrics collection
            if args.calc_metrics:
                h, w = img.shape[:2]
                img_sizes.append((w, h))

                # Low-conf pass for complete PR curve
                if args.metrics_conf < args.conf:
                    metrics_dets = infer_one_image(
                        model, img, args.metrics_conf, args.tile_size,
                        args.iou_merge, device, img_path.name,
                    )
                else:
                    metrics_dets = dets

                all_predictions.append([
                    (d.x1, d.y1, d.x2, d.y2, d.conf, d.cls_id) for d in metrics_dets
                ])

                filtered_gt = filter_gt_to_tile_region(gt_boxes, w, h, args.tile_size, img_path.name)
                remapped_gt = [
                    (gt_to_model_cls.get(cls, cls), cx, cy, bw, bh)
                    for cls, cx, cy, bw, bh in filtered_gt
                ]
                all_ground_truth.append(remapped_gt)

            # Save outputs
            if args.save_txt:
                h, w = img.shape[:2]
                lines = [to_yolo_line(d, w, h) for d in dets]
                (labels_dir / f"{img_path.stem}.txt").write_text("\n".join(lines))

            if args.save_img and vis is not None:
                cv2.imwrite(str(vis_dir / img_path.name), vis)
                if gt_found:
                    vis_gt = draw_dets(img, [], names=None, gt_boxes=gt_boxes)
                    cv2.imwrite(str(vis_gt_dir / img_path.name), vis_gt)

            if args.show and vis is not None:
                cv2.imshow("tiled_inference", vis)

            # Timing
            dt = time.perf_counter() - t0
            frames_total += 1
            time_total_s += dt
            fps_inst = 1.0 / dt if dt > 0 else 0.0
            fps_avg = frames_total / time_total_s if time_total_s > 0 else 0.0

            frame_metrics.append({
                "frame": frames_total,
                "image": img_path.name,
                "dets": len(dets),
                "latency_ms": dt * 1000.0,
                "fps_inst": fps_inst,
                "fps_avg": fps_avg,
                **{k: round(v, 2) for k, v in stage_times.items()},
            })

            stage_str = " | ".join(f"{k}={v:.1f}" for k, v in stage_times.items())
            print(
                f"Frame {frames_total} | {img_path.name} | dets={len(dets)} | "
                f"{dt*1000:.1f}ms | fps={fps_inst:.2f} | {stage_str}"
            )

            # Interactive controls
            if args.show:
                wait = args.delay_ms if args.delay_ms > 0 else 1
                key = cv2.waitKey(wait) & 0xFF
                if key in (ord("q"), 27):
                    cv2.destroyAllWindows()
                    _write_metrics(out_dir, frame_metrics, args, model_path, device,
                                   all_predictions, all_ground_truth, img_sizes)
                    print(f"\nDone. Outputs in: {out_dir}")
                    return 0
            elif args.delay_ms > 0:
                time.sleep(args.delay_ms / 1000.0)

            if args.max_frames and frames_total >= args.max_frames:
                break

        if args.max_frames and frames_total >= args.max_frames:
            break
        if not args.loop:
            break

    if args.show:
        cv2.destroyAllWindows()

    _write_metrics(out_dir, frame_metrics, args, model_path, device,
                   all_predictions, all_ground_truth, img_sizes)
    print(f"\nDone. Outputs in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
