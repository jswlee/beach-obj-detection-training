#!/usr/bin/env python3
"""
Unified preprocessing + training pipeline for the YOLO beach detection model.

What this script does:
  1. Preprocess raw Roboflow-exported images (ROI masking, optional tiling,
     class filtering, small-annotation removal).
  2. Train a YOLO model on the processed dataset.
  3. Run test-set evaluation (standard or merged-tiled) and save metrics.

Usage examples:
  # Basic training (no tiling, all classes)
  python scripts/preprocess_and_train.py

  # Tiled training, person-only, filter tiny boxes
  python scripts/preprocess_and_train.py --enable-slicing --person-only --min-pixel-size 6

  # Resume from a checkpoint
  python scripts/preprocess_and_train.py --resume-from runs/.../weights/last.pt --epochs 350

All default hyperparameters are defined in config.py and can be overridden
via CLI flags.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

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
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TRAINING_RESULTS_DIR,
    TRAIN_DEFAULTS,
    INFERENCE_DEFAULTS,
    PERSON_CLASS_ID,
)
from lib.preprocessing import (
    process_yolo_dataset,
    is_youtube_snapshot,
    should_apply_mask,
    get_bottom_three_tiles,
    get_tiles,
    crop_and_mask_tiles,
    EXCLUSION_POLYGON,
    OVERLAP_FULL,
)
from lib.metrics import nms_merge, compute_ap
from lib.utils import (
    patch_torch_load,
    select_device,
    save_training_summary,
)

# Apply PyTorch compatibility patch (must happen before any YOLO loads)
patch_torch_load()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_yolo_model(
    dataset_yaml: str,
    model_size: str,
    epochs: int,
    batch_size: int,
    img_size: int,
    run_name: str,
    freeze: int,
    lr0: float,
    patience: int,
    warmup_epochs: int,
    optimizer: str,
    resume_from: str | None = None,
    p2: bool = False,
) -> str:
    """Train a YOLO model and return the path to the best weights.

    All augmentation and loss-weight hyperparameters come from
    ``config.TRAIN_DEFAULTS`` so they stay in one place.

    Args:
        dataset_yaml:  Path to the data.yaml produced by preprocessing.
        model_size:    Base model name or .pt checkpoint path.
        epochs:        Total epoch count (when resuming, training continues
                       until this total is reached).
        batch_size:    Images per training batch.
        img_size:      Image resolution for training.
        run_name:      Name for this training run (used as subfolder name).
        freeze:        Number of backbone layers to freeze.
        lr0:           Initial learning rate.
        patience:      Early-stopping patience.
        warmup_epochs: Number of warmup epochs.
        optimizer:     Optimizer name ("auto", "AdamW", "SGD", etc.).
        resume_from:   Optional .pt checkpoint to resume training from.
        p2:            If True, use the P2 (stride-4) architecture for
                       small-object detection.

    Returns:
        Absolute path to the saved best.pt model weights.
    """
    start_time = datetime.now()

    print("Beach Detection – YOLO Training")
    print("=" * 70)

    # Prepare output directories
    output_dir = str(TRAINING_RESULTS_DIR)
    (TRAINING_RESULTS_DIR / "runs").mkdir(parents=True, exist_ok=True)
    (TRAINING_RESULTS_DIR / "models").mkdir(parents=True, exist_ok=True)

    if not Path(dataset_yaml).exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")

    device = select_device()
    print(f"Using device: {device}\n")

    # Build or resume model
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
        model = YOLO(resume_from)
    elif p2:
        base_name = Path(model_size).stem  # e.g. "yolov8n"
        p2_yaml = f"{base_name}-p2.yaml"
        print(f"Loading P2 architecture: {p2_yaml} with weights from {model_size}")
        model = YOLO(p2_yaml).load(model_size)
    else:
        print(f"Loading model: {model_size}")
        model = YOLO(model_size)

    # Pull fixed hyper-params from config (augmentation, loss weights, etc.)
    hp = TRAIN_DEFAULTS

    print("\nStarting training...")
    print("-" * 70)
    model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=output_dir,
        name=run_name,
        save=True,
        save_period=-1,
        val=True,
        plots=True,
        verbose=True,
        resume=bool(resume_from),
        # Optimiser
        optimizer=optimizer,
        lr0=lr0,
        lrf=hp["lrf"],
        momentum=hp["momentum"],
        weight_decay=hp["weight_decay"],
        warmup_epochs=warmup_epochs,
        warmup_momentum=hp["warmup_momentum"],
        warmup_bias_lr=hp["warmup_bias_lr"],
        # Loss weights
        box=hp["box"],
        cls=hp["cls"],
        dfl=hp["dfl"],
        freeze=freeze,
        # Augmentation
        hsv_h=hp["hsv_h"],
        hsv_s=hp["hsv_s"],
        hsv_v=hp["hsv_v"],
        degrees=hp["degrees"],
        translate=hp["translate"],
        scale=hp["scale"],
        shear=hp["shear"],
        perspective=hp["perspective"],
        flipud=hp["flipud"],
        fliplr=hp["fliplr"],
        mosaic=hp["mosaic"],
        mixup=hp["mixup"],
        copy_paste=hp["copy_paste"],
        patience=patience,
        close_mosaic=hp["close_mosaic"],
    )

    end_time = datetime.now()

    # Locate best weights written by Ultralytics
    run_dir = Path(output_dir) / "detect" / run_name
    best_auto = run_dir / "weights" / "best.pt"
    best_model_path = TRAINING_RESULTS_DIR / "models" / f"{run_name}.pt"
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    if best_auto.exists():
        best_model_path.write_bytes(best_auto.read_bytes())
    else:
        model.save(str(best_model_path))

    print(f"\n✓ Training complete!")
    print(f"  Best model: {best_model_path}")
    print(f"  Run dir:    {run_dir}\n")

    save_training_summary(run_dir, best_model_path, start_time, end_time)
    return str(best_model_path)


# ---------------------------------------------------------------------------
# Post-training test evaluation (standard, non-tiled)
# ---------------------------------------------------------------------------

def run_test_inference(
    model_path: str,
    dataset_yaml: str,
    img_size: int,
    run_dir: Path,
    conf: float = 0.15,
) -> None:
    """Run Ultralytics val on the test split and save results under *run_dir*.

    Produces annotated images, per-class mAP, PR / F1 / confusion-matrix
    plots, and YOLO-format prediction labels.
    """
    print("Running test-set inference...")
    print("=" * 70)

    device = select_device()
    print(f"Device: {device}\n")

    model = YOLO(model_path)
    results = model.val(
        data=dataset_yaml,
        split="test",
        imgsz=img_size,
        device=device,
        batch=1,
        project=str(run_dir),
        name="test_inference",
        save=True,
        save_txt=True,
        save_hybrid=False,
        plots=True,
        verbose=True,
        conf=conf,
    )

    print(f"\nTest inference completed → {results.save_dir}\n")


# ---------------------------------------------------------------------------
# Post-training test evaluation (merged tiled inference)
# ---------------------------------------------------------------------------

def run_merged_test_inference(
    model_path: str,
    raw_data_dir: str,
    img_size: int,
    run_dir: Path,
    conf: float = 0.20,
    iou_merge: float = 0.50,
    iou_match: float = 0.50,
    person_only: bool = False,
    person_class_id: int = PERSON_CLASS_ID,
) -> None:
    """Run tiled inference on raw test images, merge with NMS, and compute metrics.

    For youtube_snapshot images: 3 bottom tiles + ROI masking.
    For other images: full-resolution overlapping grid, no masking.
    Detections from overlapping tiles are merged via NMS.
    """
    print("Running merged tiled test inference...")
    print("=" * 70)

    device = select_device()
    model = YOLO(model_path)

    test_img_dir = Path(raw_data_dir) / "test" / "images"
    test_lbl_dir = Path(raw_data_dir) / "test" / "labels"
    if not test_img_dir.exists():
        print(f"No test images directory: {test_img_dir}")
        return

    image_files = sorted(
        list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png"))
    )
    if not image_files:
        print("No test images found.")
        return

    print(f"Device: {device}")
    print(f"Test images: {len(image_files)}")
    print(f"Conf: {conf} | NMS IoU: {iou_merge} | Match IoU: {iou_match}\n")

    out_dir = run_dir / "merged_test_inference"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)

    # Accumulators
    total_tp = total_fp = total_fn = 0
    per_image_results: list[dict] = []
    all_detections: list[tuple[float, bool, int]] = []
    total_gt_count = 0

    for img_file in image_files:
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Tiling strategy
        yt_snap = is_youtube_snapshot(img_file.name)
        tiles = get_bottom_three_tiles(w, h, img_size) if yt_snap else get_tiles(w, h, img_size, OVERLAP_FULL)

        # Masking
        apply_mask = should_apply_mask(img_file.name) if yt_snap else False
        mask = None
        if apply_mask:
            mask = np.ones((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [EXCLUSION_POLYGON], 0)
            img_inf = img.copy()
            img_inf[mask == 0] = 0
        else:
            img_inf = img

        # Extract tiles
        tile_imgs, tile_offsets = [], []
        for tx, ty in tiles:
            tile = img_inf[ty : ty + img_size, tx : tx + img_size]
            if tile.shape[0] == img_size and tile.shape[1] == img_size:
                tile_imgs.append(tile)
                tile_offsets.append((tx, ty))
        if not tile_imgs:
            continue

        # Batch inference
        results = model.predict(
            source=tile_imgs, imgsz=img_size, conf=conf,
            device=device, verbose=False, save=False,
        )

        # Map detections back to full-image coords
        all_boxes_l, all_scores_l, all_cls_l = [], [], []
        for res, (tx, ty) in zip(results, tile_offsets):
            if res.boxes is None or len(res.boxes) == 0:
                continue
            xyxy = res.boxes.xyxy.cpu().numpy().copy()
            xyxy[:, [0, 2]] += tx
            xyxy[:, [1, 3]] += ty
            all_boxes_l.append(xyxy)
            all_scores_l.append(res.boxes.conf.cpu().numpy())
            all_cls_l.append(res.boxes.cls.cpu().numpy().astype(np.int32))

        # NMS merge
        if all_boxes_l:
            boxes_np = np.concatenate(all_boxes_l).astype(np.float32)
            scores_np = np.concatenate(all_scores_l).astype(np.float32)
            cls_np = np.concatenate(all_cls_l)
            keep = nms_merge(boxes_np, scores_np, cls_np, iou_merge)
            pred_boxes, pred_scores, pred_cls = boxes_np[keep], scores_np[keep], cls_np[keep]
        else:
            pred_boxes = np.empty((0, 4), dtype=np.float32)
            pred_scores = np.empty(0, dtype=np.float32)
            pred_cls = np.empty(0, dtype=np.int32)

        # Save predictions
        pred_lines = []
        for i in range(len(pred_boxes)):
            x1, y1, x2, y2 = pred_boxes[i]
            cx = ((x1 + x2) / 2.0) / w
            cy = ((y1 + y2) / 2.0) / h
            bw_n = (x2 - x1) / w
            bh_n = (y2 - y1) / h
            pred_lines.append(
                f"{int(pred_cls[i])} {cx:.6f} {cy:.6f} {bw_n:.6f} {bh_n:.6f} {pred_scores[i]:.4f}"
            )
        (pred_dir / f"{img_file.stem}.txt").write_text("\n".join(pred_lines))

        # --- Load and filter ground truth ---
        y_band_top = max(0, h - img_size) if yt_snap else 0
        gt_boxes_xyxy, gt_cls_list = [], []
        lbl_file = test_lbl_dir / f"{img_file.stem}.txt"
        if lbl_file.exists():
            for line in lbl_file.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                except (ValueError, IndexError):
                    continue
                if person_only and cls_id != person_class_id:
                    continue
                if apply_mask and mask is not None:
                    px = max(0, min(int(cx * w), w - 1))
                    py = max(0, min(int(cy * h), h - 1))
                    if mask[py, px] == 0:
                        continue
                if cy * h < y_band_top:
                    continue
                x1 = (cx - bw / 2.0) * w
                y1 = (cy - bh / 2.0) * h
                x2 = (cx + bw / 2.0) * w
                y2 = (cy + bh / 2.0) * h
                gt_boxes_xyxy.append([x1, y1, x2, y2])
                gt_cls_list.append(0 if person_only else cls_id)

        gt_boxes_np = np.array(gt_boxes_xyxy, dtype=np.float32) if gt_boxes_xyxy else np.empty((0, 4), dtype=np.float32)
        gt_cls_np = np.array(gt_cls_list, dtype=np.int32) if gt_cls_list else np.empty(0, dtype=np.int32)

        # --- Greedy IoU matching ---
        n_pred, n_gt = len(pred_boxes), len(gt_boxes_np)
        total_gt_count += n_gt
        gt_matched = np.zeros(n_gt, dtype=bool)
        img_tp = img_fp = 0

        if n_pred > 0:
            for idx in np.argsort(-pred_scores):
                p_box, p_cls_id, p_conf = pred_boxes[idx], pred_cls[idx], float(pred_scores[idx])
                best_iou, best_gt = 0.0, -1
                for g in range(n_gt):
                    if gt_matched[g] or gt_cls_np[g] != p_cls_id:
                        continue
                    ix1 = max(p_box[0], gt_boxes_np[g, 0])
                    iy1 = max(p_box[1], gt_boxes_np[g, 1])
                    ix2 = min(p_box[2], gt_boxes_np[g, 2])
                    iy2 = min(p_box[3], gt_boxes_np[g, 3])
                    if ix2 <= ix1 or iy2 <= iy1:
                        continue
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    area_p = (p_box[2] - p_box[0]) * (p_box[3] - p_box[1])
                    area_g = (gt_boxes_np[g, 2] - gt_boxes_np[g, 0]) * (gt_boxes_np[g, 3] - gt_boxes_np[g, 1])
                    iou = inter / (area_p + area_g - inter)
                    if iou > best_iou:
                        best_iou, best_gt = iou, g
                if best_iou >= iou_match and best_gt >= 0:
                    gt_matched[best_gt] = True
                    img_tp += 1
                    all_detections.append((p_conf, True, int(p_cls_id)))
                else:
                    img_fp += 1
                    all_detections.append((p_conf, False, int(p_cls_id)))

        img_fn = n_gt - img_tp
        total_tp += img_tp
        total_fp += img_fp
        total_fn += img_fn

        per_image_results.append({
            "image": img_file.name,
            "tiles": len(tile_offsets),
            "tiling": "3-bottom" if yt_snap else "full-grid",
            "masked": apply_mask,
            "gt_count": n_gt,
            "pred_count": len(pred_boxes),
            "tp": img_tp, "fp": img_fp, "fn": img_fn,
        })

        img_prec = img_tp / (img_tp + img_fp) if (img_tp + img_fp) else 0.0
        img_rec = img_tp / (img_tp + img_fn) if (img_tp + img_fn) else 0.0
        print(
            f"  {img_file.name}: tiles={len(tile_offsets)} gt={n_gt} "
            f"pred={len(pred_boxes)} TP={img_tp} FP={img_fp} FN={img_fn} "
            f"P={img_prec:.2f} R={img_rec:.2f}"
        )

    # --- Aggregate ---
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    ap50 = compute_ap(all_detections, total_gt_count)

    summary = {
        "model": str(model_path), "raw_data_dir": str(raw_data_dir),
        "conf": conf, "iou_merge": iou_merge, "iou_match": iou_match,
        "person_only": person_only, "images": len(per_image_results),
        "total_gt": total_gt_count, "total_tp": total_tp,
        "total_fp": total_fp, "total_fn": total_fn,
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4), "ap50": round(ap50, 4),
    }

    print(f"\n{'=' * 70}")
    print(f"Merged Test Results: images={len(per_image_results)} GT={total_gt_count}")
    print(f"  TP={total_tp} FP={total_fp} FN={total_fn}")
    print(f"  P={precision:.4f} R={recall:.4f} F1={f1:.4f} AP@50={ap50:.4f}")
    print("=" * 70)

    results_path = out_dir / "merged_metrics.json"
    with open(results_path, "w") as f:
        json.dump({"summary": summary, "per_image": per_image_results}, f, indent=2)
    print(f"Results saved to: {results_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    td = TRAIN_DEFAULTS
    inf = INFERENCE_DEFAULTS

    parser = argparse.ArgumentParser(
        description="Preprocess and train YOLO model for beach detection"
    )

    # --- Preprocessing args ---
    g_pre = parser.add_argument_group("Preprocessing")
    g_pre.add_argument("--raw-data-dir", type=str, default=str(RAW_DATA_DIR),
                       help="Directory containing raw annotated data from Roboflow")
    g_pre.add_argument("--processed-data-dir", type=str, default=str(PROCESSED_DATA_DIR),
                       help="Directory for processed (ROI-masked / tiled) images")
    g_pre.add_argument("--skip-preprocessing", action="store_true",
                       help="Skip preprocessing if processed data already exists")
    g_pre.add_argument("--enable-slicing", action="store_true",
                       help="Enable image slicing into 640×640 tiles with overlap")
    g_pre.add_argument("--person-only", action="store_true",
                       help="Keep only person-class annotations")
    g_pre.add_argument("--person-class-id", type=int, default=PERSON_CLASS_ID,
                       help="Class ID for person in the raw dataset")
    g_pre.add_argument("--min-pixel-size", type=int, default=0,
                       help="Filter annotations where sqrt(w*h) < N pixels")

    # --- Training args ---
    g_train = parser.add_argument_group("Training")
    g_train.add_argument("--epochs", type=int, default=td["epochs"])
    g_train.add_argument("--batch-size", type=int, default=td["batch_size"])
    g_train.add_argument("--img-size", type=int, default=td["img_size"])
    g_train.add_argument("--model", type=str, default=td["model"],
                         help="Base model weights or .pt checkpoint path")
    g_train.add_argument("--run-name", type=str, default=None,
                         help="Name for this training run (auto-generated if omitted)")
    g_train.add_argument("--freeze", type=int, default=td["freeze"])
    g_train.add_argument("--lr0", type=float, default=td["lr0"])
    g_train.add_argument("--patience", type=int, default=td["patience"])
    g_train.add_argument("--warmup-epochs", type=int, default=td["warmup_epochs"])
    g_train.add_argument("--optimizer", type=str, default=td["optimizer"])
    g_train.add_argument("--conf", type=float, default=inf["conf"],
                         help="Confidence threshold for post-training inference")
    g_train.add_argument("--p2", action="store_true",
                         help="Use P2 (stride-4) architecture for small-object detection")
    g_train.add_argument("--resume-from", type=str, default=None,
                         help="Checkpoint .pt to resume training from")

    args = parser.parse_args()

    # Resolve paths
    raw_data_path = Path(args.raw_data_dir).resolve()
    processed_dir = args.processed_data_dir
    if args.min_pixel_size > 0:
        processed_dir = f"{processed_dir}_sub{args.min_pixel_size}px_removed"
    processed_data_path = Path(processed_dir).resolve()

    print("=" * 70)
    print("Beach Detection – Unified Preprocessing & Training")
    print("=" * 70 + "\n")

    # ------------------------------------------------------------------
    # Step 1: Preprocessing
    # ------------------------------------------------------------------
    dataset_yaml_path = processed_data_path / "data.yaml"

    if args.skip_preprocessing and dataset_yaml_path.exists():
        print(f"Skipping preprocessing – using: {processed_data_path}\n")
    else:
        print("Step 1: Preprocessing")
        print("-" * 70)
        print(f"  Input:  {raw_data_path}")
        print(f"  Output: {processed_data_path}\n")

        if not raw_data_path.exists():
            print(f"ERROR: Raw data directory not found: {raw_data_path}")
            return 1

        try:
            dataset_yaml_path = Path(process_yolo_dataset(
                str(raw_data_path),
                str(processed_data_path),
                enable_slicing=args.enable_slicing,
                person_only=args.person_only,
                person_class_id=args.person_class_id,
                img_size=args.img_size,
                min_pixel_size=args.min_pixel_size,
            ))
            print(f"  Dataset YAML: {dataset_yaml_path}\n")
        except Exception as e:
            print(f"Preprocessing failed: {e}")
            return 1

    # ------------------------------------------------------------------
    # Step 2: Training
    # ------------------------------------------------------------------
    print("Step 2: Training")
    print("-" * 70)

    if args.run_name is None:
        args.run_name = f"beach_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"  Run:       {args.run_name}")
    print(f"  Model:     {args.model}")
    print(f"  Epochs:    {args.epochs}")
    print(f"  Batch:     {args.batch_size}")
    print(f"  Img size:  {args.img_size}")
    print(f"  LR:        {args.lr0}")
    print(f"  Optimizer: {args.optimizer}\n")

    try:
        model_path = train_yolo_model(
            dataset_yaml=str(dataset_yaml_path),
            model_size=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            run_name=args.run_name,
            freeze=args.freeze,
            lr0=args.lr0,
            patience=args.patience,
            warmup_epochs=args.warmup_epochs,
            optimizer=args.optimizer,
            resume_from=args.resume_from,
            p2=args.p2,
        )

        # Step 3: Post-training evaluation
        run_dir = TRAINING_RESULTS_DIR / "detect" / args.run_name
        if args.enable_slicing:
            run_merged_test_inference(
                model_path=model_path,
                raw_data_dir=str(raw_data_path),
                img_size=args.img_size,
                run_dir=run_dir,
                conf=args.conf,
                person_only=args.person_only,
                person_class_id=args.person_class_id,
            )
        else:
            run_test_inference(
                model_path=model_path,
                dataset_yaml=str(dataset_yaml_path),
                img_size=args.img_size,
                run_dir=run_dir,
                conf=args.conf,
            )

        print(f"\n{'=' * 70}")
        print("✓ Workflow completed successfully!")
        print(f"  Model:   {model_path}")
        print(f"  Results: {run_dir}")
        print("=" * 70 + "\n")
        return 0

    except Exception as e:
        print(f"\n{'=' * 70}")
        print(f"❌ Training failed: {e}")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
