#!/usr/bin/env python3
"""
Bootstrap YOLO annotations for beach images (multi-class).
- Runs full-res or tiled inference (with cross-tile NMS).
- Filters to a chosen set of classes (default: person, boat, surfboard).
- Writes YOLO .txt labels and visualization images.
- Optional class ID remap for training (compact IDs 0..N-1 in your dataset).

Usage:
  python3 preprocessing/bootstrap_yolo_annotations.py \
      --source CV_Data/Kaanapali_Beach_raw \
      --out_dir data/yolo_bootstrap \
      --model yolov8x.pt \
      --tile 640 --overlap 0.2 \
      --keep person,boat,surfboard \
      --conf 0.15 \
      --remap

Notes:
- For boats/surfboards, use a detection model (not -pose).
- Canoes/kayaks generally map to COCO "boat".
"""

import argparse
from pathlib import Path
import shutil
import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------- NMS ----------------------
def iou_xyxy(a, b):
    """IoU between two boxes [x1,y1,x2,y2]."""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    denom = area_a + area_b - inter + 1e-9
    return inter / denom if denom > 0 else 0.0

def nms_per_class(boxes, scores, classes, iou_thr=0.5):
    """
    Simple per-class NMS.
    boxes: (N,4) xyxy
    scores: (N,)
    classes: (N,)
    returns indices to keep
    """
    keep_indices = []
    boxes = np.asarray(boxes)
    scores = np.asarray(scores)
    classes = np.asarray(classes, dtype=int)

    for cls in np.unique(classes):
        cls_mask = classes == cls
        b = boxes[cls_mask]
        s = scores[cls_mask]
        idxs = np.argsort(-s)
        kept = []

        while idxs.size > 0:
            i = idxs[0]
            kept.append(i)
            if idxs.size == 1:
                break
            ious = np.array([iou_xyxy(b[i], b[j]) for j in idxs[1:]])
            idxs = idxs[1:][ious <= iou_thr]

        # map back to original indices
        original = np.where(cls_mask)[0]
        keep_indices.extend(list(original[kept]))

    return keep_indices

# ------------------ Inference helpers ------------------
def run_full_image(model, img_path, conf):
    """Predict on full image at native resolution."""
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    res = model.predict(img_path, imgsz=max(h, w), conf=conf, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return np.zeros((0,4), dtype=float), np.zeros((0,), dtype=int), np.zeros((0,), dtype=float), (h, w)
    return (res.boxes.xyxy.cpu().numpy(),
            res.boxes.cls.cpu().numpy().astype(int),
            res.boxes.conf.cpu().numpy().astype(float),
            (h, w))

def run_tiled(model, img_path, tile_size=640, overlap=0.2, conf=0.15):
    """Split image into overlapping tiles, run model, merge coords back."""
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    stride = int(tile_size * (1 - overlap))
    all_boxes, all_cls, all_conf = [], [], []

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            x2 = min(x + tile_size, w); y2 = min(y + tile_size, h)
            tile = img[y:y2, x:x2]
            if tile.shape[0] < 50 or tile.shape[1] < 50:
                continue
            res = model.predict(tile, imgsz=tile_size, conf=conf, verbose=False)[0]
            if res.boxes is None or len(res.boxes) == 0:
                continue
            b = res.boxes.xyxy.cpu().numpy()
            c = res.boxes.cls.cpu().numpy().astype(int)
            s = res.boxes.conf.cpu().numpy().astype(float)

            # Offset tile boxes back to full image coords
            b[:, [0, 2]] += x
            b[:, [1, 3]] += y

            all_boxes.append(b)
            all_cls.append(c)
            all_conf.append(s)

    if len(all_boxes) == 0:
        return (np.zeros((0,4), dtype=float),
                np.zeros((0,), dtype=int),
                np.zeros((0,), dtype=float),
                (h, w))

    boxes = np.concatenate(all_boxes, axis=0)
    clses = np.concatenate(all_cls, axis=0)
    confs = np.concatenate(all_conf, axis=0)
    return boxes, clses, confs, (h, w)

# ------------------ Utility ------------------
def yolo_line(cls_id, x1, y1, x2, y2, W, H):
    xc = ((x1 + x2) / 2) / W
    yc = ((y1 + y2) / 2) / H
    bw = (x2 - x1) / W
    bh = (y2 - y1) / H
    return f"{int(cls_id)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"

def parse_keep_ids(model_names, keep_names_csv):
    """
    model_names: dict {id: name}
    keep_names_csv: "person,boat,surfboard"
    returns: (keep_ids_set, ordered_keep_ids, id_to_name)
    """
    name_to_id = {v: k for k, v in model_names.items()}
    keep_names = [s.strip() for s in keep_names_csv.split(",") if s.strip()]
    keep_ids = []
    for n in keep_names:
        if n not in name_to_id:
            raise ValueError(f"Requested class '{n}' not in model names. Available: {sorted(model_names.values())}")
        keep_ids.append(name_to_id[n])
    return set(keep_ids), keep_ids, model_names

def build_remap(ordered_keep_ids):
    """
    Build compact ID remap: kept class IDs -> 0..N-1 in the given order.
    Example: keep_ids [0(person),9(boat),36(surfboard)] -> {0:0,9:2,36:3}
    """
    return {orig: i for i, orig in enumerate(ordered_keep_ids)}

def color_for_cls(c):
    # deterministic pseudo-color
    rng = (int((37*c) % 255), int((17*c) % 255), int((97*c) % 255))
    return (rng[0], rng[1], rng[2])

# ------------------ Main ------------------
def main(args):
    source = Path(args.source)
    out_dir = Path(args.out_dir)
    img_out = out_dir / "images"
    lbl_out = out_dir / "labels"
    viz_out = out_dir / "viz"

    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    viz_out.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    model_names = model.names  # id->name dict (COCO names for COCO models)

    keep_set, keep_ordered, _ = parse_keep_ids(model_names, args.keep)
    remap = build_remap(keep_ordered) if args.remap else None

    img_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        img_files.extend(source.glob(ext))
    img_files = sorted(img_files)
    print(f"Found {len(img_files)} images in {source}")

    for img_path in img_files:
        # Inference (tiled or full)
        if args.tile:
            boxes, clses, confs, (H, W) = run_tiled(
                model, img_path, tile_size=args.tile, overlap=args.overlap, conf=args.conf
            )
        else:
            boxes, clses, confs, (H, W) = run_full_image(
                model, img_path, conf=args.conf
            )

        # Filter to kept classes
        if boxes.shape[0] == 0:
            kept_idx = []
        else:
            mask = np.array([c in keep_set for c in clses], dtype=bool)
            boxes = boxes[mask]
            clses = clses[mask]
            confs = confs[mask]
            # Per-class NMS to merge overlapping tile detections
            kept_idx = nms_per_class(boxes, confs, clses, iou_thr=args.nms_iou)

        boxes = boxes[kept_idx] if len(kept_idx) else np.zeros((0,4))
        clses = clses[kept_idx] if len(kept_idx) else np.zeros((0,), dtype=int)
        confs = confs[kept_idx] if len(kept_idx) else np.zeros((0,), dtype=float)

        # Copy original
        new_img_path = img_out / img_path.name
        shutil.copy(img_path, new_img_path)

        # Write labels + viz
        viz = cv2.imread(str(img_path))
        label_lines = []
        for (x1, y1, x2, y2), c, s in zip(boxes, clses, confs):
            cls_out = remap[c] if remap is not None else int(c)
            label_lines.append(yolo_line(cls_out, x1, y1, x2, y2, W, H))

            name = model_names[int(c)]
            color = color_for_cls(int(c) if remap is None else cls_out)
            cv2.rectangle(viz, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(viz, f"{name} {s:.2f}", (int(x1), int(y1)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Save label file (even if empty)
        with open(lbl_out / f"{img_path.stem}.txt", "w") as f:
            f.write("\n".join(label_lines))

        # Save viz
        cv2.imwrite(str(viz_out / img_path.name), viz)

        # Log
        cls_counts = {}
        for c in clses:
            key = model_names[int(c)]
            cls_counts[key] = cls_counts.get(key, 0) + 1
        print(f"Processed {img_path.name}: {sum(cls_counts.values())} boxes {cls_counts}")

    print(f"\n✅ Bootstrapped dataset saved to {out_dir}")
    print("   images/, labels/, viz/ ready.")
    if args.remap:
        print("ℹ️  Class ID remap applied:", {model.names[k]: v for k, v in build_remap(keep_ordered).items()})
        print("    Make sure your dataset.yaml uses these compact class names in the same order you passed to --keep.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Path to raw images folder")
    ap.add_argument("--out_dir", required=True, help="Output YOLO dataset dir")
    ap.add_argument("--model", default="yolov8x.pt", help="YOLO model weights (e.g., yolov10x.pt, yolov9e.pt, yolov8x.pt)")
    ap.add_argument("--tile", type=int, default=None, help="Tile size (e.g., 640). If not set, uses full image.")
    ap.add_argument("--overlap", type=float, default=0.2, help="Tile overlap fraction [0..1)")
    ap.add_argument("--conf", type=float, default=0.15, help="Confidence threshold")
    ap.add_argument("--nms_iou", type=float, default=0.5, help="NMS IoU threshold across tiles")
    ap.add_argument("--keep", type=str, default="person,boat,surfboard",
                    help="Comma-separated class names to keep (must exist in model.names)")
    ap.add_argument("--remap", action="store_true",
                    help="If set, remap kept classes to compact IDs 0..N-1 in the order of --keep")
    args = ap.parse_args()
    main(args)
