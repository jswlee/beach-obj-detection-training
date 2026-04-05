"""
Detection metrics and post-processing utilities.

Provides:
  - nms_merge()            – class-aware Non-Maximum Suppression (pure numpy)
  - box_iou_numpy()        – pairwise IoU between two sets of xyxy boxes
  - calculate_map_metrics() – COCO-style mAP@0.5 and mAP@0.5:0.95
  - compute_ap()           – 101-point interpolated Average Precision

These are used by both the training post-eval and the standalone tiled
inference script.
"""

import numpy as np
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Non-Maximum Suppression (class-aware, pure numpy)
# ---------------------------------------------------------------------------

def nms_merge(
    boxes: np.ndarray,
    scores: np.ndarray,
    cls_ids: np.ndarray,
    iou_thr: float,
) -> np.ndarray:
    """Class-wise NMS using the class-offset trick + greedy suppression.

    Args:
        boxes:   (N, 4) float32 array in xyxy format.
        scores:  (N,) float32 confidence scores.
        cls_ids: (N,) int class IDs.
        iou_thr: IoU threshold – boxes above this are suppressed.

    Returns:
        1-D int array of kept indices into the original arrays.
    """
    if len(boxes) == 0:
        return np.empty(0, dtype=np.intp)

    # Offset boxes by class so cross-class boxes never overlap
    max_coord = boxes.max() + 1.0
    offsets = cls_ids.astype(np.float32) * max_coord
    x1 = boxes[:, 0] + offsets
    y1 = boxes[:, 1] + offsets
    x2 = boxes[:, 2] + offsets
    y2 = boxes[:, 3] + offsets

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter)
        order = rest[iou <= iou_thr]

    return np.array(keep, dtype=np.intp)


# ---------------------------------------------------------------------------
# IoU computation
# ---------------------------------------------------------------------------

def box_iou_numpy(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of xyxy boxes.

    Args:
        boxes1: (N, 4) float32
        boxes2: (M, 4) float32

    Returns:
        (N, M) float32 IoU matrix.
    """
    x1 = np.maximum(boxes1[:, 0:1], boxes2[:, 0:1].T)
    y1 = np.maximum(boxes1[:, 1:2], boxes2[:, 1:2].T)
    x2 = np.minimum(boxes1[:, 2:3], boxes2[:, 2:3].T)
    y2 = np.minimum(boxes1[:, 3:4], boxes2[:, 3:4].T)
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    return inter / (area1[:, None] + area2[None, :] - inter + 1e-7)


# ---------------------------------------------------------------------------
# COCO-style mAP
# ---------------------------------------------------------------------------

def calculate_map_metrics(
    all_predictions: List[List[Tuple[float, float, float, float, float, int]]],
    all_ground_truth: List[List[Tuple[int, float, float, float, float]]],
    img_sizes: List[Tuple[int, int]],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    num_classes: int = 4,
) -> Dict[str, float]:
    """Calculate COCO-style mAP metrics across a set of images.

    Performs greedy IoU matching at 10 thresholds (0.50 … 0.95) and
    computes per-class AP with all-point interpolation.

    Args:
        all_predictions:  Per-image list of (x1, y1, x2, y2, conf, cls_id).
        all_ground_truth: Per-image list of (cls_id, cx, cy, bw, bh) in
                          normalised YOLO format.
        img_sizes:        Per-image (width, height).
        conf_threshold:   Minimum confidence to count a prediction.
        iou_threshold:    (unused – kept for API compat; thresholds are 0.5–0.95).
        num_classes:      Total number of dataset classes.

    Returns:
        Dict with keys: map50, map50_95, precision, recall, f1.
    """
    _zero = {"map50": 0.0, "map50_95": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not all_predictions or not all_ground_truth:
        return _zero

    iou_thresholds = np.linspace(0.5, 0.95, 10)
    n_iou = len(iou_thresholds)

    all_tp: list[np.ndarray] = []
    all_conf: list[np.ndarray] = []
    all_pred_cls: list[np.ndarray] = []
    all_target_cls: list[np.ndarray] = []

    for preds, gts, (w, h) in zip(all_predictions, all_ground_truth, img_sizes):
        # --- Predictions ---
        if preds:
            pred_arr = np.array(preds, dtype=np.float32)
            mask = pred_arr[:, 4] >= conf_threshold
            pred_arr = pred_arr[mask]
        else:
            pred_arr = np.zeros((0, 6), dtype=np.float32)

        # --- Ground truth (YOLO normalised → pixel xyxy) ---
        if gts:
            gt_cls_l, gt_box_l = [], []
            for cls, cx, cy, bw, bh in gts:
                gt_cls_l.append(int(cls))
                gt_box_l.append([
                    (cx - bw / 2) * w, (cy - bh / 2) * h,
                    (cx + bw / 2) * w, (cy + bh / 2) * h,
                ])
            gt_cls = np.array(gt_cls_l, dtype=np.int32)
            gt_boxes = np.array(gt_box_l, dtype=np.float32)
        else:
            gt_cls = np.zeros(0, dtype=np.int32)
            gt_boxes = np.zeros((0, 4), dtype=np.float32)

        npr, ngt = pred_arr.shape[0], gt_cls.shape[0]
        all_target_cls.append(gt_cls)

        if npr == 0:
            all_tp.append(np.zeros((0, n_iou), dtype=bool))
            all_conf.append(np.zeros(0, dtype=np.float32))
            all_pred_cls.append(np.zeros(0, dtype=np.int32))
            continue

        pred_boxes = pred_arr[:, :4]
        pred_conf = pred_arr[:, 4]
        pred_cls = pred_arr[:, 5].astype(np.int32)
        all_conf.append(pred_conf)
        all_pred_cls.append(pred_cls)

        if ngt == 0:
            all_tp.append(np.zeros((npr, n_iou), dtype=bool))
            continue

        iou_mat = box_iou_numpy(pred_boxes, gt_boxes)
        cls_match = pred_cls[:, None] == gt_cls[None, :]

        tp = np.zeros((npr, n_iou), dtype=bool)
        for t, thr in enumerate(iou_thresholds):
            valid = (iou_mat >= thr) & cls_match
            pi, gi = np.where(valid)
            if pi.size == 0:
                continue
            ious = iou_mat[pi, gi]
            order = ious.argsort()[::-1]
            pi, gi = pi[order], gi[order]
            matched_pred: set[int] = set()
            matched_gt: set[int] = set()
            for p, g in zip(pi, gi):
                if p in matched_pred or g in matched_gt:
                    continue
                tp[p, t] = True
                matched_pred.add(p)
                matched_gt.add(g)
        all_tp.append(tp)

    # --- Aggregate across images ---
    tp = np.concatenate(all_tp, axis=0) if all_tp else np.zeros((0, n_iou), dtype=bool)
    conf = np.concatenate(all_conf, axis=0) if all_conf else np.zeros(0, dtype=np.float32)
    p_cls = np.concatenate(all_pred_cls, axis=0) if all_pred_cls else np.zeros(0, dtype=np.int32)
    t_cls = np.concatenate(all_target_cls, axis=0) if all_target_cls else np.zeros(0, dtype=np.int32)

    if conf.size == 0 and t_cls.size == 0:
        return _zero

    order = conf.argsort()[::-1]
    tp = tp[order]
    conf = conf[order]
    p_cls = p_cls[order]

    unique_cls = np.unique(np.concatenate([p_cls, t_cls]))
    ap = np.zeros((len(unique_cls), n_iou))

    for ci, c in enumerate(unique_cls):
        c_mask = p_cls == c
        n_gt_c = (t_cls == c).sum()
        if n_gt_c == 0 or c_mask.sum() == 0:
            continue
        tp_c = tp[c_mask]
        for t in range(n_iou):
            tp_cum = np.cumsum(tp_c[:, t])
            fp_cum = np.cumsum(~tp_c[:, t])
            rec = tp_cum / (n_gt_c + 1e-16)
            prec = tp_cum / (tp_cum + fp_cum + 1e-16)
            # All-point interpolation
            mrec = np.concatenate(([0.0], rec, [1.0]))
            mpre = np.concatenate(([1.0], prec, [0.0]))
            for j in range(len(mpre) - 1, 0, -1):
                mpre[j - 1] = max(mpre[j - 1], mpre[j])
            idx = np.where(mrec[1:] != mrec[:-1])[0]
            ap[ci, t] = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])

    map50 = float(ap[:, 0].mean()) if ap.size > 0 else 0.0
    map50_95 = float(ap.mean()) if ap.size > 0 else 0.0

    total_tp = float(tp[:, 0].sum())
    precision = total_tp / (len(tp) + 1e-16)
    recall = total_tp / (len(t_cls) + 1e-16)
    f1 = 2 * precision * recall / (precision + recall + 1e-16)

    return {
        "map50": map50,
        "map50_95": map50_95,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def compute_ap(
    all_detections: list[tuple[float, bool, int]],
    total_gt: int,
) -> float:
    """Compute Average Precision using 101-point COCO interpolation.

    Args:
        all_detections: List of (confidence, is_true_positive, class_id).
        total_gt:       Total number of ground-truth boxes.

    Returns:
        AP value (float in [0, 1]).
    """
    if not all_detections or total_gt == 0:
        return 0.0

    sorted_dets = sorted(all_detections, key=lambda x: -x[0])

    tp_cumsum = 0
    fp_cumsum = 0
    precisions: list[float] = []
    recalls: list[float] = []

    for _conf, is_tp, _cls in sorted_dets:
        if is_tp:
            tp_cumsum += 1
        else:
            fp_cumsum += 1
        precisions.append(tp_cumsum / (tp_cumsum + fp_cumsum))
        recalls.append(tp_cumsum / total_gt)

    prec_np = np.array(precisions)
    rec_np = np.array(recalls)

    # Make precision monotonically decreasing
    for i in range(len(prec_np) - 2, -1, -1):
        prec_np[i] = max(prec_np[i], prec_np[i + 1])

    # 101-point interpolation
    recall_thresholds = np.linspace(0, 1, 101)
    ap = 0.0
    for t in recall_thresholds:
        mask = rec_np >= t
        if mask.any():
            ap += prec_np[mask].max()
    ap /= 101.0

    return float(ap)
