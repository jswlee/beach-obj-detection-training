"""
Central configuration for the beach object-detection training pipeline.

All tuneable hyperparameters, file-path conventions, and dataset-specific
constants live here so that every script imports from one place.
Edit values below to change behaviour across the whole project.
"""

from pathlib import Path
import numpy as np


# ---------------------------------------------------------------------------
# Project paths (relative to repo root)
# ---------------------------------------------------------------------------

# Where Roboflow exports land (gitignored)
RAW_DATA_DIR = Path("roboflow_data/Beach Counter.v16i.yolov8")

# Where preprocessed / tiled data is written (gitignored)
PROCESSED_DATA_DIR = Path("processed_data/processed_images_v16_tiled")

# Top-level directory for all Ultralytics training outputs (gitignored)
TRAINING_RESULTS_DIR = Path("training_results")


# ---------------------------------------------------------------------------
# ROI masking – polygon that blacks-out the pool / boardwalk area
# ---------------------------------------------------------------------------
# Vertices were traced with the Roboflow Polygon Zone tool on a 1920×1080
# youtube_snapshot frame.  Only images whose filename matches certain
# patterns get this mask applied (see lib/preprocessing.py).

EXCLUSION_POLYGON = np.array([
    [3, 1077], [1903, 1079], [1911, 1074], [1909, 571], [1852, 585],
    [1851, 610], [1792, 627], [1792, 660], [1759, 677], [1792, 692],
    [1654, 744], [1610, 728], [1446, 777], [1308, 808], [1284, 806],
    [1212, 826], [1156, 816], [906, 866], [806, 909], [806, 924],
    [775, 930], [776, 941], [709, 934], [635, 958], [595, 956],
    [363, 991], [254, 1017], [211, 1023], [188, 1018], [46, 1053],
], dtype=np.int32)


# ---------------------------------------------------------------------------
# Tiling parameters
# ---------------------------------------------------------------------------

TILE_SIZE = 640              # Width and height of each tile (pixels)
OVERLAP_YOUTUBE = 0.2        # Overlap ratio for youtube-snapshot 3-tile layout
OVERLAP_FULL = 0.1           # Overlap ratio for full-resolution grid tiling
MIN_OVERLAP_RATIO = 0.5      # A ground-truth box must overlap a tile by at
                             # least this fraction to be assigned to that tile


# ---------------------------------------------------------------------------
# Dataset class mapping
# ---------------------------------------------------------------------------

# In the raw Roboflow export the "person" class has this ID.
# When --person-only is used we remap it to class 0.
PERSON_CLASS_ID = 3


# ---------------------------------------------------------------------------
# Training hyperparameters (passed to Ultralytics model.train)
# ---------------------------------------------------------------------------

TRAIN_DEFAULTS = {
    # Architecture / checkpoint
    "model": "yolov8n.pt",       # Base model weights (or path to a .pt checkpoint)

    # Schedule
    "epochs": 300,               # Maximum training epochs
    "patience": 20,              # Early-stopping patience (epochs without improvement)
    "batch_size": 2,             # Images per batch
    "img_size": 640,             # Training image resolution (matches tile size)

    # Optimiser
    "optimizer": "auto",         # "auto" lets Ultralytics choose; or "AdamW", "SGD", etc.
    "lr0": 0.001,                # Initial learning rate
    "lrf": 0.01,                 # Final learning-rate fraction (cosine decay target)
    "momentum": 0.937,
    "weight_decay": 0.0005,

    # Warmup
    "warmup_epochs": 5,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,

    # Loss weights
    "box": 9.0,                  # Box regression loss weight
    "cls": 0.5,                  # Classification loss weight
    "dfl": 1.5,                  # Distribution focal loss weight

    # Transfer learning
    "freeze": 0,                 # Number of backbone layers to freeze (0 = train all)

    # Data augmentation
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.15,
    "copy_paste": 0.3,

    # Misc
    "close_mosaic": 10,          # Disable mosaic for the last N epochs
}


# ---------------------------------------------------------------------------
# Inference defaults
# ---------------------------------------------------------------------------

INFERENCE_DEFAULTS = {
    "conf": 0.20,                # Confidence threshold for display / saving
    "metrics_conf": 0.20,        # Confidence threshold for mAP calculation
    "iou_merge": 0.50,           # IoU threshold for cross-tile NMS merge
    "iou_match": 0.50,           # IoU threshold for pred ↔ GT matching
    "num_classes": 4,            # Number of classes in the dataset
}
