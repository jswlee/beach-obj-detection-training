# Beach Object Detection Training

YOLO-based pipeline for detecting people on beach webcam feeds. Handles ROI masking (to exclude pools/boardwalks), image tiling for small-object detection, and full training + evaluation workflows.

## Project Structure

```
├── config.py                          # All hyperparameters, paths, and constants
├── lib/                               # Shared library modules
│   ├── __init__.py
│   ├── preprocessing.py               #   ROI masking, tiling, dataset processing
│   ├── metrics.py                     #   NMS, IoU, mAP calculations
│   └── utils.py                       #   Device selection, torch patch, summaries
├── scripts/                           # Runnable scripts
│   ├── preprocess_and_train.py        #   Main training pipeline
│   ├── run_tiled_inference.py         #   Standalone tiled inference
│   └── analyze_annotation_sizes.py    #   Annotation size analysis tool
├── annotation_analysis/               # Output from annotation analysis (checked in)
│   ├── histogram_*.png                #   Per-split and aggregate size histograms
│   └── summary_statistics.txt         #   Detailed annotation size statistics
├── check_architecture.py              # Utility to inspect P2 vs P3 model architecture
├── install_cuda.sh                    # Install PyTorch + deps with CUDA 12.1 (general)
├── install_cuda_linux.sh              # Install PyTorch + deps with CUDA 12.1 (Linux)
├── roboflow_data/                     # Raw Roboflow exports (gitignored)
├── processed_data/                    # Preprocessed / tiled datasets (gitignored)
├── training_results/                  # Training outputs and saved models (gitignored)
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites

- Python 3.10+
- (Optional) NVIDIA GPU with CUDA 12.1 for GPU-accelerated training

### Install dependencies

```bash
pip install -r requirements.txt
```

For **GPU support**, install PyTorch with CUDA first:

```bash
# Windows / general
bash install_cuda.sh

# Linux
bash install_cuda_linux.sh
```

Or manually:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
pip install -r requirements.txt
```

### Dataset

Place your Roboflow dataset export in `roboflow_data/` with the standard YOLO directory layout:

```
roboflow_data/Beach Counter.v16i.yolov8/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

## Usage

### Training (preprocess + train + evaluate)

The main entry point is `scripts/preprocess_and_train.py`. It runs three steps in sequence:

1. **Preprocess** — ROI masking, optional tiling, class filtering, small-annotation removal.
2. **Train** — YOLO model training via Ultralytics.
3. **Evaluate** — Test-set inference (standard or merged-tiled, depending on whether slicing is enabled).

```bash
# Defaults: tiling enabled, person-only, filter annotations < 4px
python scripts/preprocess_and_train.py

# Customise tiling and filtering
python scripts/preprocess_and_train.py --enable-slicing --person-only --min-pixel-size 6

# Skip preprocessing if processed data already exists
python scripts/preprocess_and_train.py --skip-preprocessing

# Resume training from a checkpoint
python scripts/preprocess_and_train.py --resume-from training_results/.../weights/last.pt --epochs 350

# Use P2 (stride-4) architecture for small-object detection
python scripts/preprocess_and_train.py --p2 --model yolov8m.pt

# Override training hyperparameters
python scripts/preprocess_and_train.py --lr0 0.0005 --batch-size 4 --patience 30 --optimizer AdamW
```

#### Preprocessing CLI flags

| Flag | Default | Description |
|---|---|---|
| `--raw-data-dir` | `roboflow_data/Beach Counter.v16i.yolov8` | Raw annotated data directory |
| `--processed-data-dir` | `processed_data/processed_images_v16_tiled` | Output directory for processed data |
| `--skip-preprocessing` | off | Skip if processed data already exists |
| `--enable-slicing` | on | Slice images into 640×640 tiles with overlap |
| `--person-only` | on | Keep only person-class annotations |
| `--person-class-id` | `3` | Class ID for person in the raw dataset |
| `--min-pixel-size` | `4` | Filter annotations where √(w×h) < N pixels |

#### Training CLI flags

| Flag | Default | Description |
|---|---|---|
| `--model` | `yolov8n.pt` | Base model weights or checkpoint path |
| `--epochs` | `300` | Maximum training epochs |
| `--batch-size` | `2` | Images per batch |
| `--img-size` | `640` | Training image resolution |
| `--lr0` | `0.001` | Initial learning rate |
| `--patience` | `20` | Early-stopping patience (epochs) |
| `--optimizer` | `auto` | Optimizer (`auto`, `AdamW`, `SGD`, etc.) |
| `--freeze` | `0` | Number of backbone layers to freeze |
| `--warmup-epochs` | `5` | Warmup epoch count |
| `--p2` | off | Use P2 (stride-4) architecture |
| `--resume-from` | — | Checkpoint `.pt` to resume from |
| `--conf` | `0.20` | Confidence threshold for post-training eval |
| `--run-name` | auto-generated | Name for the training run |

### Standalone Tiled Inference

`scripts/run_tiled_inference.py` runs tiled YOLO inference on full-resolution images with cross-tile NMS merge. Supports visualisation, label export, live preview, and COCO-style mAP evaluation.

```bash
# Run inference with annotated image output
python scripts/run_tiled_inference.py --model path/to/best.pt --save-img --save-txt

# Calculate mAP / precision / recall against ground truth
python scripts/run_tiled_inference.py --model path/to/best.pt --calc-metrics

# Live preview window (press 'q' to quit)
python scripts/run_tiled_inference.py --model path/to/best.pt --show

# FP16 inference on CUDA
python scripts/run_tiled_inference.py --model path/to/best.pt --half --save-img

# Loop continuously over images with a frame delay
python scripts/run_tiled_inference.py --model path/to/best.pt --show --loop --delay-ms 100
```

| Flag | Default | Description |
|---|---|---|
| `--model` | *(required)* | Path to trained YOLO `.pt` model |
| `--input-dir` | `roboflow_data/.../test/images` | Directory with input images |
| `--output-dir` | auto (next to model) | Output directory |
| `--conf` | `0.20` | Confidence threshold for display / saving |
| `--metrics-conf` | `0.20` | Confidence threshold for mAP calculation |
| `--tile-size` | `640` | Tile size in pixels (must match training) |
| `--iou-merge` | `0.50` | IoU threshold for cross-tile NMS merge |
| `--device` | `auto` | `auto`, `cpu`, or `cuda` |
| `--save-img` | off | Save annotated full-size images |
| `--save-txt` | off | Save YOLO-format label files |
| `--half` | off | Use FP16 inference on CUDA |
| `--show` | off | Live preview window |
| `--loop` | off | Loop over images continuously |
| `--delay-ms` | `0` | Delay between frames (ms) |
| `--max-frames` | `0` (unlimited) | Stop after N frames |
| `--calc-metrics` | off | Calculate mAP metrics (requires GT labels) |
| `--num-classes` | `4` | Number of classes in the dataset |
| `--gt-person-class-id` | `3` | Person class ID in ground-truth labels |

### Annotation Analysis

`scripts/analyze_annotation_sizes.py` calculates bounding-box sizes (`√(w_px × h_px)`) for person annotations across train/valid/test splits and produces histograms plus a summary statistics file.

```bash
# Analyse the default dataset
python scripts/analyze_annotation_sizes.py

# Specify a custom dataset and output directory
python scripts/analyze_annotation_sizes.py --dataset_path path/to/dataset --output_dir results/
```

Output is written to `annotation_analysis/` by default (histograms + `summary_statistics.txt`).

### Check Model Architecture

`check_architecture.py` inspects a trained YOLO checkpoint to determine whether it uses the P2 (stride-4) or P3 (stride-8) detection head.

```bash
python check_architecture.py
```

> **Note:** Edit the `model_path` variable inside the script to point at the checkpoint you want to inspect.

## Configuration

All default hyperparameters, file paths, and dataset-specific constants are centralised in `config.py`. Edit that file to change defaults globally, or override individual values via CLI flags.

Key sections in `config.py`:

- **Project paths** — `RAW_DATA_DIR`, `PROCESSED_DATA_DIR`, `TRAINING_RESULTS_DIR`
- **ROI masking** — `EXCLUSION_POLYGON` vertices (traced on 1920×1080 frames) for blacking out pools/boardwalks
- **Tiling parameters** — `TILE_SIZE` (640), `OVERLAP_YOUTUBE` (0.2), `OVERLAP_FULL` (0.1), `MIN_OVERLAP_RATIO` (0.5)
- **Class mapping** — `PERSON_CLASS_ID` (3 in the raw Roboflow export, remapped to 0 when `--person-only`)
- **Training hyperparameters** (`TRAIN_DEFAULTS`) — optimizer, learning rate schedule, warmup, loss weights (box/cls/dfl), augmentation (HSV, mosaic, mixup, copy-paste, etc.)
- **Inference defaults** (`INFERENCE_DEFAULTS`) — confidence thresholds, NMS IoU, number of classes

## Library Modules

### `lib/preprocessing.py`

- **`process_yolo_dataset()`** — End-to-end dataset processing: ROI masking, optional tiling, class filtering, small-annotation removal, `data.yaml` generation.
- **`create_roi_mask()` / `get_roi_mask_u8()`** — Build binary exclusion masks from the polygon in `config.py`.
- **`get_tiles()` / `get_bottom_three_tiles()` / `get_tiles_for_image()`** — Tiling strategies: full overlapping grid for high-res images, 3-tile bottom band for youtube snapshots.
- **`crop_and_mask_tiles()`** — Crop tiles from an image with optional ROI masking applied.
- **`is_youtube_snapshot()` / `should_apply_mask()`** — Filename-based routing for masking and tiling strategy.

### `lib/metrics.py`

- **`nms_merge()`** — Class-aware Non-Maximum Suppression (pure numpy, class-offset trick).
- **`box_iou_numpy()`** — Pairwise IoU between two sets of xyxy boxes.
- **`calculate_map_metrics()`** — COCO-style mAP@0.5 and mAP@0.5:0.95 with all-point interpolation.
- **`compute_ap()`** — 101-point interpolated Average Precision.

### `lib/utils.py`

- **`patch_torch_load()`** — Monkey-patch for PyTorch 2.6+ `weights_only` default change (required for loading Ultralytics checkpoints).
- **`select_device()`** — Auto-detect best compute device (CUDA → MPS → CPU).
- **`collect_images()`** — Gather image file paths from a directory.
- **`save_training_summary()`** — Write a detailed training report (config, best/final metrics, epoch-by-epoch table) from Ultralytics outputs.
- **`save_test_summary()`** — Write a test-set evaluation report with overall and per-class metrics.

## Key Dependencies

| Package | Purpose |
|---|---|
| `torch` / `torchvision` | Deep learning framework (CUDA 12.1 recommended) |
| `ultralytics` | YOLOv8 training, inference, and evaluation |
| `opencv-python` | Image I/O, masking, drawing |
| `numpy` | Array operations, NMS, IoU |
| `pandas` | Reading training results CSVs |
| `matplotlib` / `seaborn` | Histogram and analysis plots |
| `Pillow` | Image dimension reading |
| `tqdm` | Progress bars |
| `PyYAML` | Dataset YAML read/write |
