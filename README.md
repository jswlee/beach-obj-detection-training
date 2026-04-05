# Beach Object Detection Training

YOLO-based pipeline for detecting people on beach webcam feeds. Handles ROI masking (to exclude pools/boardwalks), image tiling for small-object detection, and full training + evaluation workflows.

## Project Structure

```
├── config.py                          # All hyperparameters, paths, and constants
├── lib/                               # Shared library modules
│   ├── preprocessing.py               #   ROI masking, tiling, dataset processing
│   ├── metrics.py                     #   NMS, IoU, mAP calculations
│   └── utils.py                       #   Device selection, torch patch, summaries
├── scripts/                           # Runnable scripts
│   ├── preprocess_and_train.py        #   Main training pipeline
│   ├── run_tiled_inference.py         #   Standalone tiled inference
│   └── analyze_annotation_sizes.py    #   Annotation size analysis tool
├── roboflow_data/                     #   Raw Roboflow exports (gitignored)
├── processed_data/                    #   Preprocessed datasets (gitignored)
├── training_results/                  #   Training outputs (gitignored)
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Place your Roboflow dataset export in `roboflow_data/` (e.g. `roboflow_data/Beach Counter.v16i.yolov8/`).

## Usage

### Training (preprocess + train + evaluate)

```bash
# Basic: preprocess raw data, train, and run test evaluation
python scripts/preprocess_and_train.py

# With tiling (640x640 tiles), person-only, filter tiny annotations
python scripts/preprocess_and_train.py --enable-slicing --person-only --min-pixel-size 6

# Skip preprocessing if already done
python scripts/preprocess_and_train.py --skip-preprocessing

# Resume from a checkpoint
python scripts/preprocess_and_train.py --resume-from runs/.../weights/last.pt --epochs 350
```

### Standalone Tiled Inference

```bash
# Run inference on test images with visualisation
python scripts/run_tiled_inference.py --model path/to/best.pt --save-img --save-txt

# Calculate mAP metrics against ground truth
python scripts/run_tiled_inference.py --model path/to/best.pt --calc-metrics

# Live preview
python scripts/run_tiled_inference.py --model path/to/best.pt --show
```

### Annotation Analysis

```bash
python scripts/analyze_annotation_sizes.py
python scripts/analyze_annotation_sizes.py --dataset_path path/to/dataset --output_dir results/
```

## Configuration

All default hyperparameters, file paths, and dataset-specific constants are centralised in `config.py`. Edit that file to change defaults globally, or override individual values via CLI flags.

Key sections in `config.py`:
- **Project paths** — where raw data, processed data, and results live
- **ROI masking** — exclusion polygon vertices
- **Tiling parameters** — tile size, overlap ratios
- **Training hyperparameters** — learning rate, augmentation, loss weights, etc.
- **Inference defaults** — confidence thresholds, NMS settings
