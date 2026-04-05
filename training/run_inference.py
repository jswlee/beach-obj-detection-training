#!/usr/bin/env python3
"""Simple script to run YOLOv8 inference on a directory of images.

- Uses a fixed trained model: training_results/models/beach_detection_20260104_101723_best.pt
- Reads all images from an input directory
- Writes YOLOv8-format txt files (class cx cy w h, normalized) to an output directory

Usage:

    python training/run_simple_inference.py \
        --input-dir path/to/images \
        --output-dir path/to/output_labels

"""

import argparse
from pathlib import Path

from sympy import Float
from ultralytics import YOLO


def run_inference(input_dir: str, output_dir: str, conf: 0.2, imgsz=1920) -> None:
    weights_path = Path("runs/detect/training_results/runs/beach_detection_20260117_212333_26m_v9_mAP740/weights/best.pt")
    model = YOLO(str(weights_path))

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect image files (basic extensions)
    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
        image_files.extend(input_path.glob(ext))

    if not image_files:
        print(f"No images found in {input_path}")
        return

    print(f"Using model: {weights_path}")
    print(f"Input images: {input_path} ({len(image_files)} files)")
    print(f"Output labels: {output_path}")

    for img_path in image_files:
        # Run inference on a single image
        results = model(str(img_path), verbose=False, conf=conf, imgsz=imgsz)[0]

        # Prepare output label file path
        label_path = output_path / f"{img_path.stem}.txt"

        lines = []
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes
            # cls: (N,), xywhn: (N,4) normalized [0,1]
            cls_ids = boxes.cls.cpu().tolist()
            xywhn = boxes.xywhn.cpu().tolist()

            for cls_id, (cx, cy, w, h) in zip(cls_ids, xywhn):
                line = f"{int(cls_id)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                lines.append(line)

        # Write label file (empty file if no detections)
        if lines:
            label_path.write_text("\n".join(lines))
        else:
            label_path.touch()

        print(f"Processed: {img_path.name} -> {label_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLOv8 inference and save YOLO-format labels.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory with input images")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save YOLO-format labels")
    parser.add_argument("--conf", type=Float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=1920)

    args = parser.parse_args()

    run_inference(args.input_dir, args.output_dir, conf=args.conf, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
