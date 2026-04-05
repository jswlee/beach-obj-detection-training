#!/usr/bin/env python3
"""
Unified preprocessing and training workflow for YOLO beach detection model.
This script:
1. Preprocesses raw annotated images by applying ROI masks (if not already done)
2. Trains a YOLO model on the processed data
3. Saves all training metrics and parameters to a text file
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime

# Add preprocessing directory to path
sys.path.append(str(Path(__file__).parent.parent / "preprocessing"))

from slice_and_preprocess import process_yolo_dataset

# Import training function
from train_yolo_simple import train_yolo_model, run_test_inference, run_merged_test_inference

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess and train YOLO model for beach detection"
    )
    
    # Preprocessing arguments
    parser.add_argument("--raw-data-dir", type=str, default="roboflow_data/Beach Counter.v16i.yolov8", help="Directory containing raw annotated data from Roboflow")
    parser.add_argument("--processed-data-dir", type=str, default="processed_data/processed_images_v16_tiled", help="Directory for processed (ROI-masked) images")
    parser.add_argument("--skip-preprocessing", action="store_true", help="Skip preprocessing if processed data already exists")
    parser.add_argument("--enable-slicing", action="store_true", help="Enable image slicing into 640x640 tiles with overlap")
    parser.add_argument("--person-only", action="store_true", help="Keep only person class annotations (filter out all other classes)")
    parser.add_argument("--person-class-id", type=int, default=3, help="Class ID for person in the dataset (default: 3)")
    parser.add_argument("--min-pixel-size", type=int, default=0, help="Filter out annotations where max(width,height) < this many pixels. Appends '_sub{N}px_removed' to output dir name.")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=300, help="Total number of training epochs (resuming from a checkpoint will continue training until this total is reached)")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for training (increased for larger dataset)")
    parser.add_argument("--img-size", type=int, default=640, help="Image size for training (should match tile size)")
    # parser.add_argument("--model", type=str, default="training_results/models/beach_detection_20251125_133123.pt", help="Model to use for training")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Model to use for training")
    parser.add_argument("--run-name", type=str, default=None, help="Name for this training run")
    parser.add_argument("--freeze", type=int, default=0, help="Number of layers to freeze")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate (increased for larger dataset)")
    parser.add_argument("--patience", type=int, default=20, help="Patience for early stopping")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Number of warmup epochs (reduced for larger dataset)")
    parser.add_argument("--optimizer", type=str, default="auto", help="Optimizer to use for training")
    parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold for inference")
    parser.add_argument("--p2", action="store_true", help="Use P2 (stride-4) architecture variant for small-object detection")
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help=(
            "Optional checkpoint (.pt) to resume from, e.g. "
            "runs/…/weights/last.pt. Set --epochs to the total target epoch count "
            "(e.g. 131 to continue from epoch 130 for one more epoch)."
        ),
    )
    
    args = parser.parse_args()
    
    # Convert paths to absolute
    raw_data_path = Path(args.raw_data_dir).resolve()
    processed_data_dir = args.processed_data_dir
    if args.min_pixel_size > 0:
        processed_data_dir = f"{processed_data_dir}_sub{args.min_pixel_size}px_removed"
    processed_data_path = Path(processed_data_dir).resolve()
    
    print("=" * 70)
    print("Beach Detection - Unified Preprocessing and Training Workflow")
    print("=" * 70)
    print()
    
    # Step 1: Preprocessing
    dataset_yaml_path = processed_data_path / "data.yaml"
    
    if args.skip_preprocessing and dataset_yaml_path.exists():
        print(f"Skipping preprocessing - using existing data at: {processed_data_path}")
        print()
    else:
        print("Step 1: Preprocessing raw data with ROI masks")
        print("-" * 70)
        print(f"Input:  {raw_data_path}")
        print(f"Output: {processed_data_path}")
        print()
        
        if not raw_data_path.exists():
            print(f"Error: Raw data directory not found: {raw_data_path}")
            return 1
        
        try:
            # Process the dataset with ROI masks
            dataset_yaml_path = process_yolo_dataset(
                str(raw_data_path),
                str(processed_data_path),
                enable_slicing=args.enable_slicing,
                person_only=args.person_only,
                person_class_id=args.person_class_id,
                img_size=args.img_size,
                min_pixel_size=args.min_pixel_size,
            )
            print()
            print(f"Preprocessing complete!")
            print(f"  Dataset YAML: {dataset_yaml_path}")
            print()
        except Exception as e:
            print(f"Preprocessing failed: {e}")
            return 1
    
    # Step 2: Training
    print("Step 2: Training YOLO model")
    print("-" * 70)
    
    # Generate run name if not provided
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"beach_detection_{timestamp}"
    
    print(f"Run name: {args.run_name}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.img_size}")
    print(f"Freeze layers: {args.freeze}")
    print(f"Learning rate: {args.lr0}")
    print(f"Patience: {args.patience}")
    print(f"Warmup epochs: {args.warmup_epochs}")
    print(f"Optimizer: {args.optimizer}")
    print()
    
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
        
        # Run test-set inference and save predictions/metrics alongside training outputs
        # Ultralytics creates: training_results/detect/{run_name}
        run_dir = Path("training_results") / "detect" / args.run_name
        if args.enable_slicing:
            # Merged tiled evaluation: run inference on raw images, NMS-merge
            # overlapping tile detections, then compute image-level metrics.
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

        print()
        print("=" * 70)
        print("✓ Workflow completed successfully!")
        print("=" * 70)
        print(f"Trained model: {model_path}")
        print(f"Training results: {run_dir}")
        print(f"Test inference results: {run_dir / 'test_inference'}")
        print()
        
        return 0
        
    except Exception as e:
        print()
        print("=" * 70)
        print(f"❌ Training failed: {e}")
        print("=" * 70)
        return 1

if __name__ == "__main__":
    sys.exit(main())