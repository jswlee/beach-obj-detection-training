"""Quick script to fine-tune from a checkpoint with a new optimizer."""
import sys, argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from utils import patch_torch_load

patch_torch_load()

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", help="Path to .pt checkpoint")
    p.add_argument("--data", default="processed_data/processed_images_v16_tiled/data.yaml")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--optimizer", default="AdamW")
    p.add_argument("--lr0", type=float, default=0.0001)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--name", default=None)
    args = p.parse_args()

    model = YOLO(args.checkpoint)
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device="cuda",
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=0.01,
        patience=args.patience,
        warmup_epochs=5,
        resume=False,
        amp=False,
        project="training_results/runs",
        name=args.name or "finetune",
        save=True,
        val=True,
        plots=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
