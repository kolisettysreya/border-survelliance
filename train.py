"""
train.py
========
Train YOLOv8 on VisDrone (or any YOLO-formatted dataset).
Run: python train.py --data configs/visdrone.yaml --epochs 100
"""

import argparse
from ultralytics import YOLO
from pathlib import Path


def train(data_yaml: str, weights: str, epochs: int, imgsz: int, batch: int, device: str):
    model = YOLO(weights)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="outputs/training",
        name="perimeter_defense",
        save=True,
        save_period=10,
        val=True,
        patience=10,             # Early stopping
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        flipud=0.1,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        plots=True,
        verbose=True
    )

    print(f"\n[INFO] Training complete. Best weights: outputs/training/perimeter_defense/weights/best.pt")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    type=str, default="configs/visdrone.yaml")
    parser.add_argument("--weights", type=str, default="yolov8m.pt",  # medium = best balance
                        help="yolov8n/s/m/l/x.pt or path to custom .pt")
    parser.add_argument("--epochs",  type=int, default=30)
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--batch",   type=int, default=16)
    parser.add_argument("--device",  type=str, default="0",  # GPU 0 or 'cpu'
                        help="GPU device id or 'cpu'")
    args = parser.parse_args()

    train(args.data, args.weights, args.epochs, args.imgsz, args.batch, args.device)