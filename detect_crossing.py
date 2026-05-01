"""
detect_crossing.py
==================
Main runtime script for perimeter crossing detection.
Run: python detect_crossing.py --source video.mp4 --config configs/config.yaml
"""

import cv2
import torch
import numpy as np
import yaml
import argparse
import time
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from utils.tracker import CentroidTracker
from utils.line_crossing import PerimeterLine
from utils.alert import AlertSystem
from utils.logger import EventLogger


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def draw_overlay(frame, detections, tracks, perimeter_lines, fps, alerts, total_alerts=0):
    """Draw bounding boxes, tracks, lines, and HUD on frame."""
    h, w = frame.shape[:2]

    # Draw semi-transparent HUD background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (280, 110), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # HUD Info
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)
    cv2.putText(frame, f"Objects: {len(tracks)}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)
    cv2.putText(frame, f"Alerts: {total_alerts}", (10, 75),  # ← FIXED: total_alerts
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 50, 255) if total_alerts > 0 else (0, 255, 100), 2)
    cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # Draw perimeter lines
    for line in perimeter_lines:
        line.draw(frame)

    # Draw detections with class labels
    CLASS_COLORS = {
        "person": (0, 255, 200),
        "vehicle": (255, 180, 0),
        "motorcycle": (200, 100, 255),
        "drone": (0, 180, 255),
        "animal": (255, 100, 100),
    }

    for track_id, bbox, cls_name, conf in detections:
        x1, y1, x2, y2 = map(int, bbox)
        color = CLASS_COLORS.get(cls_name, (200, 200, 200))

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background
        label = f"#{track_id} {cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Centroid dot
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 4, color, -1)

    # Flash red border on alert (only on the frame crossing happens)
    if alerts:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
        cv2.putText(frame, "PERIMETER BREACH", (w // 2 - 160, h - 20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 2)

    return frame


def run(config: dict, source: str):
    # Load model
    model = YOLO(config["model"]["weights"])
    model.conf = config["model"]["confidence"]
    model.iou = config["model"]["iou"]

    # Initialize components
    tracker = CentroidTracker(max_disappeared=config["tracker"]["max_disappeared"])
    alert_system = AlertSystem(config["alerts"])
    logger = EventLogger(config["logging"]["output_dir"])

    # Define perimeter lines from config
    perimeter_lines = [
        PerimeterLine(
            name=line["name"],
            start=tuple(line["start"]),
            end=tuple(line["end"]),
            direction=line.get("direction", "both"),
            color=tuple(line.get("color", [0, 0, 255]))
        )
        for line in config["perimeter"]["lines"]
    ]

    # Open video source
    cap = cv2.VideoCapture(source if source != "0" else 0)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25

    # Video writer (optional)
    out = None
    if config.get("save_output"):
        output_path = Path(config["logging"]["output_dir"]) / f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (W, H))

    prev_time = time.time()
    frame_count = 0
    total_alerts = 0  # ← ADDED: persistent counter across all frames

    print(f"[INFO] Starting detection on: {source}")
    print(f"[INFO] Perimeter lines defined: {[l.name for l in perimeter_lines]}")
    print("[INFO] Press 'q' to quit, 's' to save screenshot\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Run YOLO detection
        results = model(frame, verbose=False)[0]
        boxes = results.boxes

        detections_raw = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            # Map to unified classes
            cls_name = config["class_map"].get(cls_name, cls_name)
            if cls_name in config["model"]["target_classes"]:
                detections_raw.append(([x1, y1, x2, y2], conf, cls_name))

        # Update tracker
        tracked = tracker.update(detections_raw)

        # Check perimeter crossing
        active_alerts = []  # resets every frame — used only for red border flash
        for track_id, (bbox, cls_name, conf) in tracked.items():
            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)

            for line in perimeter_lines:
                crossed, direction = line.check_crossing(track_id, (cx, cy))
                if crossed:
                    event = {
                        "frame": frame_count,
                        "track_id": track_id,
                        "class": cls_name,
                        "line": line.name,
                        "direction": direction,
                        "timestamp": datetime.now().isoformat()
                    }
                    logger.log(event)
                    alert_system.trigger(event, frame)
                    active_alerts.append(event)
                    total_alerts += 1  # ← ADDED: increment persistent counter
                    print(f"[ALERT] Frame {frame_count}: {cls_name} #{track_id} crossed '{line.name}' ({direction})")

        # Compute FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-9)
        prev_time = curr_time

        # Build detections list for drawing
        draw_list = [
            (tid, bbox, cls, conf)
            for tid, (bbox, cls, conf) in tracked.items()
        ]

        # Draw — pass both active_alerts (for flash) and total_alerts (for HUD counter)
        frame = draw_overlay(frame, draw_list, tracked, perimeter_lines, fps, active_alerts, total_alerts)

        if out:
            out.write(frame)

        cv2.imshow("Perimeter Defense System", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            shot_path = f"outputs/screenshot_{frame_count}.jpg"
            cv2.imwrite(shot_path, frame)
            print(f"[INFO] Screenshot saved: {shot_path}")

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print(f"\n[INFO] Done. Total frames: {frame_count}")
    print(f"[INFO] Total alerts triggered: {total_alerts}")
    logger.summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perimeter Crossing Detection System")
    parser.add_argument("--source", type=str, default="0", help="Video path or '0' for webcam")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")
    args = parser.parse_args()

    config = load_config(args.config)
    run(config, args.source)