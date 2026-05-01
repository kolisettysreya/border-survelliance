"""
main.py
=======
Entry point for the Perimeter Defense Detection System.
Provides a menu-driven interface to train, configure, or run detection.

Usage:
    python main.py                  # Interactive menu
    python main.py --mode detect    # Jump straight to detection
    python main.py --mode train     # Jump straight to training
    python main.py --mode draw      # Jump straight to line drawing tool
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

# Always resolve all paths relative to where main.py lives
BASE_DIR = Path(__file__).parent.resolve()
os.chdir(BASE_DIR)

BANNER = """
╔══════════════════════════════════════════════════════════╗
║        🛡️  PERIMETER DEFENSE DETECTION SYSTEM 🛡️         ║
║          Bounding Box + Line Crossing Detection          ║
║                  Powered by YOLOv8                       ║
╚══════════════════════════════════════════════════════════╝
"""

MENU = """
  [1]  Run Detection          (detect_crossing.py)
  [2]  Train Model            (train.py)
  [3]  Draw Perimeter Lines   (tools/draw_perimeter.py)
  [4]  Check Setup            (verify all files exist)
  [5]  Exit

"""


def check_setup():
    """Verify all required files and folders exist."""
    print("\n── Setup Check ─────────────────────────────────────")

    required_files = [
        "configs/config.yaml",
        "configs/visdrone.yaml",
        "utils/__init__.py",
        "utils/tracker.py",
        "utils/line_crossing.py",
        "utils/alert.py",
        "utils/logger.py",
        "tools/draw_perimeter.py",
        "detect_crossing.py",
        "train.py",
        "requirements.txt",
    ]

    required_dirs = [
        "configs", "utils", "tools",
        "models", "data", "outputs",
        "outputs/snapshots", "outputs/logs",
    ]

    all_ok = True

    # Check directories
    for d in required_dirs:
        exists = Path(d).is_dir()
        status = "✅" if exists else "❌ MISSING"
        print(f"  {status}  {d}/")
        if not exists:
            all_ok = False
            Path(d).mkdir(parents=True, exist_ok=True)
            print(f"       → Created automatically")

    print()

    # Check files
    for f in required_files:
        exists = Path(f).is_file()
        status = "✅" if exists else "❌ MISSING"
        print(f"  {status}  {f}")
        if not exists:
            all_ok = False

    # Check model weights
    model_exists = Path("models/best.pt").is_file()
    print(f"\n  {'✅' if model_exists else '⚠️  NOT FOUND'}  models/best.pt  {'(train first!)' if not model_exists else ''}")

    # Check dataset
    data_exists = Path("data/VisDrone").is_dir()
    print(f"  {'✅' if data_exists else '⚠️  NOT FOUND'}  data/VisDrone/  {'(download from Kaggle!)' if not data_exists else ''}")

    # Check ultralytics installed
    try:
        import ultralytics
        print(f"  ✅  ultralytics {ultralytics.__version__} installed")
    except ImportError:
        print("  ❌  ultralytics NOT installed  →  run: pip install -r requirements.txt")
        all_ok = False

    # Check OpenCV installed
    try:
        import cv2
        print(f"  ✅  opencv {cv2.__version__} installed")
    except ImportError:
        print("  ❌  opencv NOT installed  →  run: pip install -r requirements.txt")
        all_ok = False

    print()
    if all_ok:
        print("  ✅  All checks passed! Ready to run.\n")
    else:
        print("  ⚠️   Some items need attention (see above).\n")

    return all_ok


def run_detection():
    print("\n── Detection Settings ──────────────────────────────")
    source = input("  Enter video path or '0' for webcam [default: 0]: ").strip() or "0"
    config = input("  Config file path [default: configs/config.yaml]: ").strip() or "configs/config.yaml"

    if not Path(config).exists():
        print(f"  ❌ Config not found: {config}")
        return

    if source != "0" and not Path(source).exists():
        print(f"  ❌ Video not found: {source}")
        return

    print(f"\n  Starting detection on: {source}")
    print("  Press 'q' in the video window to quit.\n")
    subprocess.run([sys.executable, "detect_crossing.py",
                    "--source", source,
                    "--config", config])


def run_training():
    print("\n── Training Settings ───────────────────────────────")
    data    = input("  Dataset YAML [default: configs/visdrone.yaml]: ").strip() or "configs/visdrone.yaml"
    weights = input("  Base weights [default: yolov8m.pt]: ").strip() or "yolov8m.pt"
    epochs  = input("  Epochs [default: 30]: ").strip() or "30"
    device  = input("  Device — 0 for GPU, cpu for CPU [default: 0]: ").strip() or "0"

    print(f"\n  Starting training for {epochs} epochs...\n")
    subprocess.run([sys.executable, "train.py",
                    "--data",    data,
                    "--weights", weights,
                    "--epochs",  epochs,
                    "--device",  device])


def draw_perimeter():
    print("\n── Draw Perimeter Lines ────────────────────────────")
    source = input("  Enter video path or '0' for webcam [default: 0]: ").strip() or "0"
    print("\n  Instructions:")
    print("    • Click and drag to draw a line")
    print("    • Press 'S' to save coordinates → configs/perimeter_coords.yaml")
    print("    • Press 'C' to clear all lines")
    print("    • Press 'Q' to quit\n")
    subprocess.run([sys.executable, "tools/draw_perimeter.py", "--source", source])


def interactive_menu():
    print(BANNER)

    while True:
        print(MENU)
        choice = input("  Enter choice [1-5]: ").strip()

        if choice == "1":
            run_detection()
        elif choice == "2":
            run_training()
        elif choice == "3":
            draw_perimeter()
        elif choice == "4":
            check_setup()
        elif choice == "5":
            print("\n  Goodbye! 🛡️\n")
            sys.exit(0)
        else:
            print("  Invalid choice. Enter 1–5.\n")


def main():
    parser = argparse.ArgumentParser(description="Perimeter Defense Detection System")
    parser.add_argument("--mode", type=str, choices=["detect", "train", "draw", "check"],
                        help="Skip menu and jump to mode directly")
    args = parser.parse_args()

    if args.mode == "detect":
        run_detection()
    elif args.mode == "train":
        run_training()
    elif args.mode == "draw":
        draw_perimeter()
    elif args.mode == "check":
        check_setup()
    else:
        interactive_menu()


if __name__ == "__main__":
    main()