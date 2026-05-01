"""
tools/draw_perimeter.py
=======================
Interactive tool to draw perimeter lines on a video frame.
Outputs pixel coordinates you can paste directly into configs.yaml.

Usage: python tools/draw_perimeter.py --source your_video.mp4
       python tools/draw_perimeter.py --source 0   (webcam)
Controls:
  Left-click drag  → Draw a line
  'n'              → Name the current line
  's'              → Save all lines to configs/perimeter_coords.yaml
  'c'              → Clear all lines
  'q'              → Quit
"""

import cv2
import yaml
import argparse
from datetime import datetime

lines = []
drawing = False
start_point = None
temp_end = None
current_frame = None
COLORS = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (255, 0, 255), (0, 255, 0)]


def mouse_callback(event, x, y, flags, param):
    global drawing, start_point, temp_end

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        temp_end = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end = (x, y)
        color = COLORS[len(lines) % len(COLORS)]
        name = f"LINE_{len(lines) + 1}"
        lines.append({"name": name, "start": list(start_point), "end": list(end), "color": list(color)})
        print(f"[+] Added: {name} | start={start_point} | end={end}")
        start_point = None
        temp_end = None


def run(source):
    global current_frame

    cap = cv2.VideoCapture(source if source != "0" else 0)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}")
        return

    cv2.namedWindow("Draw Perimeter Lines")
    cv2.setMouseCallback("Draw Perimeter Lines", mouse_callback)

    # Grab a stable frame
    for _ in range(5):
        ret, frame = cap.read()
    cap.release()

    if not ret:
        print("[ERROR] Could not read frame.")
        return

    base_frame = frame.copy()
    print("\n[INSTRUCTIONS]")
    print("  Click + drag to draw a line")
    print("  Press 's' to save coordinates")
    print("  Press 'c' to clear lines")
    print("  Press 'q' to quit\n")

    while True:
        display = base_frame.copy()

        # Draw committed lines
        for i, line in enumerate(lines):
            p1 = tuple(line["start"])
            p2 = tuple(line["end"])
            color = tuple(line["color"])
            cv2.line(display, p1, p2, color, 2)
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            cv2.putText(display, line["name"], mid,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.circle(display, p1, 5, color, -1)
            cv2.circle(display, p2, 5, color, -1)

        # Draw in-progress line
        if drawing and start_point and temp_end:
            cv2.line(display, start_point, temp_end, (200, 200, 200), 1)

        # Instructions overlay
        cv2.putText(display, "Drag to draw | S=save | C=clear | Q=quit",
                    (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Draw Perimeter Lines", display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("c"):
            lines.clear()
            print("[INFO] Lines cleared.")
        elif key == ord("s"):
            save_lines()

    cv2.destroyAllWindows()


def save_lines():
    output = {
        "perimeter": {
            "lines": [
                {
                    "name": l["name"],
                    "start": l["start"],
                    "end": l["end"],
                    "direction": "both",
                    "color": l["color"]
                }
                for l in lines
            ]
        }
    }
    path = "configs/perimeter_coords.yaml"
    with open(path, "w") as f:
        yaml.dump(output, f, default_flow_style=False)
    print(f"\n[SAVED] {path}")
    print("Copy the 'perimeter' section into your configs.yaml\n")
    for l in lines:
        print(f"  {l['name']}: start={l['start']} end={l['end']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0")
    args = parser.parse_args()
    run(args.source)