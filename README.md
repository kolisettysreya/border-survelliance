# 🛡️ Perimeter Defense Detection System — UPDATED

Real-time bounding box detection of **people, vehicles, drones, and animals** crossing virtual perimeter lines — built for defense and surveillance applications using YOLOv8.

---

## 📁 Project Structure

```
perimeter_defense/
├── detect_crossing.py        ← MAIN RUNTIME (run this)
├── train.py                  ← Train YOLOv8 on your dataset
├── requirements.txt
│
├── configs/
│   ├── config.yaml           ← All runtime settings (lines, model, alerts)
│   └── visdrone.yaml         ← Dataset config for training
│
├── utils/
│   ├── tracker.py            ← Centroid tracker (assigns persistent IDs)
│   ├── line_crossing.py      ← Virtual perimeter line logic
│   ├── alert.py              ← Alert system (snapshot, beep, webhook)
│   └── logger.py             ← CSV + JSON event logger
│
├── tools/
│   └── draw_perimeter.py     ← Interactive tool to draw lines on video
│
├── models/                   ← Put your trained best.pt here
├── data/                     ← Put your VisDrone dataset here
└── outputs/
    ├── snapshots/            ← Auto-saved alert screenshots
    └── logs/                 ← CSV + JSON event logs
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download VisDrone dataset
- Kaggle: https://www.kaggle.com/datasets/kushagrapandya/visdrone-dataset
- Roboflow (YOLO-ready): https://universe.roboflow.com/visdrone/visdrone-lzsy1
- Place it in `data/VisDrone/`

### 3. Train the model
```bash
python train.py --data configs/visdrone.yaml --epochs 100 --device 0
```
Best weights saved to: `outputs/training/perimeter_defense/weights/best.pt`
Copy to: `models/best.pt`

### 4. Define your perimeter lines
```bash
python tools/draw_perimeter.py --source your_camera_feed.mp4
```
Draw lines interactively → press `S` to save coords → paste into `configs/config.yaml`

### 5. Run detection
```bash
# On a video file
python detect_crossing.py --source video.mp4 --configs configs/configs.yaml

# On webcam
python detect_crossing.py --source 0 --configs configs/configs.yaml
```

---

## ⚙️ Configuration (configs/config.yaml)

### Define perimeter lines
```yaml
perimeter:
  lines:
    - name: "NORTH FENCE"
      start: [0, 300]        # pixel x,y
      end:   [1280, 300]
      direction: "both"      # 'AtoB', 'BtoA', or 'both'
      color: [0, 0, 255]     # BGR
```

### Set confidence and classes
```yaml
model:
  weights: "models/best.pt"
  confidence: 0.45
  target_classes: [person, vehicle, motorcycle, drone]
```

---

## 🎯 Detection Classes

| ID | Class      | Covers |
|----|------------|--------|
| 0  | person     | pedestrian, human, people |
| 1  | vehicle    | car, van, truck, bus |
| 2  | motorcycle | motorbike, bicycle |
| 3  | drone      | UAV, airborne object |
| 4  | animal     | wildlife (BIRDSAI) |

---

## 📊 Output

Every crossing event is logged automatically:

**CSV log** (`outputs/logs/events_TIMESTAMP.csv`):
```
timestamp,frame,track_id,class,line,direction
2024-01-15T14:32:01,342,7,person,NORTH FENCE,AtoB
```

**Alert snapshots** (`outputs/snapshots/`):
Annotated frame saved on every breach.

---

## 🔔 Alerts

| Alert Type | Config Key | Default |
|------------|-----------|---------|
| Terminal beep | `alerts.beep` | true |
| Save snapshot | `alerts.save_snapshots` | true |
| Webhook POST | `alerts.webhook_url` | null |
| Cooldown | `alerts.cooldown_seconds` | 3 |

---

## 🗝️ Keyboard Controls (during detection)

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save screenshot of current frame |

---

## 📦 Datasets Used

| Dataset | Purpose | Link |
|---------|---------|------|
| VisDrone | Primary training data | kaggle.com/datasets/kushagrapandya/visdrone-dataset |
| HIT-UAV | Thermal/IR night detection | github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset |
| KAIST | Multispectral pedestrian | soonminhwang.github.io/rgbt-ped-detection |