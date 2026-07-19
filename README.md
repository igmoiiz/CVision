# VisionSense 🎯

**A production-quality, real-time computer vision desktop application built with Python, OpenCV, Ultralytics YOLO11, and ByteTrack.**

VisionSense provides a modular, config-driven pipeline for live object detection, multi-object tracking, instance segmentation, pose estimation, analytics, line/region counting, video recording, snapshot capture, and custom model fine-tuning — all from a single webcam.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎥 **Webcam Streaming** | Configurable resolution, FPS, and horizontal flip |
| 🧠 **YOLO11 Detection** | Fast, accurate object detection (COCO 80 classes) |
| 🎭 **Instance Segmentation** | Per-object alpha-blended colour masks |
| 🦴 **Pose Estimation** | 17-keypoint human skeleton overlays |
| 🔢 **ByteTrack Tracking** | Persistent IDs across frames with centroid trails |
| 📊 **Analytics Engine** | Current / cumulative / peak per-class counts + heatmap |
| ↔️ **Line Crossing Counter** | Bidirectional IN/OUT counting with per-class breakdown |
| 🔷 **Region Counter** | Polygon zone occupancy counter |
| 🎬 **Video Recording** | MP4/AVI with codec selection, timestamped filenames |
| 📷 **Snapshot Capture** | Full-res annotated PNG screenshots |
| ⚙️ **Config System** | YAML-driven — zero hardcoded values |
| 📦 **Dataset Collector** | Capture YOLO-format training data from live feed |
| 🎓 **Fine-Tuning Trainer** | One-call custom YOLO training on your dataset |
| 📈 **FPS & Inference Metrics** | Rolling-window FPS + per-frame inference timer |
| 🖥️ **HUD Overlay** | Semi-transparent, toggleable heads-up display |

---

## 🗂️ Project Structure

```
CVision/
├── main.py                     # Application entry point
├── config.yaml                 # Runtime configuration (edit this!)
├── requirements.txt
│
├── src/
│   ├── config.py               # Dataclass config system (YAML I/O)
│   ├── camera.py               # Webcam capture
│   ├── detector.py             # YOLO detect / segment / pose
│   ├── fps.py                  # FPSCounter + InferenceTimer
│   ├── analytics.py            # Per-class stats + heatmap + CSV export
│   ├── line_counter.py         # Line crossing + region polygon counter
│   ├── renderer.py             # Full visual overlay renderer
│   ├── recorder.py             # Video recording
│   ├── snapshot.py             # Screenshot capture
│   ├── dataset_collector.py    # Training data collection
│   └── trainer.py              # YOLO fine-tuning wrapper
│
├── models/                     # YOLO weights (auto-downloaded)
│   └── yolo11n.pt
│
├── datasets/
│   └── raw/                    # Collected training frames + labels
│       ├── images/
│       ├── labels/
│       └── dataset.yaml
│
├── outputs/
│   ├── recordings/             # MP4 recordings
│   ├── screenshots/            # PNG snapshots
│   └── visionsense.log         # Application log
│
└── assets/                     # Icons and custom assets
```

---

## 🚀 Prerequisites

- **Python** 3.10 or newer
- **Webcam** accessible at index 0 (configurable)
- (Optional) **NVIDIA GPU** with CUDA 12.x for accelerated inference
- (Optional) **Apple Silicon** — MPS backend supported

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/CVision.git
cd CVision
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

**CPU only (recommended for testing):**
```bash
pip install -r requirements.txt
```

**NVIDIA GPU (CUDA 12.x):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 4. Download a YOLO model (automatic on first run)

VisionSense auto-downloads `yolo11n.pt` on first launch. To pre-download:
```bash
python -c "from ultralytics import YOLO; YOLO('models/yolo11n.pt')"
```

Available models: `yolo11n`, `yolo11s`, `yolo11m`, `yolo11l`, `yolo11x` (n=nano, fastest; x=extra-large, most accurate).

---

## ▶️ Running VisionSense

```bash
python main.py
```

With a custom configuration file:
```bash
python main.py --config my_config.yaml
```

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|---|---|
| `q` / `ESC` | Quit the application |
| `r` | Toggle video **recording** (start / stop) |
| `s` | Save a **snapshot** (screenshot) |
| `h` | Toggle **HUD** overlay |
| `t` | Cycle inference **mode** (detect → segment → pose → detect…) |
| `c` | Toggle **region counter** |
| `l` | Toggle **line counter** |
| `m` | Toggle **heatmap** overlay |
| `d` | Toggle **dataset collection** mode |

---

## ⚙️ Configuration

All parameters live in `config.yaml`. The most commonly changed options:

```yaml
camera:
  index: 0           # Change if you have multiple cameras
  width: 1280
  height: 720

detection:
  model_path: models/yolo11n.pt   # Swap model here
  confidence: 0.45                # Lower = more detections
  mode: detect                    # detect | segment | pose
  device: ""                      # "" = auto, "cpu", "0" = GPU 0

line_counter:
  enabled: true
  start: [0, 360]     # Line start (x, y)
  end: [1280, 360]    # Line end   (x, y)

region_counter:
  enabled: false
  polygon:
    - [200, 100]
    - [1080, 100]
    - [1080, 620]
    - [200, 620]
```

---

## 🎭 Inference Modes

Switch between modes at runtime by pressing **`t`**:

### Detect (default)
Standard bounding-box detection with ByteTrack IDs and centroid trails.

### Segment
Instance segmentation — each detected object gets a unique colour mask overlaid on its shape.  
*Requires a segmentation model, e.g. `yolo11n-seg.pt`.*

### Pose
Human keypoint detection with full 17-point skeleton connections drawn between joints.  
*Requires a pose model, e.g. `yolo11n-pose.pt`.*

> **Tip:** Set `detection.model_path` in `config.yaml` to the appropriate model before switching modes, then restart, or keep all three models ready and swap them via config.

---

## 📊 Analytics

VisionSense tracks:

- **Current** — objects detected in this frame per class
- **Cumulative** — total detections per class since startup
- **Peak** — maximum simultaneous count seen per class
- **Heatmap** — spatial density of object centroids (toggle with `m`)

At shutdown, a CSV report is automatically saved to `outputs/`.

---

## 📦 Dataset Collection

Press **`d`** to enter dataset collection mode:

1. Each collected frame is saved as a JPEG in `datasets/raw/images/`.
2. A matching YOLO-format `.txt` annotation is written to `datasets/raw/labels/`.
3. A `dataset.yaml` is automatically generated for training.

Auto-collection at fixed frame intervals:
```yaml
dataset:
  auto_interval: 30   # Collect every 30th frame automatically
  save_labels: true
```

---

## 🎓 Custom Model Training

After collecting a dataset:

```python
from src.trainer import Trainer
from src.config import Config

cfg = Config.from_yaml("config.yaml")
trainer = Trainer(cfg)
result = trainer.train("datasets/raw/dataset.yaml")
print("Best model:", trainer.best_model_path)
```

The best checkpoint is saved to `models/custom/best_custom.pt`.

Training parameters in `config.yaml`:
```yaml
training:
  epochs: 50
  batch: 16
  imgsz: 640
  device: ""        # "" = auto
  pretrained: true
```

Export to ONNX for deployment:
```python
trainer.export(format="onnx")
```

---

## 🔌 Useful Datasets (Hugging Face)

| Dataset | Use Case |
|---|---|
| `detection-datasets/coco` | General 80-class detection |
| `keremberke/coins-detection` | Coin detection |
| `keremberke/helmet-detection` | Safety helmet detection |
| `keremberke/license-plate-object-detection` | License plates |
| `Voxel51/fisheye8k` | Fisheye-lens cameras |
| `cppe-5` | Medical PPE detection |

---

## 🗺️ Roadmap

- [x] Webcam streaming
- [x] YOLO object detection
- [x] ByteTrack multi-object tracking
- [x] Analytics engine (current / cumulative / peak)
- [x] Line crossing counter (bidirectional)
- [x] Region polygon counter
- [x] Video recording (MP4/AVI)
- [x] Snapshot capture
- [x] Configuration system (YAML)
- [x] Dataset collection (YOLO-format)
- [x] YOLO fine-tuning wrapper
- [x] Instance segmentation
- [x] Pose estimation
- [ ] Face recognition
- [ ] Object speed estimation
- [ ] OCR integration
- [ ] QR / barcode reading
- [ ] Depth estimation (stereo / MiDaS)
- [ ] REST API server
- [ ] Flutter mobile client
- [ ] PyInstaller packaging as standalone executable

---

## 🛠️ Tech Stack

| Library | Version | Role |
|---|---|---|
| Python | ≥ 3.10 | Language |
| OpenCV | ≥ 4.9 | Frame I/O, drawing, VideoWriter |
| Ultralytics | ≥ 8.3 | YOLO11 detect / seg / pose |
| ByteTrack | built-in | Multi-object tracking |
| NumPy | ≥ 1.26 | Array operations, heatmap |
| PyTorch | ≥ 2.3 | Deep-learning backend |
| PyYAML | ≥ 6.0.1 | Configuration I/O |
| lapx | ≥ 0.5.9 | Hungarian algorithm for ByteTrack |

---

## 📝 License

This project is governed by a **proprietary licence**. See [LICENSE.md](LICENSE.md) for full terms.

**Summary:** All rights are reserved. You must obtain prior written consent from the owner before using, copying, modifying, or distributing this software. Any permitted use must cite the owner.

---

## 👤 Author

**Muhammad Izaan**  
Computer Vision & AI Engineer  

*If you use or reference this project in any work, you are required to credit the author as described in [LICENSE.md](LICENSE.md).*
