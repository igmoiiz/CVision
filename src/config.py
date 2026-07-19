"""
VisionSense — Configuration System
====================================
Centralised, type-hinted configuration dataclass with YAML load/save support.
All runtime parameters live here so every module stays decoupled from hardcoded
values.

Usage
-----
    from src.config import Config
    cfg = Config.from_yaml("config.yaml")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import yaml  # PyYAML

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sub-configurations
# ---------------------------------------------------------------------------


@dataclass
class CameraConfig:
    """Webcam / video-source settings."""

    index: int = 0
    """Camera device index (0 = default webcam)."""

    width: int = 1280
    """Capture frame width in pixels."""

    height: int = 720
    """Capture frame height in pixels."""

    fps: int = 30
    """Target capture frame-rate."""

    flip_horizontal: bool = False
    """Mirror the frame left-right."""


@dataclass
class DetectionConfig:
    """YOLO inference settings."""

    model_path: str = "models/yolo11n.pt"
    """Path to YOLO weights file (.pt). Downloads automatically if absent."""

    confidence: float = 0.45
    """Minimum detection confidence threshold (0-1)."""

    iou: float = 0.50
    """Non-maximum suppression IoU threshold (0-1)."""

    device: str = ""
    """Inference device: '' = auto, 'cpu', '0', '0,1', 'cuda', 'mps'."""

    classes: Optional[List[int]] = None
    """Filter to specific COCO class IDs. None = all classes."""

    mode: str = "detect"
    """Active inference mode: 'detect' | 'segment' | 'pose'."""

    tracker: str = "bytetrack.yaml"
    """Tracker config file (ByteTrack or BoT-SORT)."""

    warmup_frames: int = 2
    """Number of dummy frames to run on model load for GPU warmup."""


@dataclass
class LineCounterConfig:
    """Virtual line crossing counter settings."""

    enabled: bool = True
    """Whether to run the line counter."""

    start: Tuple[int, int] = (0, 360)
    """Line start point (x, y)."""

    end: Tuple[int, int] = (1280, 360)
    """Line end point (x, y)."""

    color: Tuple[int, int, int] = (0, 255, 255)
    """Line colour in BGR."""

    thickness: int = 2
    """Line drawing thickness in pixels."""


@dataclass
class RegionCounterConfig:
    """Polygon region object counter settings."""

    enabled: bool = False
    """Whether to run the region counter."""

    polygon: List[Tuple[int, int]] = field(
        default_factory=lambda: [(200, 100), (1080, 100), (1080, 620), (200, 620)]
    )
    """Polygon vertices as list of (x, y) tuples."""

    color: Tuple[int, int, int] = (255, 128, 0)
    """Polygon colour in BGR."""


@dataclass
class RendererConfig:
    """HUD and visualisation settings."""

    show_hud: bool = True
    """Master toggle for the heads-up display overlay."""

    show_analytics: bool = True
    """Show per-class object count panel."""

    show_trails: bool = True
    """Draw centroid trail lines for tracked objects."""

    trail_length: int = 30
    """Maximum number of historic centroid points per track."""

    show_confidence: bool = True
    """Append confidence score to detection labels."""

    show_track_id: bool = True
    """Append tracker ID to detection labels."""

    box_thickness: int = 2
    """Bounding box line thickness."""

    font_scale: float = 0.55
    """Text size multiplier for labels."""

    text_thickness: int = 1
    """Label text stroke thickness."""

    mask_alpha: float = 0.35
    """Segmentation mask overlay opacity (0-1)."""

    hud_alpha: float = 0.55
    """HUD panel background opacity (0-1)."""

    show_heatmap: bool = False
    """Overlay accumulated centroid heatmap on the frame."""


@dataclass
class RecordingConfig:
    """Video recording settings."""

    output_dir: str = "outputs/recordings"
    """Directory where recordings are saved."""

    codec: str = "mp4v"
    """FourCC codec string: 'mp4v', 'XVID', 'H264', 'avc1'."""

    extension: str = ".mp4"
    """Output file extension."""


@dataclass
class SnapshotConfig:
    """Screenshot settings."""

    output_dir: str = "outputs/screenshots"
    """Directory where snapshots are saved."""

    extension: str = ".png"
    """Image file format."""


@dataclass
class DatasetConfig:
    """Dataset collection settings."""

    output_dir: str = "datasets/raw"
    """Root directory for collected frames and labels."""

    auto_interval: int = 0
    """Auto-collect every N frames (0 = manual keypress only)."""

    save_labels: bool = True
    """Write YOLO-format .txt annotations alongside saved images."""

    dataset_name: str = "custom_dataset"
    """Name used in dataset.yaml."""


@dataclass
class TrainingConfig:
    """YOLO fine-tuning settings."""

    epochs: int = 50
    """Number of training epochs."""

    batch: int = 16
    """Batch size (-1 = auto)."""

    imgsz: int = 640
    """Input image size for training."""

    device: str = ""
    """Training device (same format as DetectionConfig.device)."""

    project: str = "models/custom"
    """Directory to save training runs."""

    name: str = "train"
    """Experiment name."""

    pretrained: bool = True
    """Start from pretrained COCO weights."""


# ---------------------------------------------------------------------------
# Root Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """Root VisionSense configuration object."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    line_counter: LineCounterConfig = field(default_factory=LineCounterConfig)
    region_counter: RegionCounterConfig = field(default_factory=RegionCounterConfig)
    renderer: RendererConfig = field(default_factory=RendererConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    snapshot: SnapshotConfig = field(default_factory=SnapshotConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Global flags
    window_name: str = "VisionSense"
    log_level: str = "INFO"
    log_file: str = "outputs/visionsense.log"

    # ---------------------------------------------------------------------------
    # Serialisation helpers
    # ---------------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path = "config.yaml") -> "Config":
        """Load config from a YAML file, falling back to defaults for missing keys.

        Parameters
        ----------
        path : str | Path
            Path to the YAML configuration file.

        Returns
        -------
        Config
            Populated configuration object.
        """
        p = Path(path)
        if not p.exists():
            logger.warning("Config file '%s' not found — using defaults.", p)
            return cls()

        with open(p, "r", encoding="utf-8") as fh:
            raw: dict = yaml.safe_load(fh) or {}

        cfg = cls()

        def _apply(obj, data: dict) -> None:
            """Recursively overwrite dataclass fields from a dict."""
            for key, value in data.items():
                if not hasattr(obj, key):
                    logger.warning("Unknown config key '%s' — skipping.", key)
                    continue
                attr = getattr(obj, key)
                if hasattr(attr, "__dataclass_fields__") and isinstance(value, dict):
                    _apply(attr, value)
                else:
                    # Handle tuple fields stored as lists in YAML
                    field_type = type(attr)
                    if field_type is tuple and isinstance(value, list):
                        value = tuple(map(tuple, value)) if value and isinstance(value[0], list) else tuple(value)
                    setattr(obj, key, value)

        _apply(cfg, raw)
        logger.info("Configuration loaded from '%s'.", p)
        return cfg

    def to_yaml(self, path: str | Path = "config.yaml") -> None:
        """Persist the current configuration to a YAML file.

        Parameters
        ----------
        path : str | Path
            Destination file path.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            yaml.dump(asdict(self), fh, default_flow_style=False, sort_keys=False)
        logger.info("Configuration saved to '%s'.", p)

    def ensure_directories(self) -> None:
        """Create all output/dataset directories defined in config."""
        dirs = [
            self.recording.output_dir,
            self.snapshot.output_dir,
            self.dataset.output_dir,
            Path(self.log_file).parent,
            "models",
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)