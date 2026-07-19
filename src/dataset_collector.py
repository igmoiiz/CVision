"""
VisionSense — Dataset Collector
==================================
Captures annotated training frames from the live webcam feed for custom
YOLO fine-tuning. Supports both manual (keypress) and automatic (interval)
collection modes, and writes YOLO-format annotation ``.txt`` files alongside
each saved image.

On first collection it also writes a ``dataset.yaml`` file compatible with
the Ultralytics training pipeline.

Usage
-----
    from src.dataset_collector import DatasetCollector
    from src.config import Config

    cfg = Config.from_yaml("config.yaml")
    collector = DatasetCollector(cfg)

    # In the frame loop:
    collector.maybe_collect(frame, results, frame_index)  # auto mode
    collector.collect(frame, results)                     # manual trigger
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import yaml
from ultralytics.engine.results import Results

from src.config import Config

logger = logging.getLogger(__name__)


class DatasetCollector:
    """Collects training frames and YOLO-format annotations from the live feed.

    Parameters
    ----------
    config : Config
        Root application configuration. Dataset parameters come from
        ``config.dataset``.
    """

    def __init__(self, config: Config) -> None:
        self._cfg = config.dataset
        self._det_cfg = config.detection

        self._root = Path(self._cfg.output_dir)
        self._img_dir = self._root / "images"
        self._lbl_dir = self._root / "labels"
        self._img_dir.mkdir(parents=True, exist_ok=True)
        self._lbl_dir.mkdir(parents=True, exist_ok=True)

        self._count: int = 0
        self._class_names: dict = {}
        logger.info(
            "DatasetCollector initialised → images: %s  labels: %s",
            self._img_dir, self._lbl_dir,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect(
        self,
        frame: np.ndarray,
        results: Optional[List[Results]] = None,
    ) -> Path:
        """Save one frame (and optional YOLO annotation) to the dataset.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame to save.
        results : list[Results], optional
            Detection results used to generate annotation labels.

        Returns
        -------
        Path
            Path to the saved image.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        img_name = f"frame_{timestamp}.jpg"
        img_path = self._img_dir / img_name
        lbl_path = self._lbl_dir / img_name.replace(".jpg", ".txt")

        cv2.imwrite(str(img_path), frame)
        self._count += 1

        if self._cfg.save_labels and results and results[0].boxes is not None:
            self._write_labels(frame, results, lbl_path)

        logger.debug("Collected frame %d → %s", self._count, img_path)
        return img_path

    def maybe_collect(
        self,
        frame: np.ndarray,
        results: Optional[List[Results]],
        frame_index: int,
    ) -> bool:
        """Collect a frame automatically if the interval condition is met.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame.
        results : list[Results] or None
            Detection results for annotation.
        frame_index : int
            Running frame counter from the main loop.

        Returns
        -------
        bool
            ``True`` if a frame was collected this call.
        """
        interval = self._cfg.auto_interval
        if interval > 0 and frame_index % interval == 0:
            self.collect(frame, results)
            return True
        return False

    @property
    def count(self) -> int:
        """Number of frames collected this session."""
        return self._count

    def write_dataset_yaml(self, class_names: dict) -> Path:
        """Write a ``dataset.yaml`` compatible with Ultralytics training.

        Parameters
        ----------
        class_names : dict
            ``{class_id: class_name}`` mapping from the YOLO model.

        Returns
        -------
        Path
            Path to the written YAML file.
        """
        self._class_names = class_names
        names_list = [class_names[i] for i in sorted(class_names.keys())]

        data = {
            "path": str(self._root.resolve()),
            "train": "images",
            "val": "images",
            "nc": len(names_list),
            "names": names_list,
        }

        yaml_path = self._root / "dataset.yaml"
        with open(yaml_path, "w", encoding="utf-8") as fh:
            yaml.dump(data, fh, default_flow_style=False, sort_keys=False)

        logger.info("dataset.yaml written → %s", yaml_path)
        return yaml_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_labels(
        self,
        frame: np.ndarray,
        results: List[Results],
        lbl_path: Path,
    ) -> None:
        """Write YOLO-format annotation file for one frame.

        Parameters
        ----------
        frame : np.ndarray
            Source image (used for normalisation dimensions).
        results : list[Results]
            Detection results.
        lbl_path : Path
            Destination ``.txt`` annotation file.
        """
        h, w = frame.shape[:2]
        boxes = results[0].boxes

        lines: List[str] = []
        for box in boxes:
            conf = float(box.conf[0])
            if conf < self._det_cfg.confidence:
                continue

            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(float, box.xyxy[0])

            # Normalise to [0, 1]
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        with open(lbl_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
