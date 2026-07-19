"""
VisionSense — Analytics Engine
=================================
Tracks per-class object counts across frames, maintains cumulative
session statistics, accumulates a centroid heatmap, and can export
a CSV session report.

Usage
-----
    from src.analytics import Analytics
    from src.config import Config

    cfg = Config.from_yaml("config.yaml")
    ana = Analytics(cfg, frame_size=(1280, 720))
    ana.update(results)
    summary = ana.get_summary()
    ana.save_report("outputs/session_report.csv")
"""

from __future__ import annotations

import csv
import logging
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from ultralytics.engine.results import Results

from src.config import Config

logger = logging.getLogger(__name__)


class Analytics:
    """Session-level detection analytics with heatmap accumulation.

    Parameters
    ----------
    config : Config
        Root application configuration.
    frame_size : (int, int)
        Frame resolution ``(width, height)`` used to initialise the heatmap.
    """

    def __init__(self, config: Config, frame_size: Tuple[int, int] = (1280, 720)) -> None:
        self._cfg = config.detection
        self._w, self._h = frame_size

        # Current-frame counts (reset each frame)
        self._current: Counter = Counter()

        # Cumulative totals per class over the whole session
        self._cumulative: Counter = Counter()

        # Peak count seen simultaneously per class
        self._peak: Dict[str, int] = defaultdict(int)

        # Total detections summed across all frames
        self._total_detections: int = 0

        # Total frames processed
        self._total_frames: int = 0

        # Heatmap: float32 accumulator, normalised for display
        self._heatmap: np.ndarray = np.zeros((self._h, self._w), dtype=np.float32)

        # Session start timestamp
        self._session_start: datetime = datetime.now()

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, results: List[Results]) -> None:
        """Process one frame's detection results.

        Parameters
        ----------
        results : list[Results]
            Ultralytics results list returned by the detector.
        """
        self._current.clear()
        self._total_frames += 1

        if not results or results[0].boxes is None:
            return

        boxes = results[0].boxes
        names: dict = results[0].names

        for box in boxes:
            conf = float(box.conf[0])
            if conf < self._cfg.confidence:
                continue

            cls_id = int(box.cls[0])
            cls_name: str = names.get(cls_id, str(cls_id))
            self._current[cls_name] += 1
            self._cumulative[cls_name] += 1
            self._total_detections += 1

            # Update peak
            if self._current[cls_name] > self._peak[cls_name]:
                self._peak[cls_name] = self._current[cls_name]

            # Accumulate heatmap centroid
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = int(np.clip((x1 + x2) / 2, 0, self._w - 1))
            cy = int(np.clip((y1 + y2) / 2, 0, self._h - 1))
            self._heatmap[cy, cx] += 1.0

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_counts(self) -> Dict[str, int]:
        """Return the current-frame per-class detection counts.

        Returns
        -------
        dict
            ``{class_name: count}`` for the last processed frame.
        """
        return dict(self._current)

    def get_summary(self) -> Dict[str, object]:
        """Return a full analytics summary dictionary.

        Returns
        -------
        dict
            Keys: ``current``, ``cumulative``, ``peak``,
            ``total_frames``, ``total_detections``, ``session_start``.
        """
        return {
            "current": dict(self._current),
            "cumulative": dict(self._cumulative),
            "peak": dict(self._peak),
            "total_frames": self._total_frames,
            "total_detections": self._total_detections,
            "session_start": self._session_start.isoformat(timespec="seconds"),
        }

    def total_objects(self) -> int:
        """Return the total count of objects in the current frame."""
        return sum(self._current.values())

    # ------------------------------------------------------------------
    # Heatmap
    # ------------------------------------------------------------------

    def get_heatmap(self) -> np.ndarray:
        """Return a normalised uint8 heatmap image ``(H, W, 3)`` in BGR.

        The heatmap is blurred for visual smoothness and coloured with the
        JET colour map.

        Returns
        -------
        np.ndarray
            BGR heatmap image.
        """
        import cv2

        hm = self._heatmap.copy()
        max_val = hm.max()
        if max_val > 0:
            hm /= max_val
        hm_u8 = (hm * 255).astype(np.uint8)
        # Gaussian blur for a smooth look
        hm_u8 = cv2.GaussianBlur(hm_u8, (51, 51), 0)
        hm_colored: np.ndarray = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
        return hm_colored

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save_report(self, path: Optional[str] = None) -> Path:
        """Export session analytics to a CSV file.

        Parameters
        ----------
        path : str, optional
            Output CSV path. Defaults to ``outputs/report_<timestamp>.csv``.

        Returns
        -------
        Path
            Path to the written CSV file.
        """
        if path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"outputs/report_{ts}.csv"

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        all_classes = sorted(
            set(self._cumulative.keys()) | set(self._peak.keys())
        )

        with open(out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["Class", "Current", "Cumulative", "Peak"])
            for cls in all_classes:
                writer.writerow([
                    cls,
                    self._current.get(cls, 0),
                    self._cumulative.get(cls, 0),
                    self._peak.get(cls, 0),
                ])
            writer.writerow([])
            writer.writerow(["Total Frames", self._total_frames])
            writer.writerow(["Total Detections", self._total_detections])
            writer.writerow(["Session Start", self._session_start.isoformat(timespec="seconds")])

        logger.info("Analytics report saved → %s", out)
        return out

    def reset(self) -> None:
        """Clear all counters and reset the session."""
        self._current.clear()
        self._cumulative.clear()
        self._peak.clear()
        self._total_detections = 0
        self._total_frames = 0
        self._heatmap[:] = 0.0
        self._session_start = datetime.now()
        logger.info("Analytics reset.")