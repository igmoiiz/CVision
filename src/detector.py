"""
VisionSense — Object Detector
================================
Wraps Ultralytics YOLO to support three inference modes:
  * **detect** — bounding-box object detection with ByteTrack tracking
  * **segment** — instance segmentation (masks + boxes)
  * **pose** — human keypoint / skeleton estimation

The model is loaded once at construction; ``warm_up()`` should be called
before the first live frame to pre-allocate GPU memory.

Usage
-----
    from src.config import Config
    from src.detector import ObjectDetector

    cfg = Config.from_yaml("config.yaml")
    det = ObjectDetector(cfg)
    det.warm_up()

    results = det.infer(frame)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from src.config import Config

logger = logging.getLogger(__name__)


class ObjectDetector:
    """YOLO-based object detector / segmentor / pose estimator.

    Parameters
    ----------
    config : Config
        Root application configuration. Detection parameters are read from
        ``config.detection``.
    """

    #: Supported inference modes
    MODES: List[str] = ["detect", "segment", "pose"]

    def __init__(self, config: Config) -> None:
        self._cfg = config.detection
        self._mode: str = self._cfg.mode

        logger.info(
            "Loading YOLO model from '%s' (mode=%s, device='%s') …",
            self._cfg.model_path, self._mode, self._cfg.device or "auto",
        )

        model_path = Path(self._cfg.model_path)

        # Auto-select a mode-appropriate model if user left a generic name
        self._model: YOLO = YOLO(str(model_path))
        logger.info("Model loaded successfully.")

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        """Current inference mode: ``'detect'`` | ``'segment'`` | ``'pose'``."""
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        if value not in self.MODES:
            raise ValueError(f"Unknown mode '{value}'. Choose from {self.MODES}.")
        self._mode = value
        logger.info("Inference mode switched to '%s'.", value)

    def cycle_mode(self) -> str:
        """Rotate through the available modes and return the new mode name.

        Returns
        -------
        str
            The newly active mode string.
        """
        idx = self.MODES.index(self._mode)
        self._mode = self.MODES[(idx + 1) % len(self.MODES)]
        logger.info("Mode cycled → '%s'.", self._mode)
        return self._mode

    # ------------------------------------------------------------------
    # Warm-up
    # ------------------------------------------------------------------

    def warm_up(self) -> None:
        """Run a few dummy inferences to initialise GPU / CUDA kernels.

        This prevents the first live frame from incurring a large latency spike.
        """
        h, w = 640, 640
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        for _ in range(self._cfg.warmup_frames):
            self._run(dummy, tracker=False)
        logger.info("Model warm-up complete (%d frames).", self._cfg.warmup_frames)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(self, frame: np.ndarray) -> List[Results]:
        """Run inference on *frame* using the active mode with ByteTrack tracking.

        Parameters
        ----------
        frame : np.ndarray
            BGR image array ``(H, W, 3)``.

        Returns
        -------
        list[Results]
            Ultralytics ``Results`` list (one entry per image).
        """
        return self._run(frame, tracker=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run(self, frame: np.ndarray, tracker: bool = True) -> List[Results]:
        """Internal inference dispatch shared by ``infer`` and ``warm_up``.

        Parameters
        ----------
        frame : np.ndarray
            Input BGR frame.
        tracker : bool
            Whether to enable the ByteTrack tracker (``persist=True``).

        Returns
        -------
        list[Results]
            Raw Ultralytics results.
        """
        common_kwargs = dict(
            source=frame,
            conf=self._cfg.confidence,
            iou=self._cfg.iou,
            verbose=False,
            classes=self._cfg.classes or None,
            device=self._cfg.device or None,
        )

        if tracker:
            return self._model.track(
                **common_kwargs,
                tracker=self._cfg.tracker,
                persist=True,
            )

        # Warm-up: plain predict without tracker
        task_map = {
            "detect": self._model.predict,
            "segment": self._model.predict,
            "pose": self._model.predict,
        }
        predict_fn = task_map.get(self._mode, self._model.predict)
        return predict_fn(**common_kwargs)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def class_names(self) -> dict:
        """Return the model's ``{id: name}`` class dictionary."""
        return self._model.names