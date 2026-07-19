"""
VisionSense — Camera Module
=============================
Thread-safe webcam capture with configurable resolution, FPS, horizontal flip,
recording, and snapshot capabilities.

The ``Camera`` class wraps ``cv2.VideoCapture`` and exposes a clean, typed
interface used by the main application loop.

Usage
-----
    from src.config import Config
    from src.camera import Camera

    cfg = Config.from_yaml("config.yaml")
    cam = Camera(cfg)
    success, frame = cam.read()
    cam.release()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from src.config import Config

logger = logging.getLogger(__name__)


class Camera:
    """Webcam capture and frame management.

    Parameters
    ----------
    config : Config
        Root application configuration object.
    """

    def __init__(self, config: Config) -> None:
        self._cfg = config.camera
        self._cap: cv2.VideoCapture = self._open()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _open(self) -> cv2.VideoCapture:
        """Open the video capture device with the configured settings.

        Returns
        -------
        cv2.VideoCapture
            Opened capture object.

        Raises
        ------
        RuntimeError
            If the camera cannot be opened.
        """
        cap = cv2.VideoCapture(self._cfg.index)

        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {self._cfg.index}. "
                "Check that the device is connected and not in use."
            )

        # Apply capture settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self._cfg.fps)

        # Read back actual values (may differ from requested)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)

        logger.info(
            "Camera #%d opened — %dx%d @ %.1f FPS.",
            self._cfg.index, actual_w, actual_h, actual_fps,
        )
        return cap

    # ------------------------------------------------------------------
    # Frame capture
    # ------------------------------------------------------------------

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Capture a single frame from the camera.

        Returns
        -------
        (success, frame)
            ``success`` is ``False`` if the frame could not be read.
            ``frame`` is an ``(H, W, 3)`` BGR NumPy array, or ``None`` on failure.
        """
        ret, frame = self._cap.read()
        if not ret or frame is None:
            logger.warning("Frame read failed (ret=%s).", ret)
            return False, None

        if self._cfg.flip_horizontal:
            frame = cv2.flip(frame, 1)

        return True, frame

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def frame_size(self) -> Tuple[int, int]:
        """Return the actual capture resolution as ``(width, height)``."""
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    @property
    def fps(self) -> float:
        """Return the camera's reported frame rate."""
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def is_opened(self) -> bool:
        """Return ``True`` if the capture device is currently open."""
        return self._cap.isOpened()

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def release(self) -> None:
        """Release the camera and close all OpenCV windows."""
        if self._cap.isOpened():
            self._cap.release()
            logger.info("Camera #%d released.", self._cfg.index)
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *_) -> None:
        self.release()