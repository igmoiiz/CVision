"""
VisionSense — Snapshot Module
===============================
Captures and saves full-resolution annotated frames to disk with a timestamp.

Usage
-----
    from src.snapshot import Snapshot
    from src.config import Config

    cfg = Config.from_yaml("config.yaml")
    snap = Snapshot(cfg)
    path = snap.save(frame)
    print("Snapshot saved to", path)
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.config import Config

logger = logging.getLogger(__name__)


class Snapshot:
    """Single-frame screenshot saver.

    Parameters
    ----------
    config : Config
        Root application configuration.
    """

    def __init__(self, config: Config) -> None:
        self._cfg = config.snapshot

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, frame: np.ndarray, tag: str = "") -> Path:
        """Save *frame* as a PNG/JPEG to the configured screenshot directory.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame (annotated or raw) to persist.
        tag : str, optional
            Optional short string appended to the filename for identification.

        Returns
        -------
        Path
            Absolute path to the saved image file.

        Raises
        ------
        RuntimeError
            If ``cv2.imwrite`` fails to write the file.
        """
        out_dir = Path(self._cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # ms precision
        suffix = f"_{tag}" if tag else ""
        filename = f"snap_{timestamp}{suffix}{self._cfg.extension}"
        out_path = out_dir / filename

        success = cv2.imwrite(str(out_path), frame)
        if not success:
            raise RuntimeError(f"cv2.imwrite failed for path '{out_path}'.")

        logger.info("Snapshot saved → %s", out_path)
        return out_path
