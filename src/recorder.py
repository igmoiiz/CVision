"""
VisionSense — Video Recorder
=============================
Wraps ``cv2.VideoWriter`` to produce timestamped MP4/AVI recordings in the
configured output directory.

Usage
-----
    from src.recorder import Recorder
    from src.config import Config

    cfg = Config.from_yaml("config.yaml")
    rec = Recorder(cfg, frame_size=(1280, 720), fps=30.0)
    rec.start()
    rec.write(frame)
    path = rec.stop()
    print("Saved to", path)
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from src.config import Config

logger = logging.getLogger(__name__)


class Recorder:
    """Timestamped video recorder backed by ``cv2.VideoWriter``.

    Parameters
    ----------
    config : Config
        Root application configuration.
    frame_size : (int, int)
        Frame resolution as ``(width, height)``.
    fps : float
        Playback frame rate for the output video.
    """

    def __init__(
        self,
        config: Config,
        frame_size: Tuple[int, int],
        fps: float = 30.0,
    ) -> None:
        self._cfg = config.recording
        self._frame_size = frame_size
        self._fps = fps
        self._writer: Optional[cv2.VideoWriter] = None
        self._output_path: Optional[Path] = None
        self._is_recording: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        """``True`` if recording is currently active."""
        return self._is_recording

    @property
    def output_path(self) -> Optional[Path]:
        """Path to the current (or most recently completed) output file."""
        return self._output_path

    def start(self) -> Path:
        """Begin a new recording session.

        A new output file is created with a timestamp in its name.

        Returns
        -------
        Path
            Path to the file being written.

        Raises
        ------
        RuntimeError
            If ``cv2.VideoWriter`` could not be initialised.
        """
        if self._is_recording:
            logger.warning("Recording already active — call stop() first.")
            return self._output_path  # type: ignore[return-value]

        out_dir = Path(self._cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"VisionSense_{timestamp}{self._cfg.extension}"
        self._output_path = out_dir / filename

        fourcc = cv2.VideoWriter_fourcc(*self._cfg.codec)
        self._writer = cv2.VideoWriter(
            str(self._output_path),
            fourcc,
            self._fps,
            self._frame_size,
        )

        if not self._writer.isOpened():
            self._writer = None
            raise RuntimeError(
                f"VideoWriter failed to open '{self._output_path}'. "
                f"Try a different codec (current: '{self._cfg.codec}')."
            )

        self._is_recording = True
        logger.info("Recording started → %s", self._output_path)
        return self._output_path

    def write(self, frame: np.ndarray) -> None:
        """Write one annotated frame to the current recording.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame to record.
        """
        if not self._is_recording or self._writer is None:
            return
        self._writer.write(frame)

    def stop(self) -> Optional[Path]:
        """Finalise and close the current recording.

        Returns
        -------
        Path or None
            Path to the completed file, or ``None`` if nothing was being recorded.
        """
        if not self._is_recording or self._writer is None:
            logger.warning("No active recording to stop.")
            return None

        self._writer.release()
        self._writer = None
        self._is_recording = False
        logger.info("Recording stopped → %s", self._output_path)
        return self._output_path
