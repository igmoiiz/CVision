"""
VisionSense — FPS Counter & Inference Timer
=============================================
Provides a smoothed, rolling-window FPS counter and a context-manager
``InferenceTimer`` for measuring per-frame model inference latency.

Usage
-----
    fps = FPSCounter(window=30)
    timer = InferenceTimer()

    with timer:
        results = model(frame)

    current_fps = fps.tick()
    stats = fps.stats()
"""

from __future__ import annotations

import time
from collections import deque
from types import TracebackType
from typing import Deque, Dict, Optional, Type


class FPSCounter:
    """Rolling-window frames-per-second counter.

    Parameters
    ----------
    window : int
        Number of recent frame intervals to average over.
    """

    def __init__(self, window: int = 30) -> None:
        self._window: int = max(1, window)
        self._timestamps: Deque[float] = deque(maxlen=self._window)
        self._session_start: float = time.perf_counter()
        self._total_frames: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tick(self) -> float:
        """Record one frame and return the current smoothed FPS.

        Returns
        -------
        float
            Exponentially smoothed FPS based on the rolling window.
        """
        now = time.perf_counter()
        self._timestamps.append(now)
        self._total_frames += 1
        return self._compute_fps()

    def fps(self) -> float:
        """Return the current FPS without recording a new tick.

        Returns
        -------
        float
            Current smoothed FPS.
        """
        return self._compute_fps()

    def stats(self) -> Dict[str, float]:
        """Return a summary of session FPS statistics.

        Returns
        -------
        dict
            Keys: ``current``, ``session_avg``, ``total_frames``, ``elapsed_s``.
        """
        elapsed = time.perf_counter() - self._session_start
        session_avg = self._total_frames / elapsed if elapsed > 0 else 0.0
        return {
            "current": round(self._compute_fps(), 1),
            "session_avg": round(session_avg, 1),
            "total_frames": self._total_frames,
            "elapsed_s": round(elapsed, 1),
        }

    def reset(self) -> None:
        """Clear the rolling window and reset the session timer."""
        self._timestamps.clear()
        self._session_start = time.perf_counter()
        self._total_frames = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        delta = self._timestamps[-1] - self._timestamps[0]
        if delta <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / delta


# ---------------------------------------------------------------------------


class InferenceTimer:
    """Context-manager that measures model inference wall-clock time.

    Keeps a rolling history of recent measurements and exposes aggregate
    statistics.

    Parameters
    ----------
    history : int
        Number of recent measurements to retain.

    Examples
    --------
    ::

        timer = InferenceTimer()
        with timer:
            results = model(frame)
        print(timer.last_ms)       # ms for last inference
        print(timer.avg_ms())      # rolling average ms
    """

    def __init__(self, history: int = 60) -> None:
        self._history: Deque[float] = deque(maxlen=max(1, history))
        self._start: float = 0.0
        self._last_ms: float = 0.0

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "InferenceTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        elapsed_ms = (time.perf_counter() - self._start) * 1_000
        self._last_ms = elapsed_ms
        self._history.append(elapsed_ms)

    # ------------------------------------------------------------------
    # Properties & methods
    # ------------------------------------------------------------------

    @property
    def last_ms(self) -> float:
        """Inference time of the last measured call in milliseconds."""
        return round(self._last_ms, 2)

    def avg_ms(self) -> float:
        """Average inference time over the rolling history in milliseconds."""
        if not self._history:
            return 0.0
        return round(sum(self._history) / len(self._history), 2)

    def min_ms(self) -> float:
        """Minimum inference time over the rolling history in milliseconds."""
        return round(min(self._history, default=0.0), 2)

    def max_ms(self) -> float:
        """Maximum inference time over the rolling history in milliseconds."""
        return round(max(self._history, default=0.0), 2)

    def stats(self) -> Dict[str, float]:
        """Return a summary dict of inference timing statistics.

        Returns
        -------
        dict
            Keys: ``last_ms``, ``avg_ms``, ``min_ms``, ``max_ms``.
        """
        return {
            "last_ms": self.last_ms,
            "avg_ms": self.avg_ms(),
            "min_ms": self.min_ms(),
            "max_ms": self.max_ms(),
        }