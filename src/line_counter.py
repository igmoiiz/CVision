"""
VisionSense — Line Counter & Region Counter
============================================
Provides two counting primitives:

* **LineCounter** — counts objects crossing a configurable virtual line.
  Supports bidirectional counting (IN / OUT) based on the direction of travel.

* **RegionCounter** — counts objects whose centroid lies inside a configurable
  convex or concave polygon region.

Both write their overlays directly onto frames via their ``draw()`` method.

Usage
-----
    from src.line_counter import LineCounter, RegionCounter
    from src.config import Config

    cfg = Config.from_yaml("config.yaml")
    lc = LineCounter(cfg)
    rc = RegionCounter(cfg)

    lc.update(results)
    rc.update(results)

    frame = lc.draw(frame)
    frame = rc.draw(frame)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np
from ultralytics.engine.results import Results

from src.config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _centroid(x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int]:
    """Return the integer centroid of a bounding box."""
    return (x1 + x2) // 2, (y1 + y2) // 2


def _side_of_line(
    px: int, py: int,
    ax: int, ay: int,
    bx: int, by: int,
) -> float:
    """Signed cross-product to determine which side of AB the point P lies on.

    Positive → left side; Negative → right side; Zero → on the line.
    """
    return float((bx - ax) * (py - ay) - (by - ay) * (px - ax))


# ---------------------------------------------------------------------------
# Line Counter
# ---------------------------------------------------------------------------


class LineCounter:
    """Virtual-line crossing counter with bidirectional IN/OUT tracking.

    Parameters
    ----------
    config : Config
        Root application configuration. Line geometry is taken from
        ``config.line_counter``.
    """

    def __init__(self, config: Config) -> None:
        self._cfg = config.line_counter
        self._ax, self._ay = self._cfg.start
        self._bx, self._by = self._cfg.end

        # Track the last-seen side per object ID
        self._prev_side: Dict[int, float] = {}

        # IDs already counted (avoid double-counting per crossing)
        self._counted_in: Set[int] = set()
        self._counted_out: Set[int] = set()

        self.in_count: int = 0
        self.out_count: int = 0

        # Per-class crossing breakdown
        self.class_in: Dict[str, int] = defaultdict(int)
        self.class_out: Dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, results: List[Results]) -> None:
        """Process detection results for one frame.

        Parameters
        ----------
        results : list[Results]
            Ultralytics results list.
        """
        if not results or results[0].boxes is None:
            return

        boxes = results[0].boxes
        names: dict = results[0].names

        for box in boxes:
            if box.id is None:
                continue

            obj_id = int(box.id[0])
            cls_id = int(box.cls[0])
            cls_name: str = names.get(cls_id, str(cls_id))

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = _centroid(x1, y1, x2, y2)

            side = _side_of_line(cx, cy, self._ax, self._ay, self._bx, self._by)

            if obj_id in self._prev_side:
                prev = self._prev_side[obj_id]

                if prev > 0 and side <= 0 and obj_id not in self._counted_out:
                    # Crossed from left → right (OUT)
                    self.out_count += 1
                    self.class_out[cls_name] += 1
                    self._counted_out.add(obj_id)
                    self._counted_in.discard(obj_id)  # allow re-counting on reverse

                elif prev < 0 and side >= 0 and obj_id not in self._counted_in:
                    # Crossed from right → left (IN)
                    self.in_count += 1
                    self.class_in[cls_name] += 1
                    self._counted_in.add(obj_id)
                    self._counted_out.discard(obj_id)

            self._prev_side[obj_id] = side

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Render the counting line and IN/OUT counters onto *frame*.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame to draw on (in-place).

        Returns
        -------
        np.ndarray
            Annotated frame.
        """
        color = tuple(self._cfg.color)
        thickness = self._cfg.thickness

        # Draw main line
        cv2.line(
            frame,
            (self._ax, self._ay),
            (self._bx, self._by),
            color,
            thickness,
        )

        # Midpoint label background + text
        mid_x = (self._ax + self._bx) // 2
        mid_y = (self._ay + self._by) // 2

        label = f"IN:{self.in_count}  OUT:{self.out_count}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(
            frame,
            (mid_x - 4, mid_y - th - baseline - 4),
            (mid_x + tw + 4, mid_y + baseline),
            (0, 0, 0),
            cv2.FILLED,
        )
        cv2.putText(
            frame, label,
            (mid_x, mid_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            color, 2, cv2.LINE_AA,
        )

        return frame

    def reset(self) -> None:
        """Reset all counters and tracking state."""
        self._prev_side.clear()
        self._counted_in.clear()
        self._counted_out.clear()
        self.in_count = 0
        self.out_count = 0
        self.class_in.clear()
        self.class_out.clear()


# ---------------------------------------------------------------------------
# Region Counter
# ---------------------------------------------------------------------------


class RegionCounter:
    """Polygon region occupancy counter.

    Objects whose centroid falls inside the configured polygon are counted as
    *inside* the region. The count updates every frame.

    Parameters
    ----------
    config : Config
        Root application configuration. Polygon vertices are read from
        ``config.region_counter``.
    """

    def __init__(self, config: Config) -> None:
        self._cfg = config.region_counter

        # Convert to numpy contour array
        self._polygon: np.ndarray = np.array(
            self._cfg.polygon, dtype=np.int32
        ).reshape((-1, 1, 2))

        # Per-class counts inside region (current frame only)
        self.class_counts: Dict[str, int] = defaultdict(int)
        self.inside_count: int = 0

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, results: List[Results]) -> None:
        """Compute region occupancy for one frame.

        Parameters
        ----------
        results : list[Results]
            Ultralytics results list.
        """
        self.class_counts.clear()
        self.inside_count = 0

        if not results or results[0].boxes is None:
            return

        boxes = results[0].boxes
        names: dict = results[0].names

        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name: str = names.get(cls_id, str(cls_id))

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = _centroid(x1, y1, x2, y2)

            dist = cv2.pointPolygonTest(self._polygon, (float(cx), float(cy)), False)
            if dist >= 0:  # inside or on the boundary
                self.inside_count += 1
                self.class_counts[cls_name] += 1

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Render the region polygon and occupancy count onto *frame*.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame to draw on.

        Returns
        -------
        np.ndarray
            Annotated frame.
        """
        color = tuple(self._cfg.color)

        # Semi-transparent fill
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self._polygon], color)
        frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)

        # Polygon border
        cv2.polylines(frame, [self._polygon], isClosed=True, color=color, thickness=2)

        # Label at top-left vertex
        anchor = tuple(self._polygon[0][0])
        label = f"Region: {self.inside_count}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(
            frame,
            (anchor[0] - 2, anchor[1] - th - baseline - 4),
            (anchor[0] + tw + 2, anchor[1] + baseline),
            (0, 0, 0),
            cv2.FILLED,
        )
        cv2.putText(
            frame, label,
            anchor,
            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            color, 2, cv2.LINE_AA,
        )

        return frame

    def reset(self) -> None:
        """Clear region counts."""
        self.class_counts.clear()
        self.inside_count = 0