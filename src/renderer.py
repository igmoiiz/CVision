"""
VisionSense — Renderer
========================
Draws all visual overlays onto frames:

  * Bounding boxes (detect mode)
  * Instance segmentation masks (segment mode)
  * Pose keypoints and skeleton connections (pose mode)
  * Centroid track trails
  * Semi-transparent HUD panel (FPS, inference time, analytics)
  * Heatmap overlay

The colour palette is auto-generated from a HSV wheel so every class
gets a visually distinct colour without manual configuration.

Usage
-----
    from src.renderer import Renderer
    from src.config import Config

    cfg = Config.from_yaml("config.yaml")
    rend = Renderer(cfg)
    frame = rend.draw(frame, results, analytics, fps, inf_ms, mode)
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics.engine.results import Results

from src.config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# COCO Skeleton pairs for pose mode (keypoint index pairs)
# ---------------------------------------------------------------------------

_SKELETON_PAIRS: List[Tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # head
    (5, 6),                                    # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),          # arms
    (5, 11), (6, 12), (11, 12),               # torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
]


class Renderer:
    """All-mode visual overlay renderer for VisionSense.

    Parameters
    ----------
    config : Config
        Root application configuration.
    """

    def __init__(self, config: Config) -> None:
        self._cfg = config.renderer
        # class_id → BGR colour
        self._palette: Dict[int, Tuple[int, int, int]] = {}
        # track_id → deque of recent centroids
        self._trails: Dict[int, Deque[Tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=self._cfg.trail_length)
        )

    # ------------------------------------------------------------------
    # Colour palette
    # ------------------------------------------------------------------

    def _get_color(self, class_id: int) -> Tuple[int, int, int]:
        """Return a stable, visually distinct BGR colour for *class_id*."""
        if class_id not in self._palette:
            hue = int((class_id * 37) % 180)
            hsv = np.uint8([[[hue, 220, 220]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
            self._palette[class_id] = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
        return self._palette[class_id]

    # ------------------------------------------------------------------
    # Master draw entry point
    # ------------------------------------------------------------------

    def draw(
        self,
        frame: np.ndarray,
        results: List[Results],
        analytics_summary: Dict[str, Any],
        fps: float,
        inf_ms: float,
        mode: str = "detect",
        heatmap: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Render all overlays onto *frame* and return the annotated copy.

        Parameters
        ----------
        frame : np.ndarray
            Input BGR frame.
        results : list[Results]
            Ultralytics inference results.
        analytics_summary : dict
            Dict returned by ``Analytics.get_summary()``.
        fps : float
            Current smoothed FPS value.
        inf_ms : float
            Inference time in milliseconds.
        mode : str
            Active inference mode (``'detect'`` | ``'segment'`` | ``'pose'``).
        heatmap : np.ndarray, optional
            Pre-computed BGR heatmap to overlay.

        Returns
        -------
        np.ndarray
            Fully annotated BGR frame.
        """
        if not results:
            return frame

        # Heatmap overlay (drawn first, under detections)
        if self._cfg.show_heatmap and heatmap is not None:
            frame = cv2.addWeighted(heatmap, 0.4, frame, 0.6, 0)

        if mode == "segment":
            frame = self._draw_segments(frame, results)
        elif mode == "pose":
            frame = self._draw_pose(frame, results)
        else:
            frame = self._draw_boxes(frame, results)

        if self._cfg.show_hud:
            frame = self._draw_hud(frame, analytics_summary, fps, inf_ms, mode)

        return frame

    # ------------------------------------------------------------------
    # Detect mode
    # ------------------------------------------------------------------

    def _draw_boxes(self, frame: np.ndarray, results: List[Results]) -> np.ndarray:
        """Draw bounding boxes and track trails (detect mode)."""
        boxes = results[0].boxes
        if boxes is None:
            return frame

        names: dict = results[0].names

        for box in boxes:
            conf = float(box.conf[0])
            if conf < 0:
                continue

            cls_id = int(box.cls[0])
            cls_name: str = names.get(cls_id, str(cls_id))
            color = self._get_color(cls_id)

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self._cfg.box_thickness)

            # Build label
            parts: List[str] = [cls_name.replace("_", " ").title()]
            if self._cfg.show_track_id and box.id is not None:
                parts.append(f"#{int(box.id[0])}")
            if self._cfg.show_confidence:
                parts.append(f"{conf:.2f}")
            label = " ".join(parts)

            self._put_label(frame, label, x1, y1, color)

            # Track trail
            if self._cfg.show_trails and box.id is not None:
                tid = int(box.id[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                self._trails[tid].append((cx, cy))
                self._draw_trail(frame, tid, color)

        return frame

    # ------------------------------------------------------------------
    # Segment mode
    # ------------------------------------------------------------------

    def _draw_segments(self, frame: np.ndarray, results: List[Results]) -> np.ndarray:
        """Draw segmentation masks and bounding boxes."""
        r = results[0]

        # Draw masks first (underneath boxes)
        if r.masks is not None:
            h, w = frame.shape[:2]
            for i, mask_data in enumerate(r.masks.data):
                cls_id = int(r.boxes.cls[i]) if r.boxes is not None else 0
                color = self._get_color(cls_id)

                mask_np = mask_data.cpu().numpy().astype(np.uint8)
                mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
                colored = np.zeros_like(frame, dtype=np.uint8)
                colored[mask_resized == 1] = color
                frame = cv2.addWeighted(colored, self._cfg.mask_alpha, frame, 1.0, 0)

        # Draw boxes on top
        frame = self._draw_boxes(frame, results)
        return frame

    # ------------------------------------------------------------------
    # Pose mode
    # ------------------------------------------------------------------

    def _draw_pose(self, frame: np.ndarray, results: List[Results]) -> np.ndarray:
        """Draw keypoints and skeleton connections."""
        r = results[0]

        if r.keypoints is None:
            return frame

        kp_data = r.keypoints.xy.cpu().numpy()  # shape (N, 17, 2)
        kp_conf = r.keypoints.conf  # may be None

        for person_idx, keypoints in enumerate(kp_data):
            person_color = self._get_color(person_idx % 80)

            # Skeleton connections
            for a_idx, b_idx in _SKELETON_PAIRS:
                if a_idx >= len(keypoints) or b_idx >= len(keypoints):
                    continue
                ax, ay = int(keypoints[a_idx][0]), int(keypoints[a_idx][1])
                bx, by = int(keypoints[b_idx][0]), int(keypoints[b_idx][1])
                if ax == 0 and ay == 0:
                    continue
                if bx == 0 and by == 0:
                    continue
                cv2.line(frame, (ax, ay), (bx, by), person_color, 2, cv2.LINE_AA)

            # Keypoints
            for kp in keypoints:
                kx, ky = int(kp[0]), int(kp[1])
                if kx == 0 and ky == 0:
                    continue
                cv2.circle(frame, (kx, ky), 4, (255, 255, 255), -1)
                cv2.circle(frame, (kx, ky), 4, person_color, 1)

        # Bounding boxes
        if r.boxes is not None:
            frame = self._draw_boxes(frame, results)

        return frame

    # ------------------------------------------------------------------
    # Trail helper
    # ------------------------------------------------------------------

    def _draw_trail(
        self,
        frame: np.ndarray,
        track_id: int,
        color: Tuple[int, int, int],
    ) -> None:
        pts = list(self._trails[track_id])
        if len(pts) < 2:
            return
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            faded = tuple(int(c * alpha) for c in color)
            cv2.line(frame, pts[i - 1], pts[i], faded, 2, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Label helper
    # ------------------------------------------------------------------

    def _put_label(
        self,
        frame: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: Tuple[int, int, int],
    ) -> None:
        """Draw a label with a dark filled background for legibility."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = self._cfg.font_scale
        thickness = self._cfg.text_thickness

        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        ty = max(y - 6, th + baseline + 4)

        # Background rect
        cv2.rectangle(
            frame,
            (x, ty - th - baseline - 4),
            (x + tw + 4, ty + baseline),
            (0, 0, 0),
            cv2.FILLED,
        )
        # Text
        cv2.putText(
            frame, text,
            (x + 2, ty - 2),
            font, scale, color, thickness, cv2.LINE_AA,
        )

    # ------------------------------------------------------------------
    # HUD panel
    # ------------------------------------------------------------------

    def _draw_hud(
        self,
        frame: np.ndarray,
        summary: Dict[str, Any],
        fps: float,
        inf_ms: float,
        mode: str,
    ) -> np.ndarray:
        """Draw the semi-transparent HUD panel in the top-left corner."""
        current_counts: Dict[str, int] = summary.get("current", {})
        total: int = sum(current_counts.values())

        lines = [
            f"  VisionSense",
            f"  Mode    : {mode.upper()}",
            f"  FPS     : {fps:.1f}",
            f"  Infer   : {inf_ms:.1f} ms",
            f"  Objects : {total}",
        ]
        if current_counts:
            lines.append("  ─────────────────")
            for cls, cnt in sorted(current_counts.items()):
                lines.append(f"  {cls.title():<16}: {cnt}")

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.52
        lh = 22        # line height px
        pad = 10

        panel_w = 230
        panel_h = pad * 2 + len(lines) * lh + 4

        # Dark semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (10, 10, 10), cv2.FILLED)
        frame = cv2.addWeighted(overlay, self._cfg.hud_alpha, frame, 1 - self._cfg.hud_alpha, 0)

        # Accent border
        cv2.rectangle(frame, (8, 8), (8 + panel_w, 8 + panel_h), (0, 200, 160), 1)

        # Render each line
        for i, line in enumerate(lines):
            y = 8 + pad + (i + 1) * lh
            if i == 0:
                # Title — brighter
                color = (0, 220, 180)
                s = 0.60
            elif "─" in line:
                color = (80, 80, 80)
                s = 0.45
            else:
                color = (200, 200, 200)
                s = scale
            cv2.putText(frame, line, (12, y), font, s, color, 1, cv2.LINE_AA)

        # Recording indicator (bright red dot) — drawn by main.py passing a flag
        return frame

    # ------------------------------------------------------------------
    # Recording indicator
    # ------------------------------------------------------------------

    @staticmethod
    def draw_recording_badge(frame: np.ndarray) -> np.ndarray:
        """Draw a red ● REC badge in the top-right corner of *frame*."""
        h, w = frame.shape[:2]
        cv2.circle(frame, (w - 30, 24), 8, (0, 0, 220), -1)
        cv2.putText(
            frame, "REC",
            (w - 20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 2, cv2.LINE_AA,
        )
        return frame

    @staticmethod
    def draw_mode_badge(frame: np.ndarray, mode: str) -> np.ndarray:
        """Draw the active inference-mode badge at the bottom of *frame*."""
        h, w = frame.shape[:2]
        text = f"[ {mode.upper()} ]"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        x = (w - tw) // 2
        y = h - 14
        cv2.putText(
            frame, text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 180), 2, cv2.LINE_AA,
        )
        return frame

    def clear_trails(self) -> None:
        """Erase all stored track trails (e.g. after a mode switch)."""
        self._trails.clear()