"""
VisionSense — Main Application Entry Point
===========================================
Orchestrates all VisionSense components into a real-time computer vision
pipeline with a full keyboard shortcut interface.

Keyboard Shortcuts
------------------
  q          Quit the application
  r          Toggle video recording (start / stop)
  s          Save a screenshot (snapshot)
  h          Toggle the HUD overlay
  t          Cycle inference mode (detect → segment → pose)
  c          Toggle region counter display
  l          Toggle line counter display
  m          Toggle heatmap overlay
  d          Toggle dataset collection mode
  ESC        Quit (same as q)

Usage
-----
    python main.py                        # default config.yaml
    python main.py --config my_cfg.yaml   # custom config file
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

from src.analytics import Analytics
from src.camera import Camera
from src.config import Config
from src.dataset_collector import DatasetCollector
from src.detector import ObjectDetector
from src.fps import FPSCounter, InferenceTimer
from src.line_counter import LineCounter, RegionCounter
from src.recorder import Recorder
from src.renderer import Renderer
from src.snapshot import Snapshot


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _setup_logging(cfg: Config) -> None:
    """Configure root logger with file and console handlers."""
    level = getattr(logging, cfg.log_level.upper(), logging.INFO)
    log_path = Path(cfg.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list = [logging.StreamHandler(sys.stdout)]
    try:
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    except OSError as exc:
        print(f"WARNING: Cannot open log file '{log_path}': {exc}", file=sys.stderr)

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="VisionSense — Real-Time Computer Vision System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        metavar="FILE",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    return parser.parse_args()


def main() -> int:
    """Application entry point.

    Returns
    -------
    int
        Exit code (0 = success, 1 = error).
    """
    args = parse_args()

    # ----------------------------------------------------------------
    # Load configuration
    # ----------------------------------------------------------------
    cfg = Config.from_yaml(args.config)
    _setup_logging(cfg)
    cfg.ensure_directories()

    # Persist default config if it didn't exist yet
    if not Path(args.config).exists():
        cfg.to_yaml(args.config)

    logger.info("=" * 60)
    logger.info("VisionSense starting up …")
    logger.info("=" * 60)

    # ----------------------------------------------------------------
    # Initialise components
    # ----------------------------------------------------------------
    try:
        camera = Camera(cfg)
    except RuntimeError as exc:
        logger.critical("Camera initialisation failed: %s", exc)
        return 1

    frame_w, frame_h = camera.frame_size

    try:
        detector = ObjectDetector(cfg)
        detector.warm_up()
    except Exception as exc:
        logger.critical("Detector initialisation failed: %s", exc)
        camera.release()
        return 1

    analytics = Analytics(cfg, frame_size=(frame_w, frame_h))
    fps_counter = FPSCounter(window=30)
    inf_timer = InferenceTimer(history=60)
    line_counter = LineCounter(cfg)
    region_counter = RegionCounter(cfg)
    renderer = Renderer(cfg)
    recorder = Recorder(cfg, frame_size=(frame_w, frame_h), fps=float(camera.fps or 30))
    snapshot_mgr = Snapshot(cfg)
    dataset_collector = DatasetCollector(cfg)

    # Write dataset.yaml once we have class names
    dataset_collector.write_dataset_yaml(detector.class_names)

    # ----------------------------------------------------------------
    # Application state
    # ----------------------------------------------------------------
    show_region = cfg.region_counter.enabled
    show_line = cfg.line_counter.enabled
    dataset_mode = False
    frame_index: int = 0

    logger.info("Entering main capture loop. Press 'q' or ESC to quit.")
    logger.info(
        "Shortcuts: [r] record  [s] snapshot  [h] HUD  "
        "[t] mode  [c] region  [l] line  [m] heatmap  [d] dataset"
    )

    # ----------------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------------
    try:
        while True:
            # ── Capture ──────────────────────────────────────────────
            success, frame = camera.read()
            if not success or frame is None:
                logger.error("Frame capture failed — exiting.")
                break

            frame_index += 1

            # ── Inference ─────────────────────────────────────────────
            with inf_timer:
                results = detector.infer(frame)

            inf_ms = inf_timer.last_ms

            # ── Analytics ─────────────────────────────────────────────
            analytics.update(results)
            summary = analytics.get_summary()

            # ── Counting ──────────────────────────────────────────────
            if show_line:
                line_counter.update(results)

            if show_region:
                region_counter.update(results)

            # ── FPS ───────────────────────────────────────────────────
            current_fps = fps_counter.tick()

            # ── Render ────────────────────────────────────────────────
            heatmap = analytics.get_heatmap() if cfg.renderer.show_heatmap else None

            frame = renderer.draw(
                frame=frame,
                results=results,
                analytics_summary=summary,
                fps=current_fps,
                inf_ms=inf_ms,
                mode=detector.mode,
                heatmap=heatmap,
            )

            # ── Counters overlay ──────────────────────────────────────
            if show_line:
                frame = line_counter.draw(frame)

            if show_region:
                frame = region_counter.draw(frame)

            # ── Badges ────────────────────────────────────────────────
            if recorder.is_recording:
                frame = Renderer.draw_recording_badge(frame)

            if dataset_mode:
                _draw_dataset_badge(frame)

            frame = Renderer.draw_mode_badge(frame, detector.mode)

            # ── Dataset auto-collect ──────────────────────────────────
            if dataset_mode:
                dataset_collector.maybe_collect(frame, results, frame_index)

            # ── Recording ─────────────────────────────────────────────
            if recorder.is_recording:
                recorder.write(frame)

            # ── Display ───────────────────────────────────────────────
            cv2.imshow(cfg.window_name, frame)

            # ── Keyboard input ────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):  # q or ESC
                logger.info("Quit requested by user.")
                break

            elif key == ord("r"):
                if recorder.is_recording:
                    path = recorder.stop()
                    logger.info("Recording saved → %s", path)
                else:
                    recorder.start()
                    logger.info("Recording started.")

            elif key == ord("s"):
                path = snapshot_mgr.save(frame)
                logger.info("Snapshot saved → %s", path)

            elif key == ord("h"):
                cfg.renderer.show_hud = not cfg.renderer.show_hud
                logger.info("HUD %s.", "ON" if cfg.renderer.show_hud else "OFF")

            elif key == ord("t"):
                new_mode = detector.cycle_mode()
                renderer.clear_trails()
                logger.info("Mode → %s", new_mode)

            elif key == ord("c"):
                show_region = not show_region
                logger.info("Region counter %s.", "ON" if show_region else "OFF")

            elif key == ord("l"):
                show_line = not show_line
                logger.info("Line counter %s.", "ON" if show_line else "OFF")

            elif key == ord("m"):
                cfg.renderer.show_heatmap = not cfg.renderer.show_heatmap
                logger.info(
                    "Heatmap %s.", "ON" if cfg.renderer.show_heatmap else "OFF"
                )

            elif key == ord("d"):
                dataset_mode = not dataset_mode
                logger.info(
                    "Dataset collection %s.", "ON" if dataset_mode else "OFF"
                )

    except KeyboardInterrupt:
        logger.info("Interrupted by Ctrl-C.")

    finally:
        # ── Graceful shutdown ─────────────────────────────────────────
        logger.info("Shutting down …")

        if recorder.is_recording:
            path = recorder.stop()
            logger.info("Recording finalised → %s", path)

        report_path = analytics.save_report()
        logger.info("Analytics report saved → %s", report_path)

        camera.release()
        logger.info("VisionSense exited cleanly.")

    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _draw_dataset_badge(frame) -> None:
    """Draw an orange DATASET badge to indicate collection mode is active."""
    h, w = frame.shape[:2]
    cv2.putText(
        frame,
        "● DATASET",
        (w - 130, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 165, 255),
        2,
        cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    sys.exit(main())