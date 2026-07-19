"""
VisionSense — YOLO Fine-Tuning Trainer
=========================================
Wraps the Ultralytics ``YOLO.train()`` API for one-command custom model
training from a collected dataset.  Exports the best checkpoint to
``models/custom/`` after training completes.

Usage
-----
    from src.trainer import Trainer
    from src.config import Config

    cfg = Config.from_yaml("config.yaml")
    trainer = Trainer(cfg)
    result = trainer.train("datasets/raw/dataset.yaml")
    print("Best model at:", trainer.best_model_path)
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

from src.config import Config

logger = logging.getLogger(__name__)


class Trainer:
    """Wrapper around Ultralytics YOLO training for custom datasets.

    Parameters
    ----------
    config : Config
        Root application configuration. Training parameters come from
        ``config.training`` and the base model from ``config.detection.model_path``.
    """

    def __init__(self, config: Config) -> None:
        self._cfg = config.training
        self._base_model_path = config.detection.model_path
        self._best_model_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def best_model_path(self) -> Optional[Path]:
        """Path to the best trained checkpoint, or ``None`` if not yet trained."""
        return self._best_model_path

    def train(self, dataset_yaml: str) -> dict:
        """Start a YOLO training run on *dataset_yaml*.

        Parameters
        ----------
        dataset_yaml : str
            Path to the Ultralytics-compatible ``dataset.yaml`` file.

        Returns
        -------
        dict
            Training results summary from Ultralytics.

        Raises
        ------
        FileNotFoundError
            If *dataset_yaml* does not exist.
        RuntimeError
            If the training process fails.
        """
        yaml_path = Path(dataset_yaml)
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"Dataset YAML not found: '{yaml_path}'. "
                "Run DatasetCollector.write_dataset_yaml() first."
            )

        logger.info(
            "Starting YOLO fine-tuning — base: '%s', data: '%s', epochs: %d",
            self._base_model_path, yaml_path, self._cfg.epochs,
        )

        model = YOLO(self._base_model_path)

        results = model.train(
            data=str(yaml_path),
            epochs=self._cfg.epochs,
            batch=self._cfg.batch,
            imgsz=self._cfg.imgsz,
            device=self._cfg.device or None,
            project=self._cfg.project,
            name=self._cfg.name,
            pretrained=self._cfg.pretrained,
            exist_ok=True,
        )

        # Locate and archive the best checkpoint
        run_dir = Path(self._cfg.project) / self._cfg.name
        best_src = run_dir / "weights" / "best.pt"
        if best_src.exists():
            dest = Path(self._cfg.project) / "best_custom.pt"
            shutil.copy2(best_src, dest)
            self._best_model_path = dest
            logger.info("Best model exported → %s", dest)
        else:
            logger.warning("best.pt not found in '%s'.", run_dir / "weights")

        return results

    def export(self, format: str = "onnx") -> Optional[Path]:
        """Export the best trained model to a deployment format.

        Parameters
        ----------
        format : str
            Export format: ``'onnx'``, ``'torchscript'``, ``'tflite'``, etc.

        Returns
        -------
        Path or None
            Path to the exported file, or ``None`` if no model has been trained.
        """
        if self._best_model_path is None:
            logger.warning("No trained model available for export.")
            return None

        logger.info("Exporting model to '%s' format …", format)
        model = YOLO(str(self._best_model_path))
        out = model.export(format=format)
        logger.info("Model exported → %s", out)
        return Path(str(out))
