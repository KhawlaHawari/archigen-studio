"""
Base Trainer for ArchiGen 3D Models
Provides reproducible training utilities: deterministic seeding, MLflow run
context, checkpoint save/resume, early stopping, and structured metric logging.
"""

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from src.config import BASE_DIR, MLOPS_CFG


def seed_everything(seed: int = 42) -> None:
    """Set deterministic seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


@dataclass
class TrainResult:
    """Structured result of a training run."""
    run_id: str
    experiment_name: str
    best_metric: float
    best_metric_name: str
    checkpoint_path: Path
    epochs_completed: int
    mlflow_uri: str
    artifacts: Dict[str, Path] = field(default_factory=dict)


class BaseTrainer:
    """Base class for ArchiGen model trainers.

    Subclasses must implement:
      - build_model()
      - build_optimizers()
      - train_step(batch)
      - validate_step(batch)
      - compute_validation_metrics(val_loader)
    """

    def __init__(
        self,
        model_name: str,
        experiment_name: str,
        device: Optional[str] = None,
        seed: int = 42,
        mlflow_tracking_uri: Optional[str] = None,
    ):
        self.model_name = model_name
        self.experiment_name = experiment_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.mlflow_tracking_uri = mlflow_tracking_uri or MLOPS_CFG.tracking_uri

        seed_everything(seed)

        self.model = self.build_model().to(self.device)
        self.optimizers = self.build_optimizers()
        self.schedulers = self.build_schedulers()

        self.checkpoint_dir = BASE_DIR / "models" / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_metric = float("inf")
        self.best_metric_name = "val_loss"
        self.best_checkpoint_path: Optional[Path] = None
        self.epochs_completed = 0
        self.run_id: Optional[str] = None
        # Extra metadata stored in every checkpoint (e.g. data_source="real", mode=...).
        self.checkpoint_meta: Dict[str, Any] = {}
        # Hyperparameters logged to MLflow at run start.
        self.hparams: Dict[str, Any] = {}
        # Limit batches per epoch (smoke tests only; None = full epoch).
        self.max_train_batches: Optional[int] = None
        # Wall-clock budget in seconds (None = unlimited). Training stops before an epoch
        # that would not fit; the number of completed epochs is recorded in the metrics.
        self.time_budget_s: Optional[float] = None
        self.stopped_by_time_budget = False

        # MLflow
        self._mlflow = None
        self._init_mlflow()

    def _init_mlflow(self) -> None:
        try:
            import mlflow  # type: ignore
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            self._mlflow = mlflow
        except Exception as exc:  # pragma: no cover - MLflow optional in CI
            print(f"[WARN] MLflow not available ({exc}). Training will run without tracking.")
            self._mlflow = None

    # ------------------------------------------------------------------
    # Abstract hooks implemented by subclasses
    # ------------------------------------------------------------------
    def build_model(self) -> torch.nn.Module:
        raise NotImplementedError

    def build_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        raise NotImplementedError

    def build_schedulers(self) -> Dict[str, Any]:
        return {}

    def train_step(self, batch: Any) -> Dict[str, float]:
        raise NotImplementedError

    def validate_step(self, batch: Any) -> Dict[str, float]:
        raise NotImplementedError

    def compute_validation_metrics(self, val_loader: Any) -> Dict[str, float]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False) -> Path:
        """Save a training checkpoint with model, optimizers, and RNG state."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizers": {name: opt.state_dict() for name, opt in self.optimizers.items()},
            "schedulers": {name: sch.state_dict() for name, sch in self.schedulers.items()},
            "metrics": metrics,
            "seed": self.seed,
            "model_name": self.model_name,
            "hparams": self.hparams,
            **self.checkpoint_meta,
        }
        # Only "last" + "best" are kept on disk (one file per epoch would cost GBs).
        path = self.checkpoint_dir / f"{self.model_name}_last.pt"
        torch.save(checkpoint, str(path), _use_new_zipfile_serialization=False)

        if is_best:
            best_path = self.checkpoint_dir / f"{self.model_name}_best.pt"
            torch.save(checkpoint, str(best_path), _use_new_zipfile_serialization=False)
            self.best_checkpoint_path = best_path
            print(f"[Trainer] New best checkpoint saved: {best_path} (metric={metrics.get(self.best_metric_name)})")

        return path

    def load_checkpoint(self, path: Path) -> int:
        """Load checkpoint and resume training state. Returns the epoch to resume from."""
        checkpoint = torch.load(str(path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        for name, opt in self.optimizers.items():
            if name in checkpoint.get("optimizers", {}):
                opt.load_state_dict(checkpoint["optimizers"][name])
        for name, sch in self.schedulers.items():
            if name in checkpoint.get("schedulers", {}):
                sch.load_state_dict(checkpoint["schedulers"][name])
        start_epoch = checkpoint.get("epoch", 0) + 1
        self.best_metric = checkpoint.get("metrics", {}).get(self.best_metric_name, float("inf"))
        print(f"[Trainer] Resumed from {path} at epoch {start_epoch}")
        return start_epoch

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: Any,
        val_loader: Any,
        epochs: int,
        start_epoch: int = 0,
        early_stopping_patience: int = 5,
        metric_mode: str = "min",  # "min" or "max"
    ) -> TrainResult:
        """Run the full training loop with validation and early stopping."""
        print(f"[Trainer] Starting training for {self.model_name} on {self.device}")

        run_context = None
        if self._mlflow is not None:
            run_context = self._mlflow.start_run(run_name=f"{self.model_name}-run")
            run_context.__enter__()
            self.run_id = self._mlflow.active_run().info.run_id
            print(f"[Trainer] MLflow run started: {self.run_id}")
            params = {**self.hparams, **{k: v for k, v in self.checkpoint_meta.items()
                                         if isinstance(v, (str, int, float, bool))}}
            if params:
                self._mlflow.log_params({k: str(v)[:250] for k, v in params.items()})

        try:
            import time as _time
            t_start = _time.time()
            patience_counter = 0
            for epoch in range(start_epoch, epochs):
                if self.time_budget_s is not None and epoch > start_epoch:
                    elapsed = _time.time() - t_start
                    per_epoch = elapsed / (epoch - start_epoch)
                    if elapsed + per_epoch > self.time_budget_s:
                        print(f"[Trainer] Time budget reached after {epoch - start_epoch} epochs "
                              f"({elapsed/60:.1f} min, ~{per_epoch/60:.1f} min/epoch).", flush=True)
                        self.stopped_by_time_budget = True
                        break
                self.epochs_completed = epoch + 1

                # ---- Training ----
                self.model.train()
                train_metrics: Dict[str, float] = {}
                num_batches = 0
                for batch in train_loader:
                    if self.max_train_batches is not None and num_batches >= self.max_train_batches:
                        break
                    batch_metrics = self.train_step(batch)
                    for k, v in batch_metrics.items():
                        train_metrics[k] = train_metrics.get(k, 0.0) + v
                    num_batches += 1
                train_metrics = {k: v / max(1, num_batches) for k, v in train_metrics.items()}

                # ---- Validation ----
                self.model.eval()
                val_metrics = self.compute_validation_metrics(val_loader)

                # Combine and log
                epoch_metrics = {**{f"train_{k}": v for k, v in train_metrics.items()},
                                 **{(k if k.startswith("val_") else f"val_{k}"): v
                                    for k, v in val_metrics.items()}}
                if self._mlflow is not None and self.run_id is not None:
                    for k, v in epoch_metrics.items():
                        self._mlflow.log_metric(k, v, step=epoch)

                print(f"[Trainer] Epoch {epoch+1}/{epochs} | "
                      f"train_loss={train_metrics.get('loss', 0.0):.4f} | "
                      f"val_loss={val_metrics.get('loss', 0.0):.4f} | "
                      f"{self.best_metric_name}={val_metrics.get(self.best_metric_name, float('nan')):.4f}",
                      flush=True)

                # ---- Checkpointing & early stopping ----
                current_metric = val_metrics.get(self.best_metric_name, val_metrics.get("loss", 0.0))
                is_best = (
                    current_metric < self.best_metric if metric_mode == "min"
                    else current_metric > self.best_metric
                )
                if is_best:
                    self.best_metric = current_metric
                    patience_counter = 0
                else:
                    patience_counter += 1

                self.save_checkpoint(epoch, epoch_metrics, is_best=is_best)

                for sch in self.schedulers.values():
                    if hasattr(sch, "step"):
                        try:
                            sch.step()
                        except TypeError:
                            sch.step(epoch_metrics.get("val_loss", epoch_metrics.get("loss")))

                if patience_counter >= early_stopping_patience:
                    print(f"[Trainer] Early stopping triggered at epoch {epoch+1} "
                          f"(patience={early_stopping_patience})")
                    break

        finally:
            if self._mlflow is not None and run_context is not None:
                run_context.__exit__(None, None, None)

        result = TrainResult(
            run_id=self.run_id or "",
            experiment_name=self.experiment_name,
            best_metric=self.best_metric,
            best_metric_name=self.best_metric_name,
            checkpoint_path=self.best_checkpoint_path or self.checkpoint_dir / f"{self.model_name}_best.pt",
            epochs_completed=self.epochs_completed,
            mlflow_uri=self.mlflow_tracking_uri,
        )
        print(f"[Trainer] Training complete. Best {self.best_metric_name}={self.best_metric:.4f}")
        return result
