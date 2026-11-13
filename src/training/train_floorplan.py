"""
Training Script: UNet++ Multi-Task Floorplan Segmentation (CubiCasa5K, REAL DATA ONLY)

Dual-head UNet++ (5 structure classes + 12 room classes), weighted CE + Dice.
Data: data/processed/cubicasa_cache/{train,val}/*.npz produced by
`python -m src.data.rasterize_cubicasa` from the OFFICIAL CubiCasa5K splits
(4200 train / 400 val / 400 test). Test is never used here.

Room pixels labelled 255 (unknown room type, e.g. "Space Undefined") are ignored
by the loss and by the metrics.

Usage:
    python -m src.training.train_floorplan
    python -m src.training.train_floorplan --epochs 1 --max-batches 5 --limit 20 --tag smoke
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from src.config import BASE_DIR, DATA_DIR, FLOORPLAN_CFG, MODELS_DIR, ROOM_CLASSES, STRUCTURE_CLASSES
from src.evaluation.metrics import ConfusionMatrix
from src.models.unet_plus_plus import FloorplanMultiTaskLoss, UNetPlusPlusMultiTask
from src.training.trainer_base import BaseTrainer, TrainResult

CACHE_DIR = DATA_DIR / "processed" / "cubicasa_cache"
METRICS_DIR = MODELS_DIR / "metrics"
IGNORE_INDEX = 255


class NPZFloorplanDataset(Dataset):
    """Loads rasterized CubiCasa5K .npz files ('image', 'structure_mask', 'room_mask')."""

    def __init__(self, npz_paths: List[Path], augment: bool = False, image_size: int = 512):
        self.npz_paths = list(npz_paths)
        self.augment = augment
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.npz_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with np.load(self.npz_paths[idx]) as data:
            img, struct, room = data["image"], data["structure_mask"], data["room_mask"]
        img_t = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32) / 255.0)
        struct_t = torch.from_numpy(struct.astype(np.int64))
        room_t = torch.from_numpy(room.astype(np.int64))
        if img_t.shape[-1] != self.image_size:  # faster reduced-resolution training
            size = (self.image_size, self.image_size)
            img_t = torch.nn.functional.interpolate(img_t[None], size=size, mode="bilinear",
                                                    align_corners=False, antialias=True)[0]
            # nearest keeps labels (incl. 255 = ignore) valid
            struct_t = torch.nn.functional.interpolate(struct_t[None, None].float(), size=size,
                                                       mode="nearest")[0, 0].long()
            room_t = torch.nn.functional.interpolate(room_t[None, None].float(), size=size,
                                                     mode="nearest")[0, 0].long()
        if self.augment:  # same geometric flip on image and both masks
            if torch.rand(1).item() < 0.5:
                img_t, struct_t, room_t = img_t.flip(-1), struct_t.flip(-1), room_t.flip(-1)
            if torch.rand(1).item() < 0.5:
                img_t, struct_t, room_t = img_t.flip(-2), struct_t.flip(-2), room_t.flip(-2)
        return img_t, struct_t, room_t


def split_files(split: str, cache_dir: Path = CACHE_DIR, limit: Optional[int] = None) -> List[Path]:
    files = sorted((Path(cache_dir) / split).glob("*.npz"))
    if not files:
        raise FileNotFoundError(
            f"No rasterized CubiCasa5K files in {Path(cache_dir) / split}. "
            "Run: python -m src.data.rasterize_cubicasa"
        )
    return files[:limit] if limit else files


def make_cubicasa_loaders(
    cache_dir: Path = CACHE_DIR, batch_size: int = 2, limit: Optional[int] = None,
    num_workers: int = 0, seed: int = 42, image_size: int = 512,
) -> Tuple[DataLoader, DataLoader]:
    g = torch.Generator().manual_seed(seed)
    train_ds = NPZFloorplanDataset(split_files("train", cache_dir, limit), augment=True, image_size=image_size)
    val_ds = NPZFloorplanDataset(split_files("val", cache_dir, limit), augment=False, image_size=image_size)
    pin = torch.cuda.is_available()
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=g,
                   pin_memory=pin, persistent_workers=num_workers > 0),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                   pin_memory=pin, persistent_workers=num_workers > 0),
    )


def load_floorplan_params() -> Dict[str, Any]:
    with open(BASE_DIR / "params.yaml", encoding="utf8") as f:
        return yaml.safe_load(f).get("training_floorplan", {})


class FloorplanTrainer(BaseTrainer):
    def __init__(
        self,
        epochs: int = 25,
        batch_size: int = 2,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        gradient_accumulation_steps: int = 4,
        early_stopping_patience: int = 5,
        seed: int = 42,
        tag: str = "",
        image_size: int = 512,
        amp: bool = False,
    ):
        self.image_size = image_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.early_stopping_patience = early_stopping_patience
        self._micro_step = 0
        super().__init__(
            model_name="floorplan_unetpp" + (f"_seed{seed}" if seed != 42 else "") + (f"_{tag}" if tag else ""),
            experiment_name="ArchiGen3D-Floorplan-Training",
            seed=seed,
        )
        self.best_metric_name = "val_room_miou"
        self.best_metric = -1.0  # maximize
        # Mixed precision (CUDA only): ~2x faster on T4 GPUs.
        self.amp = bool(amp and self.device == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.criterion = FloorplanMultiTaskLoss(struct_weights=FLOORPLAN_CFG.structure_weights).to(self.device)
        self.checkpoint_meta = {"data_source": "real:cubicasa5k", "dataset_split": "official train/val"}
        self.hparams = {
            "epochs": epochs, "batch_size": batch_size, "lr": lr, "weight_decay": weight_decay,
            "gradient_accumulation_steps": self.gradient_accumulation_steps, "seed": seed,
            "structure_weights": FLOORPLAN_CFG.structure_weights, "input_size": image_size,
            "amp": self.amp,
        }
        self.best_val_metrics: Dict[str, Any] = {}

    def build_model(self) -> torch.nn.Module:
        return UNetPlusPlusMultiTask(
            in_channels=FLOORPLAN_CFG.in_channels,
            num_structure_classes=FLOORPLAN_CFG.num_structure_classes,
            num_room_classes=FLOORPLAN_CFG.num_room_classes,
        )

    def build_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        return {"optimizer": torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)}

    def build_schedulers(self) -> Dict[str, Any]:
        return {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizers["optimizer"], mode="max", factor=0.5, patience=3)}

    def train_step(self, batch: Any) -> Dict[str, float]:
        images, struct_t, room_t = (t.to(self.device) for t in batch)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            struct_logits, room_logits = self.model(images)
        # Loss in float32 for numerical stability.
        loss_dict = self.criterion(struct_logits.float(), struct_t, room_logits.float(), room_t)
        self.scaler.scale(loss_dict["total_loss"] / self.gradient_accumulation_steps).backward()

        # Real gradient accumulation: one optimizer step every N micro-batches.
        self._micro_step += 1
        if self._micro_step % self.gradient_accumulation_steps == 0:
            self._optimizer_step()

        return {
            "loss": loss_dict["total_loss"].item(),
            "loss_struct": loss_dict["loss_struct"].item(),
            "loss_room": loss_dict["loss_room"].item(),
        }

    def _optimizer_step(self) -> None:
        self.scaler.step(self.optimizers["optimizer"])
        self.scaler.update()
        self.optimizers["optimizer"].zero_grad()

    def validate_step(self, batch: Any) -> Dict[str, float]:
        raise NotImplementedError("Use compute_validation_metrics")

    @torch.no_grad()
    def compute_validation_metrics(self, val_loader: Any) -> Dict[str, float]:
        # Flush a partial accumulation window so no gradient leaks into the next epoch.
        if self._micro_step % self.gradient_accumulation_steps != 0:
            self._optimizer_step()
            self._micro_step = 0

        self.model.eval()
        cm_s = ConfusionMatrix(len(STRUCTURE_CLASSES), IGNORE_INDEX, self.device)
        cm_r = ConfusionMatrix(len(ROOM_CLASSES), IGNORE_INDEX, self.device)
        total_loss, n_img = 0.0, 0
        for batch in val_loader:
            images, struct_t, room_t = (t.to(self.device) for t in batch)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                s_logits, r_logits = self.model(images)
            s_logits, r_logits = s_logits.float(), r_logits.float()
            total_loss += self.criterion(s_logits, struct_t, r_logits, room_t)["total_loss"].item() * len(images)
            n_img += len(images)
            cm_s.update(s_logits.argmax(1), struct_t)
            cm_r.update(r_logits.argmax(1), room_t)
        s, r = cm_s.compute(STRUCTURE_CLASSES), cm_r.compute(ROOM_CLASSES)
        siou = s["per_class_iou"]
        metrics = {
            "loss": total_loss / max(1, n_img),
            "val_struct_miou": s["miou"],
            "val_room_miou": r["miou"],
            "val_room_pixel_acc": r["pixel_acc"],
            "val_merged_wall_iou": float(np.mean(
                [v for v in (siou["exterior_wall"], siou["interior_wall"]) if v is not None] or [0.0])),
            "val_door_iou": siou["door"] or 0.0,
            "val_window_iou": siou["window"] or 0.0,
        }
        if metrics["val_room_miou"] > self.best_metric:
            self.best_val_metrics = {**metrics, "epoch": self.epochs_completed,
                                     "structure_per_class_iou": siou, "room_per_class_iou": r["per_class_iou"]}
        return {k: float(v) for k, v in metrics.items()}

    def fit(self, train_loader, val_loader, epochs=None, start_epoch=0, early_stopping_patience=None,
            metric_mode="max") -> TrainResult:
        result = super().fit(
            train_loader=train_loader, val_loader=val_loader, epochs=epochs or self.epochs,
            start_epoch=start_epoch,
            early_stopping_patience=early_stopping_patience or self.early_stopping_patience,
            metric_mode="max",
        )
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        summary = {
            "model_name": self.model_name, "split": "val", "selection_metric": "val_room_miou",
            "best": self.best_val_metrics, "hparams": self.hparams, "epochs_completed": self.epochs_completed,
            "max_train_batches": self.max_train_batches, "n_train": len(train_loader.dataset),
            "stopped_by_time_budget": self.stopped_by_time_budget,
            "time_budget_min": (self.time_budget_s / 60) if self.time_budget_s else None,
            "n_val": len(val_loader.dataset), "device": self.device, "mlflow_run_id": self.run_id,
            "checkpoint": str(result.checkpoint_path),
        }
        (METRICS_DIR / f"{self.model_name}_val.json").write_text(json.dumps(summary, indent=2, default=float))
        return result


def train_floorplan(
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    seed: int = 42,
    cache_dir: Optional[Path] = None,
    max_batches: Optional[int] = None,
    limit: Optional[int] = None,
    tag: str = "",
    num_workers: int = 0,
    image_size: int = 512,
    amp: bool = False,
    time_budget_min: Optional[float] = None,
) -> TrainResult:
    """Train on REAL rasterized CubiCasa5K. Unspecified hyperparameters come from params.yaml."""
    p = load_floorplan_params()
    trainer = FloorplanTrainer(
        epochs=epochs or p.get("epochs", 25),
        batch_size=batch_size or p.get("batch_size", 2),
        lr=lr or p.get("lr", 1e-4),
        weight_decay=p.get("weight_decay", 1e-4),
        gradient_accumulation_steps=p.get("gradient_accumulation_steps", 4),
        seed=seed,
        tag=tag,
        image_size=image_size,
        amp=amp,
    )
    trainer.max_train_batches = max_batches
    trainer.time_budget_s = time_budget_min * 60 if time_budget_min else None
    train_loader, val_loader = make_cubicasa_loaders(
        cache_dir or CACHE_DIR, trainer.batch_size, limit, num_workers, seed, image_size
    )
    print(f"[Floorplan] train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
          f"device={trainer.device} size={image_size} amp={trainer.amp}", flush=True)
    return trainer.fit(train_loader=train_loader, val_loader=val_loader)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Train UNet++ Multi-Task on real CubiCasa5K")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--max-batches", type=int, default=None, help="Batches per epoch (smoke test only)")
    ap.add_argument("--limit", type=int, default=None, help="Files per split (smoke test only)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--image-size", type=int, default=512, help="Train resolution (256 = ~4x faster)")
    ap.add_argument("--amp", action="store_true", help="Mixed precision (CUDA only)")
    ap.add_argument("--time-budget-min", type=float, default=None, help="Stop before exceeding this many minutes")
    a = ap.parse_args()
    r = train_floorplan(a.epochs, a.batch_size, a.lr, a.seed, a.cache_dir, a.max_batches, a.limit, a.tag,
                        a.workers, a.image_size, a.amp, a.time_budget_min)
    print(f"Floorplan training finished: run_id={r.run_id}, best {r.best_metric_name}={r.best_metric:.4f}")
