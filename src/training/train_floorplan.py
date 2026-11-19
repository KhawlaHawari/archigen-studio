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
    python -m src.training.train_floorplan --recipe v3 --init-checkpoint models/checkpoints/floorplan_unetpp_best.pt

Recipe v3 (Run 3) = v2 + Lovász-softmax on the room head (optimises room mIoU), x3 oversampling
of plans with rare rooms (dining, office, technical), drawing-style augmentation
(src.data.style_augment) on half of the training images, an extra validation pass on a fixed restyled copy of the val split, and
checkpoint selection on the mean of the clean and restyled validation room mIoU. With
--init-checkpoint it fine-tunes an existing model instead of starting from ImageNet.
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
from src.data.style_augment import stylize
from src.models.unet_plus_plus import FloorplanMultiTaskLoss, UNetPlusPlusMultiTask, load_imagenet_resnet34
from src.training.checkpoints import load_weights
from src.training.trainer_base import BaseTrainer, TrainResult

CACHE_DIR = DATA_DIR / "processed" / "cubicasa_cache"
FEEDBACK_CACHE = DATA_DIR / "processed" / "feedback_cache" / "train"   # studio corrections (feedback loop)
METRICS_DIR = MODELS_DIR / "metrics"
IGNORE_INDEX = 255


class NPZFloorplanDataset(Dataset):
    """Loads rasterized CubiCasa5K .npz files ('image', 'structure_mask', 'room_mask')."""

    def __init__(self, npz_paths: List[Path], augment: bool = False, image_size: int = 512,
                 style_p: float = 0.0, style_seed: Optional[int] = None):
        """style_p: probability of re-drawing a training image in another style (recipe v3).
        style_seed: when set, EVERY image is restyled deterministically (seed + index), which
        gives a fixed restyled validation set."""
        self.npz_paths = list(npz_paths)
        self.augment = augment
        self.image_size = image_size
        self.style_p = style_p
        self.style_seed = style_seed

    def __len__(self) -> int:
        return len(self.npz_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with np.load(self.npz_paths[idx]) as data:
            img, struct, room = data["image"], data["structure_mask"], data["room_mask"]
        if self.style_seed is not None:
            img = stylize(img, struct, np.random.default_rng(self.style_seed + idx))
        elif self.augment and self.style_p > 0 and torch.rand(1).item() < self.style_p:
            img = stylize(img, struct, np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item())))
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
    num_workers: int = 0, seed: int = 42, image_size: int = 512, style_p: float = 0.0,
    rare_room_oversample: float = 0.0, include_feedback: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    g = torch.Generator().manual_seed(seed)
    train_files = split_files("train", cache_dir, limit)
    if include_feedback:   # feedback loop: user-corrected plans join the TRAIN split only (never val/test)
        extra = sorted(FEEDBACK_CACHE.glob("*.npz"))
        print(f"[Floorplan] feedback loop: +{len(extra)} user-corrected plans in the training split", flush=True)
        train_files = train_files + extra
    train_ds = NPZFloorplanDataset(train_files, augment=True, image_size=image_size, style_p=style_p)
    sampler = None
    if rare_room_oversample > 0:   # recipe v3: plans containing the weakest room types are drawn more often
        weights = rare_room_sample_weights(train_ds.npz_paths, rare_room_oversample)
        sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True,
                                                         generator=g)
    val_ds = NPZFloorplanDataset(split_files("val", cache_dir, limit), augment=False, image_size=image_size)
    pin = torch.cuda.is_available()
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=sampler is None, sampler=sampler,
                   num_workers=num_workers, generator=g,
                   pin_memory=pin, persistent_workers=num_workers > 0),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                   pin_memory=pin, persistent_workers=num_workers > 0),
    )


def load_floorplan_params() -> Dict[str, Any]:
    with open(BASE_DIR / "params.yaml", encoding="utf8") as f:
        return yaml.safe_load(f).get("training_floorplan", {})


RECIPES = ("v1", "v2", "v3")
# Rare AND weak room classes after Run 2 (test IoU / share of room pixels): office 0.15 / 0.1 %,
# dining 0.54 / 0.5 %, technical 0.56 / 0.4 %. (Closet is weak too but common, so it is not boosted.)
RARE_ROOM_CLASSES = tuple(ROOM_CLASSES.index(n) for n in ("dining", "office", "technical"))


def rare_room_sample_weights(npz_paths: List[Path], boost: float) -> List[float]:
    """Sampling weight 1 + boost for plans that contain a rare room type, 1 otherwise."""
    weights = []
    for f in npz_paths:
        with np.load(f) as d:
            present = np.isin(RARE_ROOM_CLASSES, d["room_mask"]).any()
        weights.append(1.0 + boost * float(present))
    share = sum(w > 1 for w in weights) / max(1, len(weights))
    print(f"[Floorplan] rare-room oversampling: {share:.0%} of train plans weighted x{1 + boost:g}", flush=True)
    return weights
MODERN_RECIPES = ("v2", "v3")   # ImageNet encoder, cosine LR, room class weights
STYLE_VAL_SEED = 2026


def compute_room_class_weights(cache_dir: Path = CACHE_DIR, max_files: int = 600,
                               power: float = 0.5, cap: float = 10.0) -> List[float]:
    """Inverse-frequency^power room weights from the TRAIN split (ignore label 255 excluded).

    Normalised to mean 1 and capped, so rare rooms (dining, office, technical) receive gradient
    without letting a class with a handful of pixels dominate the loss.
    """
    counts = np.zeros(len(ROOM_CLASSES), dtype=np.float64)
    for f in split_files("train", cache_dir)[:max_files]:
        with np.load(f) as d:
            room = d["room_mask"]
        counts += np.bincount(room[room != IGNORE_INDEX].ravel(), minlength=len(ROOM_CLASSES))[: len(ROOM_CLASSES)]
    freq = counts / max(1.0, counts.sum())
    w = np.where(freq > 0, (freq + 1e-12) ** (-power), 0.0)
    w = w / w[w > 0].mean()
    return [float(min(cap, x)) for x in w]


def _merged_wall(siou: Dict[str, Any]) -> float:
    return float(np.mean([v for v in (siou["exterior_wall"], siou["interior_wall"]) if v is not None] or [0.0]))


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
        recipe: str = "v1",
        room_weights: Optional[List[float]] = None,
        init_checkpoint: Optional[Path] = None,
    ):
        assert recipe in RECIPES, recipe
        self.init_checkpoint = Path(init_checkpoint) if init_checkpoint else None
        self.styled_val_loader: Optional[DataLoader] = None
        # recipe v2: ImageNet-pretrained ResNet-34 encoder, cosine LR, per-class room weights.
        self.recipe = recipe
        self.room_weights = room_weights
        self.pretrained_tensors = 0
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
        self.best_metric_name = "val_score" if recipe == "v3" else "val_room_miou"
        self.best_metric = -1.0  # maximize
        # Mixed precision (CUDA only): ~2x faster on T4 GPUs.
        self.amp = bool(amp and self.device == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.criterion = FloorplanMultiTaskLoss(
            struct_weights=FLOORPLAN_CFG.structure_weights, room_weights=room_weights,
            room_lovasz_weight=0.5 if recipe == "v3" else 0.0).to(self.device)
        self.checkpoint_meta = {"data_source": "real:cubicasa5k", "dataset_split": "official train/val",
                                "recipe": recipe}
        self.hparams = {
            "epochs": epochs, "batch_size": batch_size, "lr": lr, "weight_decay": weight_decay,
            "gradient_accumulation_steps": self.gradient_accumulation_steps, "seed": seed,
            "structure_weights": FLOORPLAN_CFG.structure_weights, "input_size": image_size,
            "amp": self.amp, "recipe": recipe,
            "pretrained_encoder": "imagenet_resnet34" if self.pretrained_tensors else None,
            "pretrained_tensors": self.pretrained_tensors,
            "room_weights": [round(w, 4) for w in room_weights] if room_weights else None,
            "scheduler": "cosine" if recipe in MODERN_RECIPES else "reduce_on_plateau",
            "init_checkpoint": str(self.init_checkpoint) if self.init_checkpoint else None,
            "style_augment_p": 0.5 if recipe == "v3" else 0.0,
            "room_lovasz_weight": 0.5 if recipe == "v3" else 0.0,
            "rare_room_oversample": 2.0 if recipe == "v3" else 0.0,
            "early_stopping_patience": early_stopping_patience,
        }
        self.best_val_metrics: Dict[str, Any] = {}

    def build_model(self) -> torch.nn.Module:
        model = UNetPlusPlusMultiTask(
            in_channels=FLOORPLAN_CFG.in_channels,
            num_structure_classes=FLOORPLAN_CFG.num_structure_classes,
            num_room_classes=FLOORPLAN_CFG.num_room_classes,
        )
        if self.init_checkpoint is not None:
            meta = load_weights(model, self.init_checkpoint, require_real_data=True)
            print(f"[Floorplan] fine-tuning from {self.init_checkpoint} (epoch {meta.get('epoch')})", flush=True)
        elif self.recipe in MODERN_RECIPES:
            self.pretrained_tensors = load_imagenet_resnet34(model.encoder)
            print(f"[Floorplan] ImageNet ResNet-34 weights loaded into encoder ({self.pretrained_tensors} tensors)",
                  flush=True)
        return model

    def build_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        return {"optimizer": torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)}

    def build_schedulers(self) -> Dict[str, Any]:
        if self.recipe in MODERN_RECIPES:
            return {"scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizers["optimizer"], T_max=max(1, self.epochs), eta_min=1e-6)}
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
        loss, s_res, r_res = self._evaluate_loader(val_loader)
        siou = s_res["per_class_iou"]
        metrics = {
            "loss": loss,
            "val_struct_miou": s_res["miou"],
            "val_room_miou": r_res["miou"],
            "val_room_pixel_acc": r_res["pixel_acc"],
            "val_merged_wall_iou": _merged_wall(siou),
            "val_door_iou": siou["door"] or 0.0,
            "val_window_iou": siou["window"] or 0.0,
        }
        if self.styled_val_loader is not None:   # recipe v3: same val plans, re-drawn in other styles
            _, s2, r2 = self._evaluate_loader(self.styled_val_loader)
            metrics.update({
                "val_styled_room_miou": r2["miou"],
                "val_styled_struct_miou": s2["miou"],
                "val_styled_merged_wall_iou": _merged_wall(s2["per_class_iou"]),
            })
            metrics["val_score"] = 0.5 * (metrics["val_room_miou"] + metrics["val_styled_room_miou"])
        if metrics.get(self.best_metric_name, metrics["val_room_miou"]) > self.best_metric:
            self.best_val_metrics = {**metrics, "epoch": self.epochs_completed,
                                     "structure_per_class_iou": siou, "room_per_class_iou": r_res["per_class_iou"]}
        return {k: float(v) for k, v in metrics.items()}

    @torch.no_grad()
    def _evaluate_loader(self, loader: DataLoader):
        cm_s = ConfusionMatrix(len(STRUCTURE_CLASSES), IGNORE_INDEX, self.device)
        cm_r = ConfusionMatrix(len(ROOM_CLASSES), IGNORE_INDEX, self.device)
        total_loss, n_img = 0.0, 0
        for batch in loader:
            images, struct_t, room_t = (t.to(self.device) for t in batch)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                s_logits, r_logits = self.model(images)
            s_logits, r_logits = s_logits.float(), r_logits.float()
            total_loss += self.criterion(s_logits, struct_t, r_logits, room_t)["total_loss"].item() * len(images)
            n_img += len(images)
            cm_s.update(s_logits.argmax(1), struct_t)
            cm_r.update(r_logits.argmax(1), room_t)
        return total_loss / max(1, n_img), cm_s.compute(STRUCTURE_CLASSES), cm_r.compute(ROOM_CLASSES)

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
            "model_name": self.model_name, "split": "val", "selection_metric": self.best_metric_name,
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
    recipe: str = "v1",
    init_checkpoint: Optional[Path] = None,
    include_feedback: bool = False,
) -> TrainResult:
    """Train on REAL rasterized CubiCasa5K. Unspecified hyperparameters come from params.yaml.

    recipe v2 defaults (overridable): batch 4 x accumulation 2 (effective 8, as in v1), lr 3e-4 with
    cosine decay, ImageNet encoder, room class weights from the training split, early stopping patience 8.
    recipe v3: same, plus style augmentation (p=0.5) and the restyled validation pass; when
    fine-tuning (init_checkpoint) the default lr drops to 1e-4.
    """
    p = load_floorplan_params()
    v2 = recipe in MODERN_RECIPES
    room_weights = compute_room_class_weights(cache_dir or CACHE_DIR) if v2 else None
    if room_weights:
        print("[Floorplan] room class weights:", [round(w, 2) for w in room_weights], flush=True)
    trainer = FloorplanTrainer(
        epochs=epochs or p.get("epochs", 25),
        batch_size=batch_size or (4 if v2 else p.get("batch_size", 2)),
        lr=lr or ((1e-4 if init_checkpoint else 3e-4) if v2 else p.get("lr", 1e-4)),
        weight_decay=p.get("weight_decay", 1e-4),
        gradient_accumulation_steps=2 if v2 else p.get("gradient_accumulation_steps", 4),
        early_stopping_patience=8 if v2 else 5,
        seed=seed,
        tag=tag,
        image_size=image_size,
        amp=amp,
        recipe=recipe,
        room_weights=room_weights,
        init_checkpoint=init_checkpoint,
    )
    trainer.max_train_batches = max_batches
    trainer.time_budget_s = time_budget_min * 60 if time_budget_min else None
    train_loader, val_loader = make_cubicasa_loaders(
        cache_dir or CACHE_DIR, trainer.batch_size, limit, num_workers, seed, image_size,
        style_p=0.5 if recipe == "v3" else 0.0,
        rare_room_oversample=2.0 if recipe == "v3" else 0.0,
        include_feedback=include_feedback,
    )
    if recipe == "v3":
        styled = NPZFloorplanDataset(split_files("val", cache_dir or CACHE_DIR, limit), augment=False,
                                     image_size=image_size, style_seed=STYLE_VAL_SEED)
        trainer.styled_val_loader = DataLoader(styled, batch_size=trainer.batch_size, shuffle=False,
                                               num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                                               persistent_workers=num_workers > 0)
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
    ap.add_argument("--recipe", choices=RECIPES, default="v1",
                    help="v1 = Run-1 behaviour; v2 = ImageNet encoder + cosine LR + room class weights; "
                         "v3 = v2 + drawing-style augmentation, selected on clean + restyled val room mIoU")
    ap.add_argument("--init-checkpoint", type=Path, default=None,
                    help="Fine-tune from this checkpoint instead of ImageNet initialisation")
    ap.add_argument("--include-feedback", action="store_true",
                    help="Add user-corrected plans (python -m src.data.feedback_dataset) to the training split")
    a = ap.parse_args()
    r = train_floorplan(a.epochs, a.batch_size, a.lr, a.seed, a.cache_dir, a.max_batches, a.limit, a.tag,
                        a.workers, a.image_size, a.amp, a.time_budget_min, a.recipe, a.init_checkpoint,
                        a.include_feedback)
    print(f"Floorplan training finished: run_id={r.run_id}, best {r.best_metric_name}={r.best_metric:.4f}")
