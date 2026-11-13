"""
Training Script: CMP Facades - semantic map -> facade photo (REAL DATA ONLY)

Two modes, same generator / data / split / resolution (Cadrage, objective 2):
  * l1_baseline : U-Net generator trained with L1 loss only (reference model)
  * pix2pix     : same U-Net + PatchGAN, LSGAN + L1 + feature matching + feature loss

Data: data/raw/facades/{train,val} (400 / 100 pairs). The TEST split is never
touched here - use `python -m src.evaluation.evaluate_facade` after freezing
the protocol. Model selection uses validation L1 only.

Usage:
    python -m src.training.train_facade --mode l1_baseline
    python -m src.training.train_facade --mode pix2pix --seed 1
    python -m src.training.train_facade --mode pix2pix --epochs 1 --max-batches 5   # smoke test
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from PIL import Image

from src.config import BASE_DIR, FACADE_CFG, MODELS_DIR
from src.data.facade_dataset import make_facade_loaders, to_uint8
from src.evaluation.metrics import l1_per_image, psnr_per_image, ssim_per_image
from src.models.pix2pix_cgan import FacadeCompositeLoss, FacadeGenerator, PatchGANDiscriminator
from src.training.trainer_base import BaseTrainer, TrainResult

MODES = ("pix2pix", "l1_baseline")
SAMPLES_DIR = MODELS_DIR / "samples"
METRICS_DIR = MODELS_DIR / "metrics"


def load_facade_params() -> Dict[str, Any]:
    with open(BASE_DIR / "params.yaml", encoding="utf8") as f:
        return yaml.safe_load(f).get("training_facade", {})


def model_name_for(mode: str, seed: int, tag: str = "") -> str:
    base = "facade_pix2pix" if mode == "pix2pix" else "facade_unet_l1"
    return base + (f"_seed{seed}" if seed != 42 else "") + (f"_{tag}" if tag else "")


class FacadeTrainer(BaseTrainer):
    def __init__(
        self,
        mode: str = "pix2pix",
        epochs: int = 50,
        batch_size: int = 2,
        lr_g: float = 2e-4,
        lr_d: float = 2e-4,
        lambda_l1: float = 30.0,
        lambda_fm: float = 10.0,
        lambda_perceptual: float = 10.0,
        early_stopping_patience: int = 10,
        seed: int = 42,
        tag: str = "",
    ):
        assert mode in MODES, mode
        self.mode = mode
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.lambda_l1 = lambda_l1
        self.lambda_fm = lambda_fm
        self.lambda_perceptual = lambda_perceptual
        self.early_stopping_patience = early_stopping_patience
        self.discriminator = None
        super().__init__(
            model_name=model_name_for(mode, seed, tag),
            experiment_name="ArchiGen3D-Facade-Training",
            seed=seed,
        )
        self.best_metric_name = "val_l1"
        self.best_metric = float("inf")
        self.criterion = FacadeCompositeLoss(lambda_l1, lambda_fm, lambda_perceptual).to(self.device)
        self.l1 = torch.nn.L1Loss()
        self.checkpoint_meta = {"data_source": "real:cmp_facades", "mode": mode, "dataset_split": "train400/val100"}
        self.hparams = {
            "mode": mode, "epochs": epochs, "batch_size": batch_size, "lr_g": lr_g,
            "lr_d": lr_d if mode == "pix2pix" else None, "seed": seed,
            "lambda_l1": lambda_l1 if mode == "pix2pix" else 1.0,
            "lambda_fm": lambda_fm if mode == "pix2pix" else 0.0,
            "lambda_perceptual": lambda_perceptual if mode == "pix2pix" else 0.0,
            "ngf": FACADE_CFG.ngf, "ndf": FACADE_CFG.ndf, "image_size": 256,
        }
        self.best_val_metrics: Dict[str, float] = {}
        self._fixed_val_batch = None

    # ------------------------------------------------------------------
    def build_model(self) -> torch.nn.Module:
        return FacadeGenerator(
            in_channels=FACADE_CFG.in_channels, out_channels=FACADE_CFG.out_channels, ngf=FACADE_CFG.ngf
        )

    def build_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        opts = {"optimizer_g": torch.optim.Adam(self.model.parameters(), lr=self.lr_g, betas=(0.5, 0.999))}
        if self.mode == "pix2pix":
            self.discriminator = PatchGANDiscriminator(
                in_channels=6, ndf=FACADE_CFG.ndf, use_spectral_norm=True
            ).to(self.device)
            opts["optimizer_d"] = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_d, betas=(0.5, 0.999))
        return opts

    def save_checkpoint(self, epoch, metrics, is_best=False):
        if self.discriminator is not None:
            self.checkpoint_meta["discriminator_state_dict"] = self.discriminator.state_dict()
        return super().save_checkpoint(epoch, metrics, is_best)

    # ------------------------------------------------------------------
    def train_step(self, batch: Any) -> Dict[str, float]:
        masks, reals = (t.to(self.device) for t in batch)
        opt_g = self.optimizers["optimizer_g"]

        if self.mode == "l1_baseline":
            opt_g.zero_grad()
            loss = self.l1(self.model(masks), reals)
            loss.backward()
            opt_g.step()
            return {"loss": loss.item(), "g_l1": loss.item()}

        opt_d = self.optimizers["optimizer_d"]
        self.discriminator.train()
        # ---- Discriminator ----
        opt_d.zero_grad()
        with torch.no_grad():
            fakes = self.model(masks)
        d_real, _ = self.discriminator(masks, reals)
        d_fake, _ = self.discriminator(masks, fakes)
        d_loss = self.criterion.discriminator_loss(d_real, d_fake)
        d_loss["d_total"].backward()
        opt_d.step()
        # ---- Generator ----
        opt_g.zero_grad()
        fakes = self.model(masks)
        d_fake, fake_feats = self.discriminator(masks, fakes)
        with torch.no_grad():
            _, real_feats = self.discriminator(masks, reals)
        g = self.criterion.generator_loss(d_fake, real_feats, fake_feats, fakes, reals)
        g["g_total"].backward()
        opt_g.step()
        return {
            "loss": g["g_total"].item(), "g_adv": g["g_adv"].item(), "g_l1": g["g_l1"].item(),
            "g_fm": g["g_fm"].item(), "g_perceptual": g["g_perceptual"].item(), "d_total": d_loss["d_total"].item(),
        }

    def validate_step(self, batch: Any) -> Dict[str, float]:  # kept for API compatibility
        raise NotImplementedError("Use compute_validation_metrics")

    @torch.no_grad()
    def compute_validation_metrics(self, val_loader: Any) -> Dict[str, float]:
        """Per-image averages (not per-batch) of L1, PSNR, SSIM on the validation split."""
        self.model.eval()
        l1s, psnrs, ssims = [], [], []
        for batch in val_loader:
            masks, reals = (t.to(self.device) for t in batch)
            fakes = self.model(masks)
            l1s.append(l1_per_image(fakes, reals))
            psnrs.append(psnr_per_image(fakes, reals))
            ssims.append(ssim_per_image(fakes, reals))
            if self._fixed_val_batch is None:
                self._fixed_val_batch = (masks[:4].cpu(), reals[:4].cpu())
        l1, ps, ss = torch.cat(l1s), torch.cat(psnrs), torch.cat(ssims)
        metrics = {"loss": l1.mean().item(), "val_l1": l1.mean().item(),
                   "val_psnr": ps.mean().item(), "val_ssim": ss.mean().item(), "val_n_images": float(len(l1))}
        if metrics["val_l1"] < self.best_metric:
            self.best_val_metrics = dict(metrics)
            self.best_val_metrics["epoch"] = self.epochs_completed
        self._save_sample_grid()
        return metrics

    @torch.no_grad()
    def _save_sample_grid(self) -> None:
        """Rows: input map | generated | target, on 4 fixed validation images."""
        if self._fixed_val_batch is None:
            return
        masks, reals = self._fixed_val_batch
        fakes = self.model(masks.to(self.device)).cpu()
        rows = [np.hstack([to_uint8(m), to_uint8(f), to_uint8(r)]) for m, f, r in zip(masks, fakes, reals)]
        out = SAMPLES_DIR / self.model_name
        out.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.vstack(rows)).save(out / f"epoch_{self.epochs_completed:03d}.png")

    def fit(self, train_loader, val_loader, epochs=None, start_epoch=0, early_stopping_patience=None,
            metric_mode="min") -> TrainResult:
        result = super().fit(
            train_loader=train_loader, val_loader=val_loader, epochs=epochs or self.epochs,
            start_epoch=start_epoch,
            early_stopping_patience=early_stopping_patience or self.early_stopping_patience,
            metric_mode="min",
        )
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        summary = {
            "model_name": self.model_name, "split": "val", "selection_metric": "val_l1",
            "best": self.best_val_metrics, "hparams": self.hparams, "epochs_completed": self.epochs_completed,
            "max_train_batches": self.max_train_batches, "device": self.device,
            "stopped_by_time_budget": self.stopped_by_time_budget,
            "time_budget_min": (self.time_budget_s / 60) if self.time_budget_s else None, "mlflow_run_id": self.run_id,
            "checkpoint": str(result.checkpoint_path),
        }
        (METRICS_DIR / f"{self.model_name}_val.json").write_text(json.dumps(summary, indent=2))
        return result


def train_facade(
    mode: str = "pix2pix",
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr_g: Optional[float] = None,
    lr_d: Optional[float] = None,
    seed: int = 42,
    max_batches: Optional[int] = None,
    tag: str = "",
    num_workers: int = 0,
    time_budget_min: Optional[float] = None,
) -> TrainResult:
    """Train on REAL CMP Facades. Unspecified hyperparameters come from params.yaml."""
    p = load_facade_params()
    trainer = FacadeTrainer(
        mode=mode,
        epochs=epochs or p.get("epochs", 50),
        batch_size=batch_size or p.get("batch_size", 2),
        lr_g=lr_g or p.get("lr_g", 2e-4),
        lr_d=lr_d or p.get("lr_d", 2e-4),
        lambda_l1=p.get("lambda_l1", 30.0),
        lambda_fm=p.get("lambda_fm", 10.0),
        lambda_perceptual=p.get("lambda_perceptual", 10.0),
        seed=seed,
        tag=tag,
    )
    trainer.max_train_batches = max_batches
    trainer.time_budget_s = time_budget_min * 60 if time_budget_min else None
    train_loader, val_loader = make_facade_loaders(
        batch_size=trainer.batch_size, num_workers=num_workers, seed=seed
    )
    print(f"[Facade] mode={mode} train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
          f"device={trainer.device}", flush=True)
    return trainer.fit(train_loader=train_loader, val_loader=val_loader)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Train facade generator on real CMP Facades")
    ap.add_argument("--mode", choices=MODES, default="pix2pix")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-batches", type=int, default=None, help="Batches per epoch (smoke test only)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--time-budget-min", type=float, default=None, help="Stop before exceeding this many minutes")
    a = ap.parse_args()
    r = train_facade(mode=a.mode, epochs=a.epochs, batch_size=a.batch_size, seed=a.seed,
                     max_batches=a.max_batches, tag=a.tag, num_workers=a.workers,
                     time_budget_min=a.time_budget_min)
    print(f"Facade training finished: run_id={r.run_id}, best {r.best_metric_name}={r.best_metric:.4f}")
