"""
Training Script: CMP Facades - semantic map -> facade photo (REAL DATA ONLY)

Two modes, same generator / data / split / resolution (Cadrage, objective 2):
  * l1_baseline : U-Net generator trained with L1 loss only (reference model)
  * pix2pix     : same U-Net + PatchGAN, LSGAN + L1 + feature matching + perceptual loss

Two recipes:
  * v1 (default, Run 1): selection and early stopping on val L1, frozen random-filter
    "perceptual" loss, no EMA, no LR decay, no jitter.
  * v2 (Run 2): recipe of the stronger historical model -
      pix2pix: real LPIPS loss (SqueezeNet), EMA generator (decay 0.999), constant LR then
               linear decay over the second half, checkpoint selected on validation LPIPS
               (AlexNet) of the EMA generator, no L1-based early stopping;
      both modes: pix2pix-style scale/crop jitter (286 -> random 256 crop) on the train split.
  * v3 (Run 3): v2 plus a 2-scale PatchGAN discriminator (pix2pixHD) and, for pix2pix,
    checkpoint selection on validation KID x1000 (the metric the model is compared on)
    instead of LPIPS. The L1 baseline is unchanged from v2 (it has no discriminator).

Data: data/raw/facades/{train,val} (400 / 100 pairs). The TEST split is never
touched here - use `python -m src.evaluation.evaluate facade` after freezing the protocol.

Usage:
    python -m src.training.train_facade --mode l1_baseline
    python -m src.training.train_facade --mode pix2pix --recipe v2 --epochs 100
    python -m src.training.train_facade --mode pix2pix --recipe v3 --epochs 100
    python -m src.training.train_facade --mode pix2pix --epochs 1 --max-batches 5   # smoke test
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from PIL import Image

from src.config import BASE_DIR, FACADE_CFG, MODELS_DIR
from src.data.facade_dataset import make_facade_loaders, to_uint8
from src.evaluation.metrics import OptionalLPIPS, l1_per_image, psnr_per_image, ssim_per_image
from src.models.pix2pix_cgan import (FacadeCompositeLoss, FacadeGenerator, LPIPSLoss, MultiScaleDiscriminator,
                                      PatchGANDiscriminator)
from src.training.trainer_base import BaseTrainer, TrainResult

MODES = ("pix2pix", "l1_baseline")
RECIPES = ("v1", "v2", "v3")
MODERN_RECIPES = ("v2", "v3")   # EMA, LR decay, jitter, no L1 early stopping
SAMPLES_DIR = MODELS_DIR / "samples"
METRICS_DIR = MODELS_DIR / "metrics"


def load_facade_params() -> Dict[str, Any]:
    with open(BASE_DIR / "params.yaml", encoding="utf8") as f:
        return yaml.safe_load(f).get("training_facade", {})


def model_name_for(mode: str, seed: int, tag: str = "") -> str:
    base = "facade_pix2pix" if mode == "pix2pix" else "facade_unet_l1"
    return base + (f"_seed{seed}" if seed != 42 else "") + (f"_{tag}" if tag else "")


def lambda_lr_factor(decay_start: int, total_epochs: int):
    """LambdaLR factor: 1.0 until `decay_start`, then linear decay reaching ~0 at the last epoch."""
    span = max(1, total_epochs - decay_start)

    def factor(epoch: int) -> float:
        if epoch < decay_start:
            return 1.0
        return max(0.0, 1.0 - (epoch - decay_start + 1) / (span + 1))

    return factor


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
        recipe: str = "v1",
        ema_decay: float = 0.999,
    ):
        assert mode in MODES, mode
        assert recipe in RECIPES, recipe
        self.mode = mode
        self.recipe = recipe
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.lambda_l1 = lambda_l1
        self.lambda_fm = lambda_fm
        self.lambda_perceptual = lambda_perceptual
        self.v2_gan = recipe in MODERN_RECIPES and mode == "pix2pix"
        self.v3_gan = recipe == "v3" and mode == "pix2pix"
        # v2 pix2pix never stops on L1: the adversarial loss trades pixel L1 for realism.
        self.early_stopping_patience = (epochs + 1) if self.v2_gan else early_stopping_patience
        self.ema_decay = ema_decay
        self.lr_decay_start = epochs // 2
        self.discriminator = None
        super().__init__(
            model_name=model_name_for(mode, seed, tag),
            experiment_name="ArchiGen3D-Facade-Training",
            seed=seed,
        )
        self.best_metric_name = "val_kid_x1000" if self.v3_gan else ("val_lpips" if self.v2_gan else "val_l1")
        self.best_metric = float("inf")
        self.criterion = FacadeCompositeLoss(lambda_l1, lambda_fm, lambda_perceptual).to(self.device)
        self.ema = None
        self.val_lpips = None
        if self.v2_gan:
            self.criterion.perceptual_loss = LPIPSLoss("squeeze").to(self.device)
            self.ema = copy.deepcopy(self.model).eval()
            for prm in self.ema.parameters():
                prm.requires_grad_(False)
            self.val_lpips = OptionalLPIPS(net="alex", device=self.device)
            if not self.val_lpips.available:
                raise RuntimeError(
                    f"recipe v2 selects on validation LPIPS but LPIPS is unavailable: {self.val_lpips.reason}. "
                    "Install it with `pip install lpips`."
                )
        self.val_kid = None
        if self.v3_gan:
            try:
                from torchmetrics.image.kid import KernelInceptionDistance  # needs torch-fidelity
                self.val_kid = KernelInceptionDistance(subset_size=50, subsets=100, reset_real_features=False).to(self.device)
            except Exception as exc:
                raise RuntimeError(f"recipe v3 selects on validation KID but it is unavailable: {exc}. "
                                   "Install it with `pip install torchmetrics torch-fidelity`.") from exc
            self._kid_real_done = False
        self.l1 = torch.nn.L1Loss()
        self.checkpoint_meta = {
            "data_source": "real:cmp_facades", "mode": mode, "recipe": recipe,
            "dataset_split": "train400/val100", "selection_metric": self.best_metric_name,
            "weights": "ema" if self.ema is not None else "live",
        }
        self.hparams = {
            "mode": mode, "recipe": recipe, "epochs": epochs, "batch_size": batch_size, "lr_g": lr_g,
            "lr_d": lr_d if mode == "pix2pix" else None, "seed": seed,
            "lambda_l1": lambda_l1 if mode == "pix2pix" else 1.0,
            "lambda_fm": lambda_fm if mode == "pix2pix" else 0.0,
            "lambda_perceptual": lambda_perceptual if mode == "pix2pix" else 0.0,
            "perceptual_loss": ("lpips_squeeze" if self.v2_gan else "random_filters") if mode == "pix2pix" else None,
            "ema_decay": ema_decay if self.v2_gan else None,
            "lr_decay_start_epoch": self.lr_decay_start if recipe in MODERN_RECIPES else None,
            "jitter": recipe in MODERN_RECIPES,
            "discriminator": ("multiscale_2" if self.v3_gan else "patchgan_70") if mode == "pix2pix" else None,
            "early_stopping_patience": self.early_stopping_patience,
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
            if self.v3_gan:
                self.discriminator = MultiScaleDiscriminator(
                    in_channels=6, ndf=FACADE_CFG.ndf, num_scales=2, use_spectral_norm=True
                ).to(self.device)
            else:
                self.discriminator = PatchGANDiscriminator(
                    in_channels=6, ndf=FACADE_CFG.ndf, use_spectral_norm=True
                ).to(self.device)
            opts["optimizer_d"] = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_d, betas=(0.5, 0.999))
        return opts

    def build_schedulers(self) -> Dict[str, Any]:
        if self.recipe not in MODERN_RECIPES:
            return {}
        # Constant LR for the first half, then linear decay towards 0 (pix2pix schedule).
        factor = lambda_lr_factor(self.lr_decay_start, self.epochs)
        return {f"lr_{name}": torch.optim.lr_scheduler.LambdaLR(opt, factor) for name, opt in self.optimizers.items()}

    def export_state_dict(self) -> Dict[str, Any]:
        # The inference checkpoint carries the EMA generator when one exists.
        return (self.ema if self.ema is not None else self.model).state_dict()

    def save_checkpoint(self, epoch, metrics, is_best=False):
        if self.discriminator is not None:
            self.checkpoint_meta["discriminator_state_dict"] = self.discriminator.state_dict()
        return super().save_checkpoint(epoch, metrics, is_best)

    @torch.no_grad()
    def _update_ema(self) -> None:
        d = self.ema_decay
        for e, p in zip(self.ema.parameters(), self.model.parameters()):
            e.mul_(d).add_(p.detach(), alpha=1.0 - d)
        for e, b in zip(self.ema.buffers(), self.model.buffers()):
            e.copy_(b)

    def _eval_generator(self) -> torch.nn.Module:
        return self.ema if self.ema is not None else self.model

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
        if self.ema is not None:
            self._update_ema()
        return {
            "loss": g["g_total"].item(), "g_adv": g["g_adv"].item(), "g_l1": g["g_l1"].item(),
            "g_fm": g["g_fm"].item(), "g_perceptual": g["g_perceptual"].item(), "d_total": d_loss["d_total"].item(),
        }

    def validate_step(self, batch: Any) -> Dict[str, float]:  # kept for API compatibility
        raise NotImplementedError("Use compute_validation_metrics")

    @torch.no_grad()
    def compute_validation_metrics(self, val_loader: Any) -> Dict[str, float]:
        """Per-image averages (not per-batch) on the validation split, using the EMA generator if any."""
        gen = self._eval_generator()
        gen.eval()
        l1s, psnrs, ssims, lps = [], [], [], []
        for batch in val_loader:
            masks, reals = (t.to(self.device) for t in batch)
            fakes = gen(masks)
            l1s.append(l1_per_image(fakes, reals))
            psnrs.append(psnr_per_image(fakes, reals))
            ssims.append(ssim_per_image(fakes, reals))
            if self.val_lpips is not None:
                lps.append(self.val_lpips(fakes, reals))
            if self.val_kid is not None:
                to_u8 = lambda t: ((t.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
                if not self._kid_real_done:            # real features are computed once and kept
                    self.val_kid.update(to_u8(reals), real=True)
                self.val_kid.update(to_u8(fakes), real=False)
            if self._fixed_val_batch is None:
                self._fixed_val_batch = (masks[:4].cpu(), reals[:4].cpu())
        l1, ps, ss = torch.cat(l1s), torch.cat(psnrs), torch.cat(ssims)
        metrics = {"loss": l1.mean().item(), "val_l1": l1.mean().item(),
                   "val_psnr": ps.mean().item(), "val_ssim": ss.mean().item(), "val_n_images": float(len(l1))}
        if lps:
            metrics["val_lpips"] = torch.cat(lps).mean().item()
        if self.val_kid is not None:
            torch.manual_seed(42)                      # same random subsets every epoch
            metrics["val_kid_x1000"] = float(self.val_kid.compute()[0]) * 1000
            self._kid_real_done = True
            self.val_kid.reset()                       # drops fake features only (reset_real_features=False)
        if metrics[self.best_metric_name] < self.best_metric:
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
        fakes = self._eval_generator()(masks.to(self.device)).cpu()
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
            "model_name": self.model_name, "split": "val", "selection_metric": self.best_metric_name,
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
    recipe: str = "v1",
) -> TrainResult:
    """Train on REAL CMP Facades. Unspecified hyperparameters come from params.yaml."""
    p = load_facade_params()
    default_epochs = p.get("epochs_v2", 100) if recipe in MODERN_RECIPES else p.get("epochs", 50)
    trainer = FacadeTrainer(
        mode=mode,
        epochs=epochs or default_epochs,
        batch_size=batch_size or p.get("batch_size", 2),
        lr_g=lr_g or p.get("lr_g", 2e-4),
        lr_d=lr_d or p.get("lr_d", 2e-4),
        lambda_l1=p.get("lambda_l1", 30.0),
        lambda_fm=p.get("lambda_fm", 10.0),
        lambda_perceptual=p.get("lambda_perceptual", 10.0),
        early_stopping_patience=20 if recipe in MODERN_RECIPES else 10,
        seed=seed,
        tag=tag,
        recipe=recipe,
    )
    trainer.max_train_batches = max_batches
    trainer.time_budget_s = time_budget_min * 60 if time_budget_min else None
    train_loader, val_loader = make_facade_loaders(
        batch_size=trainer.batch_size, num_workers=num_workers, seed=seed, jitter=(recipe in MODERN_RECIPES)
    )
    print(f"[Facade] mode={mode} recipe={recipe} train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
          f"epochs={trainer.epochs} select={trainer.best_metric_name} device={trainer.device}", flush=True)
    return trainer.fit(train_loader=train_loader, val_loader=val_loader)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Train facade generator on real CMP Facades")
    ap.add_argument("--mode", choices=MODES, default="pix2pix")
    ap.add_argument("--recipe", choices=RECIPES, default="v1",
                    help="v1 = Run-1 behaviour; v2 = LPIPS loss + EMA + LR decay + jitter, selected on val LPIPS; "
                         "v3 = v2 + 2-scale discriminator, pix2pix selected on val KID")
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
                     time_budget_min=a.time_budget_min, recipe=a.recipe)
    print(f"Facade training finished: run_id={r.run_id}, best {r.best_metric_name}={r.best_metric:.4f}")
