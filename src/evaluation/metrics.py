"""
ArchiGen 3D - Evaluation metrics (all computed, never hardcoded).

Image metrics expect tensors in [-1, 1], shape [B, 3, H, W]; they return
PER-IMAGE values so callers can average over images (not over batches).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Image synthesis metrics (facades)
# ---------------------------------------------------------------------------
def l1_per_image(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute error in the [-1, 1] space (same scale as the training L1)."""
    return (pred - target).abs().flatten(1).mean(1)


def psnr_per_image(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """PSNR in dB with images mapped to [0, 1]."""
    p, t = (pred.clamp(-1, 1) + 1) / 2, (target.clamp(-1, 1) + 1) / 2
    mse = ((p - t) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return 10.0 * torch.log10(1.0 / mse)


def _gaussian_window(size: int = 11, sigma: float = 1.5, channels: int = 3, device=None):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w = (g[:, None] @ g[None, :])[None, None]
    return w.expand(channels, 1, size, size).contiguous()


def ssim_per_image(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Standard SSIM (Wang et al. 2004), 11x11 Gaussian, images in [0, 1], averaged over RGB."""
    p, t = (pred.clamp(-1, 1) + 1) / 2, (target.clamp(-1, 1) + 1) / 2
    c = p.shape[1]
    w = _gaussian_window(channels=c, device=p.device)
    mu_p = F.conv2d(p, w, groups=c)
    mu_t = F.conv2d(t, w, groups=c)
    s_pp = F.conv2d(p * p, w, groups=c) - mu_p ** 2
    s_tt = F.conv2d(t * t, w, groups=c) - mu_t ** 2
    s_pt = F.conv2d(p * t, w, groups=c) - mu_p * mu_t
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu_p * mu_t + c1) * (2 * s_pt + c2)) / ((mu_p ** 2 + mu_t ** 2 + c1) * (s_pp + s_tt + c2))
    return ssim_map.flatten(1).mean(1)


class OptionalLPIPS:
    """LPIPS if the `lpips` package is installed (pip install lpips), otherwise disabled.

    The first use downloads pretrained backbone weights (internet required once).
    """

    def __init__(self, net: str = "alex", device: str = "cpu"):
        self.available = False
        self.reason = ""
        try:
            import lpips  # type: ignore

            self.model = lpips.LPIPS(net=net, verbose=False).to(device).eval()
            self.available = True
            self.net = net
        except Exception as exc:  # not installed / no internet
            self.reason = f"{type(exc).__name__}: {exc}"

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.available:
            return None
        return self.model(pred.clamp(-1, 1), target.clamp(-1, 1)).flatten()


# ---------------------------------------------------------------------------
# Segmentation metrics (floorplans)
# ---------------------------------------------------------------------------
class ConfusionMatrix:
    """Accumulates a confusion matrix over a dataset; pixels == ignore_index are skipped."""

    def __init__(self, num_classes: int, ignore_index: int = 255, device="cpu"):
        self.n = num_classes
        self.ignore_index = ignore_index
        self.mat = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        valid = target != self.ignore_index
        idx = target[valid].long() * self.n + pred[valid].long()
        self.mat += torch.bincount(idx, minlength=self.n * self.n).reshape(self.n, self.n)

    def compute(self, class_names: Optional[List[str]] = None) -> Dict[str, object]:
        m = self.mat.double()
        tp = m.diag()
        gt = m.sum(1)
        pr = m.sum(0)
        union = gt + pr - tp
        iou = torch.where(union > 0, tp / union.clamp_min(1), torch.full_like(tp, float("nan")))
        present = gt > 0
        names = class_names or [str(i) for i in range(self.n)]
        return {
            "pixel_acc": (tp.sum() / m.sum().clamp_min(1)).item(),
            # mIoU over classes that appear in the ground truth of this split
            "miou": iou[present].mean().item() if present.any() else float("nan"),
            "per_class_iou": {names[i]: (None if torch.isnan(iou[i]) else round(iou[i].item(), 6))
                              for i in range(self.n)},
            "gt_pixels": {names[i]: int(gt[i].item()) for i in range(self.n)},
        }
