"""
Optional 4x super-resolution of generated facade textures (post-processing, not part of any model).

Network: SRVGGNetCompact, re-implemented here (no basicsr dependency), loading the public
Real-ESRGAN weights `realesr-general-x4v3.pth` (Wang et al., 2021; BSD-3-Clause,
https://github.com/xinntao/Real-ESRGAN). It sharpens edges and frames of a texture; it does not
add information that the generator did not produce, so metrics are always reported WITHOUT it.

    python -m src.models.superres --download      # fetch the 5 MB weights into models/superres/
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

WEIGHTS = Path(__file__).resolve().parents[2] / "models" / "superres" / "realesr-general-x4v3.pth"
URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"


class SRVGGNetCompact(nn.Module):
    """Plain VGG-style SR network: conv + PReLU stack, pixel-shuffle upsampling, nearest-neighbour residual."""

    def __init__(self, num_feat: int = 64, num_conv: int = 32, upscale: int = 4):
        super().__init__()
        self.upscale = upscale
        layers = [nn.Conv2d(3, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
        for _ in range(num_conv):
            layers += [nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
        layers.append(nn.Conv2d(num_feat, 3 * upscale * upscale, 3, 1, 1))
        self.body = nn.Sequential(*layers)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.upsampler(self.body(x))
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


_MODEL: Optional[SRVGGNetCompact] = None


def available() -> bool:
    return WEIGHTS.exists()


def _model(device: str) -> SRVGGNetCompact:
    global _MODEL
    if _MODEL is None:
        m = SRVGGNetCompact()
        state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
        m.load_state_dict(state.get("params", state))
        _MODEL = m.eval()
    return _MODEL.to(device)


@torch.no_grad()
def upscale(img: Image.Image, device: str = "cpu", tile: int = 256, max_long_side: int = 2048) -> Image.Image:
    """4x super-resolution, processed in overlapping tiles to bound memory; capped at `max_long_side`."""
    model = _model(device)
    x = torch.from_numpy(np.asarray(img.convert("RGB"), np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
    _, _, h, w = x.shape
    out = torch.zeros(1, 3, h * 4, w * 4, device=device)
    pad = 8
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            ys, xs = max(0, y0 - pad), max(0, x0 - pad)
            ye, xe = min(h, y0 + tile + pad), min(w, x0 + tile + pad)
            sr = model(x[:, :, ys:ye, xs:xe]).clamp(0, 1)
            oy, ox = (y0 - ys) * 4, (x0 - xs) * 4
            th, tw = (min(y0 + tile, h) - y0) * 4, (min(x0 + tile, w) - x0) * 4
            out[:, :, y0 * 4:y0 * 4 + th, x0 * 4:x0 * 4 + tw] = sr[:, :, oy:oy + th, ox:ox + tw]
    res = Image.fromarray((out[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
    if max(res.size) > max_long_side:
        s = max_long_side / max(res.size)
        res = res.resize((round(res.width * s), round(res.height * s)), Image.Resampling.LANCZOS)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    if ap.parse_args().download:
        import urllib.request
        WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, WEIGHTS)
        print("saved", WEIGHTS)
