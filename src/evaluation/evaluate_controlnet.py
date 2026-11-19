"""
Evaluate a ControlNet facade generator with exactly the Pix2Pix protocol (test split, once).

Each label map is upscaled to 512 (nearest), generated with 25 UniPC steps (guidance 5.0, fixed
seed per image), downscaled to 256 (bicubic) and scored like Pix2Pix: KID, LPIPS, L1/PSNR/SSIM and
the FCN-score (layout fidelity).

    python -m src.evaluation.evaluate_controlnet --controlnet models/controlnet_facade      # fine-tuned
    python -m src.evaluation.evaluate_controlnet --controlnet lllyasviel/control_v11p_sd15_seg --name facade_controlnet_init
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.evaluation.evaluate import evaluate_facade_generator
from src.training.train_controlnet import BASE_MODEL, NEGATIVE, OUT_DIR, PROMPT


def load_pipeline(controlnet_path: str, device: str = "cuda"):
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
    dtype = torch.float16 if device == "cuda" else torch.float32
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(BASE_MODEL, controlnet=controlnet, torch_dtype=dtype,
                                                             safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


def make_generate(pipe, steps: int = 25, guidance: float = 5.0, seed: int = 42):
    counter = {"i": 0}

    @torch.no_grad()
    def generate(masks: torch.Tensor) -> torch.Tensor:
        cond = F.interpolate((masks.float() + 1) / 2, scale_factor=2, mode="nearest").clamp(0, 1)
        n = masks.shape[0]
        gens = [torch.Generator(device=pipe.device).manual_seed(seed + counter["i"] + k) for k in range(n)]
        counter["i"] += n
        out = pipe([PROMPT] * n, image=cond.to(pipe.device, pipe.unet.dtype), negative_prompt=[NEGATIVE] * n,
                   num_inference_steps=steps, guidance_scale=guidance, generator=gens, output_type="np").images
        x = torch.from_numpy(np.ascontiguousarray(out)).permute(0, 3, 1, 2).float() * 2 - 1      # [-1, 1], 512 px
        x = F.interpolate(x, size=masks.shape[-2:], mode="bicubic", align_corners=False, antialias=True)
        return x.clamp(-1, 1).to(masks.device)

    return generate


def evaluate(controlnet_path: str, name: str, steps: int = 25, guidance: float = 5.0, batch_size: int = 8,
             split: str = "test"):
    pipe = load_pipeline(controlnet_path)
    extra = {"generator": "Stable Diffusion 1.5 + ControlNet", "controlnet": controlnet_path,
             "inference": {"steps": steps, "scheduler": "UniPC", "guidance_scale": guidance, "resolution": 512,
                           "scored_at": "256 px (bicubic downscale)", "prompt": PROMPT, "negative_prompt": NEGATIVE}}
    return evaluate_facade_generator(make_generate(pipe, steps, guidance), name, batch_size, split, extra)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ControlNet facade evaluation (same protocol as Pix2Pix)")
    ap.add_argument("--controlnet", default=str(OUT_DIR))
    ap.add_argument("--name", default="facade_controlnet")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--batch-size", type=int, default=8)
    a = ap.parse_args()
    res = evaluate(a.controlnet, a.name, a.steps, a.guidance, a.batch_size)
    print(json.dumps({k: res[k] for k in ("n_images", "l1", "psnr_db", "ssim", "lpips", "lpips_squeeze",
                                          "kid_x1000", "fcn_score")}, indent=2, default=str))
