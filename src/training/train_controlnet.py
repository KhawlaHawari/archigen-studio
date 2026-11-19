"""
Fine-tune a ControlNet (Stable Diffusion 1.5) to turn CMP facade label maps into facade photos.

* Starts from the public segmentation ControlNet (lllyasviel/control_v11p_sd15_seg) and SD 1.5;
  only the ControlNet is trained, the UNet / VAE / text encoder stay frozen (fp16).
* Data: the same 400 CMP training pairs as Pix2Pix (256 px, upscaled to 512: nearest for the label
  map, bicubic for the photo), horizontal flips. Val / test are never used for training.
* Text: one fixed prompt, dropped 10 % of the time so classifier-free guidance works at inference.
* Fits a 16 GB T4: fp16 frozen parts, mixed precision, gradient checkpointing, batch 2 x accumulation 2.

    python -m src.training.train_controlnet --max-steps 6000 --time-budget-min 330
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from src.config import MODELS_DIR
from src.data.facade_dataset import CMPFacadesDataset

BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROLNET_INIT = "lllyasviel/control_v11p_sd15_seg"
OUT_DIR = MODELS_DIR / "controlnet_facade"
PROMPT = "a photo of a building facade, frontal view, realistic, high quality architectural photograph"
NEGATIVE = "text, letters, watermark, signature, blurry, lowres, distorted, cartoon, painting"


def to_512(masks: torch.Tensor, photos: torch.Tensor):
    """[-1,1] 256 px pairs -> condition in [0,1] (nearest) and target in [-1,1] (bicubic) at 512 px."""
    cond = F.interpolate((masks + 1) / 2, scale_factor=2, mode="nearest")
    target = F.interpolate(photos, scale_factor=2, mode="bicubic", align_corners=False).clamp(-1, 1)
    return cond, target


def train(max_steps: int = 6000, batch_size: int = 2, grad_accum: int = 2, lr: float = 1e-5,
          time_budget_min: float | None = None, workers: int = 2, seed: int = 42,
          sample_every: int = 1000) -> dict:
    from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    torch.manual_seed(seed)
    device = "cuda"
    fp16 = torch.float16
    tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL, subfolder="text_encoder", torch_dtype=fp16).to(device)
    vae = AutoencoderKL.from_pretrained(BASE_MODEL, subfolder="vae", torch_dtype=fp16).to(device)
    unet = UNet2DConditionModel.from_pretrained(BASE_MODEL, subfolder="unet", torch_dtype=fp16).to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_INIT).to(device)   # fp32 master weights
    for m in (text_encoder, vae, unet):
        m.requires_grad_(False)
        m.eval()
    controlnet.enable_gradient_checkpointing()
    controlnet.train()

    with torch.no_grad():
        ids = tokenizer([PROMPT, ""], padding="max_length", max_length=tokenizer.model_max_length,
                        truncation=True, return_tensors="pt").input_ids.to(device)
        emb_prompt, emb_empty = text_encoder(ids)[0].chunk(2)

    ds = CMPFacadesDataset("train", augment=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    opt = torch.optim.AdamW(controlnet.parameters(), lr=lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda")
    step, micro, t0, losses = 0, 0, time.time(), []
    history = []
    stop = False
    while not stop:
        for masks, photos in loader:
            cond, target = to_512(masks.to(device), photos.to(device))
            with torch.no_grad():
                latents = vae.encode(target.to(fp16)).latent_dist.sample() * vae.config.scaling_factor
            noise = torch.randn_like(latents)
            t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy = noise_scheduler.add_noise(latents, noise, t)
            drop = (torch.rand(latents.shape[0], device=device) < 0.1)[:, None, None]
            emb = torch.where(drop, emb_empty.expand(latents.shape[0], -1, -1), emb_prompt.expand(latents.shape[0], -1, -1))
            with torch.autocast("cuda", dtype=fp16):
                down, mid = controlnet(noisy, t, encoder_hidden_states=emb, controlnet_cond=cond, return_dict=False)
                pred = unet(noisy, t, encoder_hidden_states=emb,
                            down_block_additional_residuals=[d.to(fp16) for d in down],
                            mid_block_additional_residual=mid.to(fp16)).sample
            loss = F.mse_loss(pred.float(), noise.float()) / grad_accum
            scaler.scale(loss).backward()
            losses.append(loss.item() * grad_accum)
            micro += 1
            if micro % grad_accum:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(controlnet.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 100 == 0:
                avg = float(np.mean(losses[-100 * grad_accum:]))
                history.append({"step": step, "loss": avg})
                print(f"[ControlNet] step {step}/{max_steps} | loss={avg:.4f} | {(time.time() - t0) / 60:.0f} min", flush=True)
            if step % sample_every == 0 or step == max_steps:
                controlnet.save_pretrained(OUT_DIR)
            budget_hit = time_budget_min and (time.time() - t0) / 60 > time_budget_min
            if step >= max_steps or budget_hit:
                if budget_hit:
                    print("[ControlNet] time budget reached", flush=True)
                stop = True
                break
    controlnet.to(torch.float16).save_pretrained(OUT_DIR)      # final weights in fp16 (half the size)
    summary = {"base_model": BASE_MODEL, "controlnet_init": CONTROLNET_INIT, "steps": step, "batch_size": batch_size,
               "grad_accum": grad_accum, "lr": lr, "resolution": 512, "prompt": PROMPT, "prompt_dropout": 0.1,
               "train_pairs": len(ds), "minutes": round((time.time() - t0) / 60, 1), "loss_history": history,
               "data_source": "real:cmp_facades (train split only)", "output": str(OUT_DIR)}
    (MODELS_DIR / "metrics").mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "metrics" / "facade_controlnet_train.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fine-tune a segmentation ControlNet on CMP Facades")
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--time-budget-min", type=float, default=None)
    a = ap.parse_args()
    s = train(a.max_steps, a.batch_size, a.grad_accum, a.lr, a.time_budget_min, a.workers)
    print(f"ControlNet finished: {s['steps']} steps in {s['minutes']} min -> {s['output']}")
