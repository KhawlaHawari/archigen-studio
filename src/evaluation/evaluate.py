"""
ArchiGen 3D - FINAL evaluation on the held-out TEST split.

Run this ONLY after the protocol is frozen (model chosen on validation).
Refuses checkpoints that were not trained on real data.

    python -m src.evaluation.evaluate facade    --checkpoint models/checkpoints/facade_pix2pix_best.pt
    python -m src.evaluation.evaluate facade    --checkpoint models/checkpoints/facade_unet_l1_best.pt
    python -m src.evaluation.evaluate floorplan --checkpoint models/checkpoints/floorplan_unetpp_best.pt

Outputs models/metrics/<checkpoint-stem>_test.json (+ a comparison grid for facades).
Facade metrics: L1 ([-1,1] scale), PSNR (dB), SSIM, LPIPS (if `pip install lpips`),
all averaged PER IMAGE over the 106 test pairs.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.config import FACADE_CFG, FLOORPLAN_CFG, MODELS_DIR, ROOM_CLASSES, STRUCTURE_CLASSES
from src.evaluation.metrics import ConfusionMatrix, OptionalLPIPS, l1_per_image, psnr_per_image, ssim_per_image
from src.training.checkpoints import load_weights

METRICS_DIR = MODELS_DIR / "metrics"


def _env() -> Dict[str, Any]:
    return {"device": "cuda" if torch.cuda.is_available() else "cpu", "torch": torch.__version__,
            "python": platform.python_version(), "date": time.strftime("%Y-%m-%d %H:%M:%S")}


def _stats(x: torch.Tensor) -> Dict[str, float]:
    return {"mean": x.mean().item(), "std": x.std().item() if len(x) > 1 else 0.0,
            "min": x.min().item(), "max": x.max().item()}


@torch.no_grad()
def evaluate_facade(checkpoint: Path, batch_size: int = 8, split: str = "test") -> Dict[str, Any]:
    from src.data.facade_dataset import CMPFacadesDataset, to_uint8
    from src.models.pix2pix_cgan import FacadeGenerator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = FacadeGenerator(FACADE_CFG.in_channels, FACADE_CFG.out_channels, FACADE_CFG.ngf)
    meta = load_weights(gen, checkpoint, require_real_data=True)
    gen.to(device).eval()

    ds = CMPFacadesDataset(split, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    lp = OptionalLPIPS(device=device)
    l1s, psnrs, ssims, lpips_v, rows = [], [], [], [], []
    for masks, reals in loader:
        masks, reals = masks.to(device), reals.to(device)
        fakes = gen(masks)
        l1s.append(l1_per_image(fakes, reals).cpu())
        psnrs.append(psnr_per_image(fakes, reals).cpu())
        ssims.append(ssim_per_image(fakes, reals).cpu())
        v = lp(fakes, reals)
        if v is not None:
            lpips_v.append(v.cpu())
        if len(rows) < 6:
            for m, f, r in zip(masks, fakes, reals):
                if len(rows) < 6:
                    rows.append(np.hstack([to_uint8(m), to_uint8(f), to_uint8(r)]))

    l1, ps, ss = torch.cat(l1s), torch.cat(psnrs), torch.cat(ssims)
    result = {
        "task": "facade", "split": split, "n_images": len(ds), "checkpoint": str(checkpoint),
        "checkpoint_meta": {k: meta.get(k) for k in ("data_source", "mode", "epoch", "seed", "hparams")},
        "l1": _stats(l1), "psnr_db": _stats(ps), "ssim": _stats(ss),
        "lpips": _stats(torch.cat(lpips_v)) if lpips_v else {"status": "not computed", "reason": lp.reason},
        "averaging": "per image", "env": _env(),
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(checkpoint).stem
    Image.fromarray(np.vstack(rows)).save(METRICS_DIR / f"{stem}_{split}_grid.png")
    (METRICS_DIR / f"{stem}_{split}.json").write_text(json.dumps(result, indent=2, default=str))
    return result


@torch.no_grad()
def evaluate_floorplan(checkpoint: Path, batch_size: int = 2, split: str = "test") -> Dict[str, Any]:
    from src.models.unet_plus_plus import UNetPlusPlusMultiTask
    from src.training.train_floorplan import NPZFloorplanDataset, split_files

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNetPlusPlusMultiTask(FLOORPLAN_CFG.in_channels, FLOORPLAN_CFG.num_structure_classes,
                                  FLOORPLAN_CFG.num_room_classes)
    meta = load_weights(model, checkpoint, require_real_data=True)
    model.to(device).eval()

    # Evaluate at the resolution the model was trained at (recorded in the checkpoint).
    image_size = int((meta.get("hparams") or {}).get("input_size", 512))
    ds = NPZFloorplanDataset(split_files(split), augment=False, image_size=image_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    cm_s = ConfusionMatrix(len(STRUCTURE_CLASSES), 255, device)
    cm_r = ConfusionMatrix(len(ROOM_CLASSES), 255, device)
    for images, s_t, r_t in loader:
        s_logits, r_logits = model(images.to(device))
        cm_s.update(s_logits.argmax(1), s_t.to(device))
        cm_r.update(r_logits.argmax(1), r_t.to(device))
    s, r = cm_s.compute(STRUCTURE_CLASSES), cm_r.compute(ROOM_CLASSES)
    walls = [v for v in (s["per_class_iou"]["exterior_wall"], s["per_class_iou"]["interior_wall"]) if v is not None]
    result = {
        "task": "floorplan", "split": split, "n_images": len(ds), "checkpoint": str(checkpoint),
        "checkpoint_meta": {k: meta.get(k) for k in ("data_source", "epoch", "seed", "hparams")},
        "structure": s, "rooms": r,
        "merged_wall_iou": float(np.mean(walls)) if walls else None,
        "eval_resolution": image_size,
        "note": "room pixels labelled 255 (unknown type) are excluded from all metrics",
        "env": _env(),
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / f"{Path(checkpoint).stem}_{split}.json").write_text(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Final test-split evaluation (real checkpoints only)")
    ap.add_argument("task", choices=["facade", "floorplan"])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--batch-size", type=int, default=None)
    a = ap.parse_args()
    if a.task == "facade":
        res = evaluate_facade(a.checkpoint, a.batch_size or 8, a.split)
        print(json.dumps({k: res[k] for k in ("n_images", "l1", "psnr_db", "ssim", "lpips")}, indent=2))
    else:
        res = evaluate_floorplan(a.checkpoint, a.batch_size or 2, a.split)
        print(json.dumps({"n_images": res["n_images"], "merged_wall_iou": res["merged_wall_iou"],
                          "struct_miou": res["structure"]["miou"], "room_miou": res["rooms"]["miou"],
                          "room_pixel_acc": res["rooms"]["pixel_acc"]}, indent=2))
