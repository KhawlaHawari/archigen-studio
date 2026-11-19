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
    from src.models.pix2pix_cgan import FacadeGenerator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = FacadeGenerator(FACADE_CFG.in_channels, FACADE_CFG.out_channels, FACADE_CFG.ngf)
    meta = load_weights(gen, checkpoint, require_real_data=True)
    gen.to(device).eval()
    return evaluate_facade_generator(lambda masks: gen(masks), Path(checkpoint).stem, batch_size, split,
                                     {"checkpoint": str(checkpoint),
                                      "checkpoint_meta": {k: meta.get(k) for k in ("data_source", "mode", "epoch", "seed", "hparams")}})


@torch.no_grad()
def evaluate_facade_generator(generate, stem: str, batch_size: int = 8, split: str = "test",
                              extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Same protocol for every facade generator. `generate` maps [-1, 1] label maps (N, 3, 256, 256)
    on the evaluation device to [-1, 1] images of the same size.

    Realism: KID x1000, LPIPS (AlexNet, SqueezeNet). Pixel fidelity: L1, PSNR, SSIM.
    Layout fidelity (FCN-score): the facade parser (if trained) parses each generated image and the
    result is compared with the input label map; the same parser on the REAL photos gives the upper bound.
    """
    from src.data.facade_dataset import CMPFacadesDataset, to_uint8
    from src.models.facade_parser import CMP_CLASSES, labels_from_tensor, parse
    from src.training.train_facade_parser import load_parser

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = CMPFacadesDataset(split, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    lp = OptionalLPIPS(net="alex", device=device)
    lp_sq = OptionalLPIPS(net="squeeze", device=device)   # same backbone as the historical benchmark
    kid = _optional_kid(device)
    parser = load_parser(device=device)
    cm_fake = ConfusionMatrix(len(CMP_CLASSES), 255, device) if parser is not None else None
    cm_real = ConfusionMatrix(len(CMP_CLASSES), 255, device) if parser is not None else None
    l1s, psnrs, ssims, lpips_v, lpips_sq, rows = [], [], [], [], [], []
    for masks, reals in loader:
        masks, reals = masks.to(device), reals.to(device)
        fakes = generate(masks).clamp(-1, 1)
        l1s.append(l1_per_image(fakes, reals).cpu())
        psnrs.append(psnr_per_image(fakes, reals).cpu())
        ssims.append(ssim_per_image(fakes, reals).cpu())
        v = lp(fakes, reals)
        if v is not None:
            lpips_v.append(v.cpu())
        v = lp_sq(fakes, reals)
        if v is not None:
            lpips_sq.append(v.cpu())
        if kid is not None:
            to_u8 = lambda t: ((t.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
            kid.update(to_u8(reals), real=True)
            kid.update(to_u8(fakes), real=False)
        if parser is not None:
            target = labels_from_tensor(masks).to(device)
            cm_fake.update(parse(parser, fakes), target)
            cm_real.update(parse(parser, reals), target)
        if len(rows) < 6:
            for m, f, r in zip(masks, fakes, reals):
                if len(rows) < 6:
                    rows.append(np.hstack([to_uint8(m), to_uint8(f), to_uint8(r)]))

    l1, ps, ss = torch.cat(l1s), torch.cat(psnrs), torch.cat(ssims)
    def fcn(cm):
        if cm is None:
            return {"status": "not computed", "reason": "no facade parser (python -m src.training.train_facade_parser)"}
        r = cm.compute(CMP_CLASSES)
        return {"pixel_acc": r["pixel_acc"], "miou": r["miou"], "per_class_iou": r["per_class_iou"]}
    result = {
        "task": "facade", "split": split, "n_images": len(ds), **(extra or {}),
        "l1": _stats(l1), "psnr_db": _stats(ps), "ssim": _stats(ss),
        "lpips": _stats(torch.cat(lpips_v)) if lpips_v else {"status": "not computed", "reason": lp.reason},
        "lpips_squeeze": _stats(torch.cat(lpips_sq)) if lpips_sq else {"status": "not computed", "reason": lp_sq.reason},
        "kid_x1000": _kid_result(kid),
        "fcn_score": fcn(cm_fake),
        "fcn_score_real_photos": fcn(cm_real),
        "averaging": "per image", "env": _env(),
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.vstack(rows)).save(METRICS_DIR / f"{stem}_{split}_grid.png")
    (METRICS_DIR / f"{stem}_{split}.json").write_text(json.dumps(result, indent=2, default=str))
    return result


@torch.no_grad()
def evaluate_floorplan(checkpoint: Path, batch_size: int = 2, split: str = "test", tta: bool = False,
                       styled: bool = False) -> Dict[str, Any]:
    """styled=True re-draws every plan in another drawing style (fixed seed, labels unchanged):
    a labelled robustness test for black-ink CAD, blueprints, dark mode, scans... (see style_augment)."""
    from src.models.unet_plus_plus import UNetPlusPlusMultiTask
    from src.training.train_floorplan import STYLE_VAL_SEED, NPZFloorplanDataset, split_files

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNetPlusPlusMultiTask(FLOORPLAN_CFG.in_channels, FLOORPLAN_CFG.num_structure_classes,
                                  FLOORPLAN_CFG.num_room_classes)
    meta = load_weights(model, checkpoint, require_real_data=True)
    model.to(device).eval()

    # Evaluate at the resolution the model was trained at (recorded in the checkpoint).
    image_size = int((meta.get("hparams") or {}).get("input_size", 512))
    # Test uses a different style seed from validation, so model selection never saw these renderings.
    style_seed = (STYLE_VAL_SEED + 100_000 if split == "test" else STYLE_VAL_SEED) if styled else None
    ds = NPZFloorplanDataset(split_files(split), augment=False, image_size=image_size, style_seed=style_seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    cm_s = ConfusionMatrix(len(STRUCTURE_CLASSES), 255, device)
    cm_r = ConfusionMatrix(len(ROOM_CLASSES), 255, device)
    for images, s_t, r_t in loader:
        images = images.to(device)
        s_logits, r_logits = _predict_tta(model, images) if tta else model(images)
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
        "tta": "4-fold (identity, h-flip, v-flip, hv-flip; mean of softmax)" if tta else None,
        "styled": f"every plan re-drawn by src.data.style_augment (seed {style_seed} + index)" if styled else None,
        "note": "room pixels labelled 255 (unknown type) are excluded from all metrics",
        "env": _env(),
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = ("_styled" if styled else "") + ("_tta" if tta else "")
    (METRICS_DIR / f"{Path(checkpoint).stem}_{split}{suffix}.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def _predict_tta(model, images: torch.Tensor):
    """4-fold flip TTA: predict on flipped inputs, un-flip, average class probabilities."""
    s_acc, r_acc = 0.0, 0.0
    for dims in ((), (-1,), (-2,), (-2, -1)):
        x = images.flip(dims) if dims else images
        s, r = model(x)
        s, r = s.softmax(1), r.softmax(1)
        if dims:
            s, r = s.flip(dims), r.flip(dims)
        s_acc, r_acc = s_acc + s, r_acc + r
    return s_acc / 4, r_acc / 4


def _optional_kid(device):
    """KID (x1000) with torchmetrics, 100 subsets of 50 images as in the historical benchmark."""
    try:
        from torchmetrics.image.kid import KernelInceptionDistance  # needs torch-fidelity
        torch.manual_seed(42)
        return KernelInceptionDistance(subset_size=50, subsets=100).to(device)
    except Exception as exc:  # not installed / weights unavailable offline
        _optional_kid.reason = f"{type(exc).__name__}: {exc}"
        return None


def _kid_result(kid) -> Dict[str, Any]:
    if kid is None:
        return {"status": "not computed", "reason": getattr(_optional_kid, "reason", "unavailable")}
    try:
        mean, std = kid.compute()
    except ValueError as exc:            # fewer images than the subset size (smoke tests)
        return {"status": "not computed", "reason": str(exc)}
    return {"mean": float(mean) * 1000, "std": float(std) * 1000, "subset_size": 50, "subsets": 100}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Final test-split evaluation (real checkpoints only)")
    ap.add_argument("task", choices=["facade", "floorplan"])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--tta", action="store_true", help="Floorplan only: 4-fold flip test-time augmentation")
    ap.add_argument("--styled", action="store_true", help="Floorplan only: re-draw each plan in another style")
    a = ap.parse_args()
    if a.task == "facade":
        res = evaluate_facade(a.checkpoint, a.batch_size or 8, a.split)
        print(json.dumps({k: res[k] for k in ("n_images", "l1", "psnr_db", "ssim", "lpips", "lpips_squeeze", "kid_x1000",
                                              "fcn_score")},
                         indent=2))
    else:
        res = evaluate_floorplan(a.checkpoint, a.batch_size or 2, a.split, tta=a.tta, styled=a.styled)
        print(json.dumps({"n_images": res["n_images"], "merged_wall_iou": res["merged_wall_iou"],
                          "struct_miou": res["structure"]["miou"], "room_miou": res["rooms"]["miou"],
                          "room_pixel_acc": res["rooms"]["pixel_acc"]}, indent=2))
