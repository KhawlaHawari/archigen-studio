"""
End-to-end GPU latency of the studio pipeline (run on Kaggle T4; mirrors the original project's breakdown).

Floor plan (20 real test plans, single pass): preprocessing (letterbox + tensor), UNet++ inference,
vectorization to the 3D layout (CPU), total. Facade: Pix2Pix 256 px, and optionally the fine-tuned
ControlNet (25 steps, 512 px). 5 warm-up runs are excluded; CUDA is synchronised around each timing.

    python -m src.mlops.benchmark_gpu
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from src.config import FACADE_CFG, FLOORPLAN_CFG, MODELS_DIR
from src.data.svg_preprocessor import isotropic_letterbox
from src.geometry.layout import layout_from_probabilities
from src.training.checkpoints import load_weights

OUT = MODELS_DIR / "metrics" / "latency_gpu.json"


def _stats(ms):
    a = np.array(ms)
    return {"n": len(a), "mean_ms": round(float(a.mean()), 1), "p50_ms": round(float(np.percentile(a, 50)), 1),
            "p95_ms": round(float(np.percentile(a, 95)), 1)}


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def benchmark(n_plans: int = 20, warmup: int = 5, controlnet: bool = True) -> dict:
    from src.models.pix2pix_cgan import FacadeGenerator
    from src.models.unet_plus_plus import UNetPlusPlusMultiTask
    from src.training.train_floorplan import split_files

    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = {"device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu", "torch": torch.__version__}

    model = UNetPlusPlusMultiTask(FLOORPLAN_CFG.in_channels, FLOORPLAN_CFG.num_structure_classes,
                                  FLOORPLAN_CFG.num_room_classes)
    load_weights(model, FLOORPLAN_CFG.checkpoint_path, require_real_data=True)
    model.to(device).eval()
    pre, inf, vec, tot = [], [], [], []
    files = split_files("test")[: n_plans + warmup]
    for k, path in enumerate(files):
        rgb = np.load(path)["image"]
        _sync(); t0 = time.perf_counter()
        boxed, _ = isotropic_letterbox(rgb, target_size=(512, 512), is_mask=False, fill_value=255)
        x = torch.from_numpy(boxed.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
        _sync(); t1 = time.perf_counter()
        s, r = model(x)
        prob_s, prob_r = s.softmax(1)[0].cpu().numpy(), r.softmax(1)[0].cpu().numpy()
        _sync(); t2 = time.perf_counter()
        layout_from_probabilities(prob_s, prob_r, boxed)
        t3 = time.perf_counter()
        if k >= warmup:
            pre.append((t1 - t0) * 1e3); inf.append((t2 - t1) * 1e3); vec.append((t3 - t2) * 1e3); tot.append((t3 - t0) * 1e3)
    result["floorplan_pipeline"] = {"preprocess": _stats(pre), "unetpp_inference": _stats(inf),
                                    "vectorize_to_3d_layout_cpu": _stats(vec), "total": _stats(tot)}

    gen = FacadeGenerator(FACADE_CFG.in_channels, FACADE_CFG.out_channels, FACADE_CFG.ngf)
    load_weights(gen, FACADE_CFG.checkpoint_path, require_real_data=True)
    gen.to(device).eval()
    x = torch.rand(1, 3, 256, 256, device=device) * 2 - 1
    times = []
    for k in range(warmup + 30):
        _sync(); t0 = time.perf_counter(); gen(x); _sync()
        if k >= warmup:
            times.append((time.perf_counter() - t0) * 1e3)
    result["facade_pix2pix_256"] = _stats(times)

    cn = MODELS_DIR / "controlnet_facade"
    if controlnet and device == "cuda" and cn.exists():
        from src.evaluation.evaluate_controlnet import load_pipeline, make_generate
        generate = make_generate(load_pipeline(str(cn)))
        times = []
        for k in range(2 + 10):
            t0 = time.perf_counter(); generate(x); _sync()
            if k >= 2:
                times.append((time.perf_counter() - t0) * 1e3)
        result["facade_controlnet_512_25steps"] = _stats(times)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    benchmark()
