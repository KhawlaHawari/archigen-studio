"""
Per-plan evaluation of the production floor-plan model (test split): one confusion matrix per plan.

Aggregated metrics (src.evaluation.evaluate) give one number per split; confidence intervals
(src.evaluation.bootstrap) and subgroup / fairness analysis (src.evaluation.fairness) need the
per-plan detail. Each plan stores its 5x5 structure and 12x12 room confusion matrices (room
pixels labelled 255 are ignored, as everywhere else), plus metadata for subgroups:
  style  : CubiCasa source folder from the file name (colorful / high_quality / high_quality_architectural)
  size   : building footprint in ground-truth pixels
  rooms  : number of ground-truth room types present

Uses the deployed ONNX model (same numbers as PyTorch, see the parity report).
    python -m src.evaluation.per_plan_eval
Output: models/metrics/floorplan_unetpp_best_test_perplan.npz
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy import ndimage

from src.config import FLOORPLAN_CFG, MODELS_DIR
from src.training.train_floorplan import split_files

OUT = MODELS_DIR / "metrics" / "floorplan_unetpp_best_test_perplan.npz"


def confusion(pred: np.ndarray, gt: np.ndarray, n: int, ignore: int = 255) -> np.ndarray:
    valid = gt != ignore
    return np.bincount(gt[valid].astype(np.int64) * n + pred[valid], minlength=n * n).reshape(n, n)


def style_of(name: str) -> str:
    m = re.match(r"(high_quality_architectural|high_quality|colorful)", name)
    return m.group(1) if m else "other"


def run(split: str = "test", limit: int | None = None) -> Path:
    sess = ort.InferenceSession(str(FLOORPLAN_CFG.onnx_path), providers=["CPUExecutionProvider"])
    files = split_files(split)[:limit] if limit else split_files(split)
    cms_s, cms_r, names, styles, sizes, n_room_types = [], [], [], [], [], []
    t0 = time.time()
    for k, f in enumerate(files):
        d = np.load(f)
        img, s_gt, r_gt = d["image"], d["structure_mask"], d["room_mask"]
        x = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        s_logits, r_logits = sess.run(None, {"image": x})[:2]
        cms_s.append(confusion(s_logits[0].argmax(0), s_gt, 5))
        cms_r.append(confusion(r_logits[0].argmax(0), r_gt, 12))
        names.append(f.stem)
        styles.append(style_of(f.stem))
        sizes.append(int(ndimage.binary_fill_holes((s_gt > 0) | ((r_gt > 0) & (r_gt != 255))).sum()))
        n_room_types.append(int(len(set(np.unique(r_gt).tolist()) - {0, 255})))
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(files)} plans ({(time.time() - t0) / (k + 1):.1f} s/plan)", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, names=np.array(names), styles=np.array(styles), sizes=np.array(sizes),
                        n_room_types=np.array(n_room_types), cm_structure=np.stack(cms_s), cm_rooms=np.stack(cms_r))
    print(f"Saved {OUT} ({len(files)} plans, {time.time() - t0:.0f} s)")
    return OUT


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    run(a.split, a.limit)
