"""
95 % bootstrap confidence intervals for the test metrics (stands in for cross-validation).

K-fold cross-validation would mean retraining a deep model k times for hours on GPU, and CubiCasa5K /
CMP have official fixed splits. The question cross-validation answers here -- how much would the score
move with a different sample of plans? -- is answered by resampling the 400 TEST plans with replacement
(B = 2000) and recomputing each metric from the summed per-plan confusion matrices (percentile method).

Inputs:  models/metrics/floorplan_unetpp_best_test_perplan.npz  (src.evaluation.per_plan_eval)
         models/metrics/floorplan_unetpp_best_test_layout.json  (per-plan 3D layout results)
Output:  models/metrics/floorplan_unetpp_best_test_bootstrap.json

    python -m src.evaluation.bootstrap
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict

import numpy as np

from src.config import MODELS_DIR

METRICS = MODELS_DIR / "metrics"
PERPLAN = METRICS / "floorplan_unetpp_best_test_perplan.npz"
LAYOUT = METRICS / "floorplan_unetpp_best_test_layout.json"
OUT = METRICS / "floorplan_unetpp_best_test_bootstrap.json"
B, SEED = 2000, 42


def iou(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm).astype(np.float64)
    union = cm.sum(0) + cm.sum(1) - tp
    return np.where(union > 0, tp / np.maximum(union, 1), np.nan)


def seg_metrics(cm_s: np.ndarray, cm_r: np.ndarray) -> Dict[str, float]:
    """Same definitions as src.evaluation.evaluate (rooms: background excluded from the 11-room mIoU)."""
    si, ri = iou(cm_s), iou(cm_r)
    present_s, present_r = cm_s.sum(1) > 0, cm_r.sum(1) > 0
    rooms = [ri[c] for c in range(1, 12) if present_r[c]]
    return {"wall_iou": float(np.nanmean(si[[1, 2]])), "exterior_wall_iou": float(si[1]), "interior_wall_iou": float(si[2]),
            "door_iou": float(si[3]), "window_iou": float(si[4]), "structure_miou": float(np.nanmean(si[present_s])),
            "room_miou_11": float(np.nanmean(rooms)) if rooms else float("nan"),
            "room_pixel_acc": float(np.trace(cm_r) / max(cm_r.sum(), 1))}


def bootstrap(n: int, stat: Callable[[np.ndarray], Dict[str, float]], b: int = B, seed: int = SEED) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    point = stat(np.arange(n))
    draws = {k: [] for k in point}
    for _ in range(b):
        s = stat(rng.integers(0, n, n))
        for k, v in s.items():
            draws[k].append(v)
    return {k: {"value": round(point[k], 4), "ci95_low": round(float(np.nanpercentile(v, 2.5)), 4),
                "ci95_high": round(float(np.nanpercentile(v, 97.5)), 4)} for k, v in draws.items()}


def segmentation_ci(cm_s: np.ndarray, cm_r: np.ndarray, b: int = B) -> Dict[str, Dict[str, float]]:
    return bootstrap(len(cm_s), lambda idx: seg_metrics(cm_s[idx].sum(0), cm_r[idx].sum(0)), b)


def layout_ci(rows, b: int = B) -> Dict[str, Dict[str, float]]:
    shell = np.array([r["shell_valid"] for r in rows], float)
    foot = np.array([r["footprint_iou"] for r in rows], float)
    room_err = np.array([abs(r["rooms_pred"] - r["rooms_gt"]) for r in rows], float)
    return bootstrap(len(rows), lambda i: {"watertight_shell_rate": shell[i].mean(), "footprint_iou_mean": foot[i].mean(),
                                           "room_count_mae": room_err[i].mean()}, b)


def run() -> Dict:
    d = np.load(PERPLAN)
    result = {"method": f"percentile bootstrap over test plans, B={B}, seed={SEED}", "n_plans": int(len(d["names"])),
              "segmentation": segmentation_ci(d["cm_structure"], d["cm_rooms"])}
    if LAYOUT.exists():
        rows = json.loads(LAYOUT.read_text(encoding="utf-8")).get("per_plan", [])
        if rows:
            result["layout_3d"] = layout_ci(rows)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    r = run()
    for group in ("segmentation", "layout_3d"):
        for k, v in r.get(group, {}).items():
            print(f"{k:<24} {v['value']:.4f}   95% CI [{v['ci95_low']:.4f}, {v['ci95_high']:.4f}]")
