"""
Object-level and 3D-layout evaluation of the floor-plan model (test split, run once).

Pixel IoU (src.evaluation.evaluate) says how well masks overlap. This script measures what the
3D studio actually uses:

* Door / window detection: each connected component is one object. A predicted object is a true
  positive when its centroid lies within R px of an unmatched ground-truth centroid (greedy
  nearest matching). Precision, recall, F1 at R = 12 and 24 px, and the centroid error of matches.
* 3D layout: the predictions go through the same vectorizer as the API. We report the share of
  plans with a closed (watertight) building shell, the footprint IoU of the vectorized envelope
  against the ground-truth footprint, and the room / door count errors. The room reference is the
  layout vectorized from the ground-truth masks (connected components merge touching rooms of the
  same type, so they cannot be counted directly); its shell rate is reported as an upper bound.

    python -m src.evaluation.evaluate_layout --checkpoint models/checkpoints/floorplan_unetpp_best.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from scipy import ndimage

from src.config import FLOORPLAN_CFG, MODELS_DIR
from src.geometry.layout import layout_from_probabilities
from src.training.checkpoints import load_weights

METRICS_DIR = MODELS_DIR / "metrics"
RADII = (12, 24)
MIN_OBJECT_PX = 10       # ignore specks smaller than this (both prediction and ground truth)


def centroids(mask: np.ndarray, min_px: int = MIN_OBJECT_PX) -> np.ndarray:
    labels, n = ndimage.label(mask)
    if n == 0:
        return np.zeros((0, 2))
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    keep = [i + 1 for i, s in enumerate(sizes) if s >= min_px]
    if not keep:
        return np.zeros((0, 2))
    return np.array(ndimage.center_of_mass(mask, labels, keep))


def match(pred: np.ndarray, gt: np.ndarray, radius: float) -> Tuple[int, List[float]]:
    """Greedy one-to-one matching by increasing distance. Returns (true positives, matched distances)."""
    if len(pred) == 0 or len(gt) == 0:
        return 0, []
    d = np.linalg.norm(pred[:, None, :] - gt[None, :, :], axis=2)
    pairs = sorted(zip(*np.where(d <= radius)), key=lambda ij: d[ij])
    used_p, used_g, dists = set(), set(), []
    for i, j in pairs:
        if i not in used_p and j not in used_g:
            used_p.add(i); used_g.add(j); dists.append(float(d[i, j]))
    return len(dists), dists


def prf(tp: int, n_pred: int, n_gt: int) -> Dict[str, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gt if n_gt else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r) if p + r else 0.0, 4),
            "tp": tp, "fp": n_pred - tp, "fn": n_gt - tp}


def gt_footprint(struct_gt: np.ndarray, room_gt: np.ndarray) -> np.ndarray:
    built = (struct_gt > 0) | ((room_gt > 0) & (room_gt != 255))
    return ndimage.binary_fill_holes(built)


def envelope_mask(layout: Dict[str, Any], shape: Tuple[int, int]) -> np.ndarray:
    scale = layout.get("metadata", {}).get("scale_meters_per_pixel") or 0.03
    mask = np.zeros(shape, dtype=np.uint8)
    for ring in layout.get("envelope", []):
        pts = np.asarray(ring, dtype=np.float64)
        if pts.ndim == 2 and len(pts) >= 3:
            cv2.fillPoly(mask, [np.rint(pts / scale).astype(np.int32)], 1)
    return mask.astype(bool)


def one_hot(mask: np.ndarray, n: int) -> np.ndarray:
    """Ground-truth mask as 'probabilities' so it can go through the same vectorizer (255 -> background)."""
    m = np.where(mask == 255, 0, mask)
    return (np.arange(n)[:, None, None] == m[None]).astype(np.float32)


def room_type_pairs(layout: Dict[str, Any], room_gt: np.ndarray) -> List[Tuple[int, int]]:
    """(true type, predicted type) for every room polygon of the 3D layout.

    The true type is the majority ground-truth room label inside the polygon (unknown 255 and
    background ignored); polygons with < 50 % known room pixels are skipped (not a real room).
    """
    from src.geometry.layout_vectorizer import ROOM_NAMES
    ids = {name: idx for idx, name in ROOM_NAMES.items()}
    scale = layout.get("metadata", {}).get("scale_meters_per_pixel") or 0.03
    pairs = []
    for room in layout.get("rooms", []):
        pred_id = ids.get(room.get("room_type"))
        poly = np.asarray(room.get("polygon_meters") or [], dtype=np.float64)
        if pred_id is None or poly.ndim != 2 or len(poly) < 3:
            continue
        region = np.zeros(room_gt.shape, np.uint8)
        cv2.fillPoly(region, [np.rint(poly / scale).astype(np.int32)], 1)
        labels = room_gt[region > 0]
        known = labels[(labels != 0) & (labels != 255)]
        if labels.size == 0 or known.size < 0.5 * labels.size:
            continue
        pairs.append((int(np.bincount(known).argmax()), int(pred_id)))
    return pairs


def room_type_summary(pairs: List[Tuple[int, int]]) -> Dict[str, Any]:
    """Room-level classification: share of 3D rooms whose type is right, overall and per true type."""
    from src.geometry.layout_vectorizer import ROOM_NAMES
    if not pairs:
        return {"n_rooms": 0}
    arr = np.array(pairs)
    per_class = {}
    for cls in sorted(set(arr[:, 0].tolist())):
        sel = arr[:, 0] == cls
        per_class[ROOM_NAMES[cls]] = {"n": int(sel.sum()), "accuracy": round(float((arr[sel, 1] == cls).mean()), 4)}
    accs = [v["accuracy"] for v in per_class.values()]
    return {"n_rooms": len(arr), "accuracy": round(float((arr[:, 0] == arr[:, 1]).mean()), 4),
            "balanced_accuracy": round(float(np.mean(accs)), 4), "per_type": per_class}


@torch.no_grad()
def evaluate_layout(checkpoint: Path, split: str = "test", limit: int | None = None) -> Dict[str, Any]:
    from src.models.unet_plus_plus import UNetPlusPlusMultiTask
    from src.training.train_floorplan import split_files

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNetPlusPlusMultiTask(FLOORPLAN_CFG.in_channels, FLOORPLAN_CFG.num_structure_classes,
                                  FLOORPLAN_CFG.num_room_classes)
    meta = load_weights(model, checkpoint, require_real_data=True)
    model.to(device).eval()

    files = split_files(split)[:limit] if limit else split_files(split)
    det = {name: {"pred": 0, "gt": 0, **{f"tp@{r}": 0 for r in RADII}, "dists": []} for name in ("door", "window")}
    shells, gt_shells, foot_ious, room_err, door_err, rows, room_types = [], [], [], [], [], [], []
    t0 = time.time()
    for k, path in enumerate(files):
        data = np.load(path)
        image, struct_gt, room_gt = data["image"], data["structure_mask"], data["room_mask"]
        x = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
        s_logits, r_logits = model(x)
        prob_s = s_logits.softmax(1)[0].cpu().numpy()
        prob_r = r_logits.softmax(1)[0].cpu().numpy()
        struct_pred = prob_s.argmax(0)

        for name, cls in (("door", 3), ("window", 4)):
            p, g = centroids(struct_pred == cls), centroids(struct_gt == cls)
            det[name]["pred"] += len(p); det[name]["gt"] += len(g)
            for r in RADII:
                tp, dists = match(p, g, r)
                det[name][f"tp@{r}"] += tp
                if r == max(RADII):
                    det[name]["dists"].extend(dists)

        layout = layout_from_probabilities(prob_s, prob_r, image)
        shell_ok = bool(layout.get("metadata", {}).get("is_building_shell_valid"))
        shells.append(shell_ok)
        fp, fg = envelope_mask(layout, struct_gt.shape), gt_footprint(struct_gt, room_gt)
        union = np.logical_or(fp, fg).sum()
        foot_ious.append(float(np.logical_and(fp, fg).sum() / union) if union else 0.0)
        gt_layout = layout_from_probabilities(one_hot(struct_gt, prob_s.shape[0]), one_hot(room_gt, prob_r.shape[0]), image)
        gt_shells.append(bool(gt_layout.get("metadata", {}).get("is_building_shell_valid")))
        n_rooms_gt = len(gt_layout.get("rooms", []))
        n_doors_gt = len(centroids(struct_gt == 3))
        for true_id, pred_id in room_type_pairs(layout, room_gt):
            room_types.append((true_id, pred_id))
        room_err.append(len(layout.get("rooms", [])) - n_rooms_gt)
        door_err.append(len(layout.get("doors", [])) - n_doors_gt)
        rows.append({"file": Path(path).stem, "shell_valid": shell_ok, "footprint_iou": round(foot_ious[-1], 4),
                     "rooms_pred": len(layout.get("rooms", [])), "rooms_gt": n_rooms_gt,
                     "doors_pred": len(layout.get("doors", [])), "doors_gt": n_doors_gt})
        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/{len(files)} plans  ({(time.time() - t0) / (k + 1):.1f} s/plan)", flush=True)

    detection = {}
    for name, d in det.items():
        detection[name] = {f"radius_{r}px": prf(d[f"tp@{r}"], d["pred"], d["gt"]) for r in RADII}
        dist = np.array(d["dists"]) if d["dists"] else np.zeros(1)
        detection[name]["centroid_error_px"] = {"median": round(float(np.median(dist)), 2),
                                                "mean": round(float(np.mean(dist)), 2)}
    room_err, door_err = np.array(room_err), np.array(door_err)
    result = {
        "task": "floorplan_layout", "split": split, "n_plans": len(files), "checkpoint": str(checkpoint),
        "checkpoint_epoch": meta.get("epoch"),
        "detection": detection,
        "layout": {
            "watertight_shell_rate": round(float(np.mean(shells)), 4),
            "watertight_shell_rate_from_gt_masks": round(float(np.mean(gt_shells)), 4),
            "footprint_iou_mean": round(float(np.mean(foot_ious)), 4),
            "footprint_iou_median": round(float(np.median(foot_ious)), 4),
            "room_count_mae": round(float(np.mean(np.abs(room_err))), 2),
            "room_count_bias": round(float(np.mean(room_err)), 2),
            "door_count_mae": round(float(np.mean(np.abs(door_err))), 2),
            "door_count_bias": round(float(np.mean(door_err)), 2),
        },
        "room_types": room_type_summary(room_types),
        "protocol": {
            "matching": "greedy nearest centroid, one-to-one", "radii_px": list(RADII),
            "min_object_px": MIN_OBJECT_PX, "room_reference": "layout vectorized from the ground-truth masks (same vectorizer)", "tta": None,
            "scale": "estimated by the vectorizer from door widths (no override)",
        },
        "per_plan": rows,
        "elapsed_s": round(time.time() - t0, 1),
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out = METRICS_DIR / f"{Path(checkpoint).stem}_{split}_layout.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"Saved {out}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Door/window detection and 3D layout evaluation")
    ap.add_argument("--checkpoint", type=Path, default=FLOORPLAN_CFG.checkpoint_path)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N plans (smoke test)")
    a = ap.parse_args()
    res = evaluate_layout(a.checkpoint, a.split, a.limit)
    print(json.dumps({"detection": res["detection"], "layout": res["layout"]}, indent=2))
