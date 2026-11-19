"""
Out-of-distribution robustness: floor plans that are NOT CubiCasa5K (no ground truth).

Each plan goes through the full studio pipeline (letterbox -> UNet++ -> layout vectorizer, scale
estimated from door widths). Without labels we report the geometric usability heuristic of the
original ArchiGen study, so the numbers are directly comparable:

    usable  = rooms >= 3 and total room area >= 25 m2
    empty   = 0 rooms detected
    shell   = closed (watertight) building envelope

Sets (data/raw): dev_external_plans (20 real plans), real_external_plans/fresh_defense_eval
(12 real plans), real_external_plans/heldout_modern (8 modern plans), external_test_plans
(10 synthetic style stress tests).

    python -m src.evaluation.evaluate_external --checkpoint models/checkpoints/floorplan_unetpp_best.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image

from src.config import DATA_DIR, FLOORPLAN_CFG, MODELS_DIR
from src.data.svg_preprocessor import isotropic_letterbox
from src.geometry.layout import layout_from_probabilities
from src.training.checkpoints import load_weights

METRICS_DIR = MODELS_DIR / "metrics"
SETS = {
    "dev20": DATA_DIR / "raw" / "dev_external_plans",
    "fresh12": DATA_DIR / "raw" / "real_external_plans" / "fresh_defense_eval",
    "modern8": DATA_DIR / "raw" / "real_external_plans" / "heldout_modern",
    "synthetic10": DATA_DIR / "raw" / "external_test_plans",
}
MIN_ROOMS, MIN_AREA_M2 = 3, 25.0


def plan_files(folder: Path) -> List[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))


@torch.no_grad()
def evaluate_external(checkpoint: Path, sets: List[str]) -> Dict[str, Any]:
    from src.models.unet_plus_plus import UNetPlusPlusMultiTask

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNetPlusPlusMultiTask(FLOORPLAN_CFG.in_channels, FLOORPLAN_CFG.num_structure_classes,
                                  FLOORPLAN_CFG.num_room_classes)
    meta = load_weights(model, checkpoint, require_real_data=True)
    model.to(device).eval()

    result: Dict[str, Any] = {"checkpoint": str(checkpoint), "checkpoint_epoch": meta.get("epoch"),
                              "criteria": {"usable": f"rooms >= {MIN_ROOMS} and area >= {MIN_AREA_M2} m2",
                                           "scale": "estimated from door widths (no override)"},
                              "sets": {}}
    t0 = time.time()
    for name in sets:
        folder = SETS[name]
        if not folder.is_dir():
            print(f"[skip] {name}: {folder} not found")
            continue
        rows = []
        for path in plan_files(folder):
            rgb = np.array(Image.open(path).convert("RGB"))
            boxed, _ = isotropic_letterbox(rgb, target_size=(512, 512), is_mask=False, fill_value=255)
            x = torch.from_numpy(boxed.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
            s, r = model(x)
            layout = layout_from_probabilities(s.softmax(1)[0].cpu().numpy(), r.softmax(1)[0].cpu().numpy(), boxed)
            rooms = layout.get("rooms", [])
            area = round(sum(room.get("surface_m2", 0.0) for room in rooms), 2)
            rows.append({"plan": path.stem, "rooms": len(rooms), "area_m2": area,
                         "doors": len(layout.get("doors", [])), "windows": len(layout.get("windows", [])),
                         "shell_valid": bool(layout.get("metadata", {}).get("is_building_shell_valid")),
                         "usable": len(rooms) >= MIN_ROOMS and area >= MIN_AREA_M2})
        n = len(rows)
        result["sets"][name] = {
            "n_plans": n,
            "usable_rate": round(sum(r["usable"] for r in rows) / n, 4) if n else None,
            "empty_rate": round(sum(r["rooms"] == 0 for r in rows) / n, 4) if n else None,
            "shell_valid_rate": round(sum(r["shell_valid"] for r in rows) / n, 4) if n else None,
            "per_plan": rows,
        }
        s = result["sets"][name]
        print(f"{name:<12} n={n:<3} usable={s['usable_rate']:.0%}  empty={s['empty_rate']:.0%}  "
              f"shell={s['shell_valid_rate']:.0%}", flush=True)
    result["elapsed_s"] = round(time.time() - t0, 1)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out = METRICS_DIR / f"{Path(checkpoint).stem}_external.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"Saved {out}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Out-of-distribution floor-plan robustness (no ground truth)")
    ap.add_argument("--checkpoint", type=Path, default=FLOORPLAN_CFG.checkpoint_path)
    ap.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
    a = ap.parse_args()
    evaluate_external(a.checkpoint, a.sets)
