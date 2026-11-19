"""
Fairness / bias analysis ("gestion des biais et fairness", MLOps level 3).

ArchiGen has no people in its data, so fairness means: does the model serve every KIND of plan
equally well, or does it quietly fail on some groups? Groups (all known before prediction):
  * drawing style : CubiCasa source folder (colorful / high_quality / high_quality_architectural)
  * plan size     : building footprint, terciles (small / medium / large)
  * complexity    : number of room types in the plan, terciles (simple / medium / complex)

For each group: metrics with 95 % bootstrap CIs; for each grouping: the worst-best GAP (points) and
the worst/best RATIO (disparity ratio, 1.0 = perfectly even). A grouping is FLAGGED when the gap
exceeds 10 points and the CIs of the best and worst group do not overlap (a real, not a noise, gap).

Input:  models/metrics/floorplan_unetpp_best_test_perplan.npz (src.evaluation.per_plan_eval)
Output: models/metrics/floorplan_unetpp_best_test_fairness.json
    python -m src.evaluation.fairness
"""
from __future__ import annotations

import json
from typing import Dict

import numpy as np

from src.evaluation.bootstrap import METRICS, PERPLAN, segmentation_ci

OUT = METRICS / "floorplan_unetpp_best_test_fairness.json"
KEY_METRICS = ("wall_iou", "door_iou", "window_iou", "room_miou_11", "room_pixel_acc")
GAP_FLAG_POINTS = 10.0


def terciles(values: np.ndarray, names=("small", "medium", "large")) -> np.ndarray:
    q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
    return np.where(values <= q1, names[0], np.where(values <= q2, names[1], names[2]))


def analyse(cm_s: np.ndarray, cm_r: np.ndarray, groups: np.ndarray, b: int = 1000) -> Dict:
    per_group = {}
    for g in sorted(set(groups.tolist())):
        idx = np.where(groups == g)[0]
        per_group[g] = {"n_plans": int(len(idx)), **segmentation_ci(cm_s[idx], cm_r[idx], b)}
    disparity = {}
    for m in KEY_METRICS:
        vals = {g: v[m]["value"] for g, v in per_group.items() if v["n_plans"] >= 10}
        if len(vals) < 2:
            continue
        best, worst = max(vals, key=vals.get), min(vals, key=vals.get)
        gap = 100 * (vals[best] - vals[worst])
        overlap = per_group[worst][m]["ci95_high"] >= per_group[best][m]["ci95_low"]
        disparity[m] = {"best_group": best, "worst_group": worst, "gap_points": round(gap, 2),
                        "ratio_worst_over_best": round(vals[worst] / vals[best], 3) if vals[best] else None,
                        "cis_overlap": bool(overlap), "flagged": bool(gap > GAP_FLAG_POINTS and not overlap)}
    return {"groups": per_group, "disparity": disparity}


def run() -> Dict:
    d = np.load(PERPLAN)
    cm_s, cm_r = d["cm_structure"], d["cm_rooms"]
    groupings = {"drawing_style": d["styles"].astype(str),
                 "plan_size": terciles(d["sizes"].astype(float)),
                 "complexity": terciles(d["n_room_types"].astype(float), ("simple", "medium", "complex"))}
    result = {"n_plans": int(len(cm_s)), "flag_rule": f"gap > {GAP_FLAG_POINTS} points and non-overlapping 95% CIs",
              "groupings": {name: analyse(cm_s, cm_r, g) for name, g in groupings.items()}}
    result["flags"] = [f"{name}: {m} ({v['worst_group']} {v['gap_points']} pts below {v['best_group']})"
                       for name, a in result["groupings"].items() for m, v in a["disparity"].items() if v["flagged"]]
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    r = run()
    for name, a in r["groupings"].items():
        print(f"\n== {name}")
        for g, v in a["groups"].items():
            print(f"  {g:<28} n={v['n_plans']:<4} " + "  ".join(f"{m} {100 * v[m]['value']:.1f}" for m in KEY_METRICS))
        for m, v in a["disparity"].items():
            print(f"  gap {m:<16} {v['gap_points']:6.2f} pts  ratio {v['ratio_worst_over_best']}  {'FLAG' if v['flagged'] else ''}")
    print("\nFlags:", r["flags"] or "none")
