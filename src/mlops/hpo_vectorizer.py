"""
Automatic hyperparameter optimisation (Optuna, TPE sampler) of the floor-plan -> 3D post-processing.

Retraining the networks needs GPU hours; the 6 hyperparameters of the layout vectorizer (minimum wall
area, polygon simplification, angle snapping, minimum room areas, door width used for the scale) were
hand-picked and directly shape the 3D house. Optuna tunes them on the VALIDATION split, every trial is
logged to MLflow, and the best setting is then measured ONCE on held-out TEST plans against the
defaults. It is installed for production (models/vectorizer_params.json, read by src/geometry/layout.py)
only if it beats the defaults on test.

Objective (higher is better), from src.evaluation.evaluate_layout:
    0.4 * footprint IoU  +  0.3 * watertight-shell rate  +  0.3 * room-type accuracy in the 3D layout

    python -m src.mlops.hpo_vectorizer --trials 25 --val-plans 30 --test-plans 60
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnxruntime as ort

from src.config import FLOORPLAN_CFG, MODELS_DIR
from src.evaluation.evaluate_layout import envelope_mask, gt_footprint, room_type_pairs
from src.geometry.layout import TUNED_PARAMS, layout_from_probabilities
from src.geometry.layout_vectorizer import FloorplanVectorizer
from src.training.train_floorplan import split_files

REPORT = MODELS_DIR / "metrics" / "vectorizer_hpo.json"
DEFAULTS = {"min_wall_area": 40, "polygon_epsilon_ratio": 0.015, "snap_angle_deg_tolerance": 12.0,
            "min_area_habitable_m2": 2.0, "min_area_service_m2": 0.8, "default_door_width_m": 0.85}


def predict(files: List[Path]):
    """Run the network ONCE per plan; trials only re-run the (cheap) vectorizer."""
    sess = ort.InferenceSession(str(FLOORPLAN_CFG.onnx_path), providers=["CPUExecutionProvider"])
    cache = []
    for f in files:
        d = np.load(f)
        x = (d["image"].astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        s, r = sess.run(None, {"image": x})[:2]
        e = np.exp(r[0] - r[0].max(0, keepdims=True))
        cache.append({"image": d["image"], "struct_argmax": s[0].argmax(0).astype(np.uint8),
                      "prob_r": (e / e.sum(0, keepdims=True)).astype(np.float16),
                      "struct_gt": d["structure_mask"], "room_gt": d["room_mask"]})
    return cache


def score(cache, params: Dict) -> Dict[str, float]:
    vec = FloorplanVectorizer(**params)
    foot, shell, pairs = [], [], []
    for c in cache:
        prob_s = (np.arange(5)[:, None, None] == c["struct_argmax"][None]).astype(np.float32)
        layout = layout_from_probabilities(prob_s, c["prob_r"].astype(np.float32), c["image"], vectorizer=vec)
        fp, fg = envelope_mask(layout, c["struct_gt"].shape), gt_footprint(c["struct_gt"], c["room_gt"])
        foot.append(float((fp & fg).sum() / max((fp | fg).sum(), 1)))
        shell.append(bool(layout.get("metadata", {}).get("is_building_shell_valid")))
        pairs += room_type_pairs(layout, c["room_gt"])
    room_acc = float(np.mean([t == p for t, p in pairs])) if pairs else 0.0
    m = {"footprint_iou": float(np.mean(foot)), "shell_rate": float(np.mean(shell)), "room_type_acc": room_acc}
    m["objective"] = 0.4 * m["footprint_iou"] + 0.3 * m["shell_rate"] + 0.3 * m["room_type_acc"]
    return m


def run(trials: int = 25, val_plans: int = 30, test_plans: int = 60, seed: int = 42) -> Dict:
    import mlflow
    import optuna

    rng = np.random.default_rng(seed)
    val_files = list(rng.choice(split_files("val"), val_plans, replace=False))
    test_files = list(rng.choice(split_files("test"), test_plans, replace=False))
    t0 = time.time()
    val_cache = predict(val_files)
    print(f"[HPO] cached predictions for {len(val_cache)} validation plans ({time.time() - t0:.0f} s)", flush=True)

    mlflow.set_tracking_uri(f"sqlite:///{Path(MODELS_DIR).parent / 'mlflow.db'}")
    mlflow.set_experiment("ArchiGen3D-Vectorizer-HPO")
    with mlflow.start_run(run_name=f"optuna_tpe_{trials}trials") as parent:
        mlflow.log_params({"trials": trials, "val_plans": val_plans, "test_plans": test_plans, "sampler": "TPE", "seed": seed})
        default_val = score(val_cache, DEFAULTS)

        def objective(trial):
            params = {
                "min_wall_area": trial.suggest_int("min_wall_area", 10, 120),
                "polygon_epsilon_ratio": trial.suggest_float("polygon_epsilon_ratio", 0.005, 0.04),
                "snap_angle_deg_tolerance": trial.suggest_float("snap_angle_deg_tolerance", 4.0, 20.0),
                "min_area_habitable_m2": trial.suggest_float("min_area_habitable_m2", 0.8, 4.0),
                "min_area_service_m2": trial.suggest_float("min_area_service_m2", 0.3, 1.5),
                "default_door_width_m": trial.suggest_float("default_door_width_m", 0.75, 0.95),
            }
            m = score(val_cache, params)
            with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                mlflow.log_params(params)
                mlflow.log_metrics({f"val_{k}": v for k, v in m.items()})
            print(f"[HPO] trial {trial.number:>2}: objective {m['objective']:.4f} "
                  f"(foot {m['footprint_iou']:.3f}, shell {m['shell_rate']:.2f}, rooms {m['room_type_acc']:.3f})", flush=True)
            return m["objective"]

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.enqueue_trial(DEFAULTS)                 # trial 0 = the hand-picked defaults
        study.optimize(objective, n_trials=trials)
        best = dict(study.best_params)

        test_cache = predict(test_files)               # held-out check, once
        test_default, test_best = score(test_cache, DEFAULTS), score(test_cache, best)
        adopted = test_best["objective"] > test_default["objective"]
        mlflow.log_params({f"best_{k}": v for k, v in best.items()})
        mlflow.log_metrics({**{f"test_default_{k}": v for k, v in test_default.items()},
                            **{f"test_best_{k}": v for k, v in test_best.items()}, "adopted": float(adopted)})
        report = {"evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "objective": "0.4 footprint IoU + 0.3 shell rate + 0.3 room-type accuracy",
                  "trials": trials, "val_plans": val_plans, "test_plans": test_plans, "defaults": DEFAULTS, "best_params": best,
                  "validation": {"defaults": default_val, "best": study.best_value},
                  "test": {"defaults": test_default, "best": test_best}, "adopted": adopted,
                  "mlflow_run_id": parent.info.run_id, "elapsed_min": round((time.time() - t0) / 60, 1)}
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if adopted:
        TUNED_PARAMS.write_text(json.dumps({"params": best, "source": str(REPORT.name), "test_objective": test_best["objective"],
                                            "default_test_objective": test_default["objective"]}, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Optuna HPO of the layout vectorizer")
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--val-plans", type=int, default=30)
    ap.add_argument("--test-plans", type=int, default=60)
    a = ap.parse_args()
    r = run(a.trials, a.val_plans, a.test_plans)
    print(json.dumps({k: r[k] for k in ("best_params", "validation", "test", "adopted")}, indent=2))
