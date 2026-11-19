"""
Retraining policy: decides WHEN to retrain, combining the two strategies of the course:

  * triggered retraining  : data drift detected (PSI / composite score from the drift detector)
                            or enough new user corrections (feedback loop);
  * scheduled (batch)     : the production model is older than `retrain_every_days`.

Thresholds come from params.yaml (section `retraining`). The decision is written to
models/metrics/retraining_decision.json. Run weekly by .github/workflows/scheduled-retraining.yml
(which opens a GitHub issue when retraining is due) or by hand:

    python -m src.mlops.retraining_policy            # exit code 0 = no retraining, 10 = retrain
    python -m src.mlops.retraining_policy --report-only   # always exit 0 (DVC stage)
Retraining itself runs on a GPU (Kaggle notebook) with `train_floorplan --include-feedback`.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parents[2]
DECISION = ROOT / "models" / "metrics" / "retraining_decision.json"
DEFAULTS = {"retrain_every_days": 30, "min_new_feedback": 25, "drift_triggers_retraining": True}


def load_policy() -> Dict[str, Any]:
    try:
        params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        params = {}
    return {**DEFAULTS, **(params.get("retraining") or {})}


def last_training_time() -> float:
    """When the production floor-plan model was produced: the date recorded INSIDE its test-metrics
    file (file modification times are useless in CI, where every checkout is "today"), else the
    checkpoint's modification time."""
    metrics = ROOT / "models" / "metrics" / "floorplan_unetpp_best_test.json"
    try:
        date = json.loads(metrics.read_text(encoding="utf-8"))["env"]["date"]
        return time.mktime(time.strptime(date, "%Y-%m-%d %H:%M:%S"))
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        pass
    ckpt = ROOT / "models" / "checkpoints" / "floorplan_unetpp_best.pt"
    return ckpt.stat().st_mtime if ckpt.exists() else 0.0


def decide(drift_report: Dict[str, Any] | None = None, now: float | None = None) -> Dict[str, Any]:
    from src.mlops.feedback import feedback_stats
    policy, now = load_policy(), now or time.time()
    trained_at = last_training_time()
    age_days = (now - trained_at) / 86400 if trained_at else None
    feedback = feedback_stats(since=trained_at or None)
    if drift_report is None:
        try:
            from src.mlops.drift_detector import DriftDetector
            drift_report = DriftDetector().detect_drift_on_buffer(domain="floorplan")
        except Exception as exc:
            drift_report = {"status": f"unavailable ({type(exc).__name__})"}
    reasons = []
    if policy["drift_triggers_retraining"] and drift_report.get("drift_detected"):
        reasons.append(f"data drift detected (max PSI {drift_report.get('max_psi')}, score {drift_report.get('drift_score')})")
    if feedback["new_since_last_training"] >= policy["min_new_feedback"]:
        reasons.append(f"{feedback['new_since_last_training']} new user corrections (threshold {policy['min_new_feedback']})")
    if age_days is not None and age_days >= policy["retrain_every_days"]:
        reasons.append(f"scheduled: production model is {age_days:.0f} days old (every {policy['retrain_every_days']} days)")
    result = {"evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "retrain": bool(reasons), "reasons": reasons,
              "policy": policy, "model_age_days": None if age_days is None else round(age_days, 1),
              "feedback": feedback, "drift": {k: drift_report.get(k) for k in ("status", "drift_detected", "drift_score", "max_psi")},
              "how_to_retrain": "Kaggle notebook (GPU) with: python -m src.training.train_floorplan --recipe v3 "
                                "--init-checkpoint models/checkpoints/floorplan_unetpp_best.pt --include-feedback"}
    DECISION.parent.mkdir(parents=True, exist_ok=True)
    DECISION.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    r = decide()
    print(json.dumps(r, indent=2))
    # --report-only (DVC stage): always exit 0; otherwise exit 10 when retraining is due (CI uses it)
    sys.exit(0 if "--report-only" in sys.argv or not r["retrain"] else 10)
