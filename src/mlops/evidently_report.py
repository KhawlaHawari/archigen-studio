"""
Evidently AI data-drift report: training reference vs recent production requests.

Reference = the feature samples stored in data/monitoring/reference_profile.json (built from the
training splits); current = the latest requests logged by the API in data/monitoring/inference_buffer.jsonl.
Both are the same per-image features the in-house drift detector (src/mlops/drift_detector.py) uses,
so Evidently gives an independent second opinion on the same data.

Output: data/monitoring/evidently_<domain>.html (interactive report, also served at
GET /monitoring/drift/evidently) and data/monitoring/evidently_<domain>_summary.json.

Evidently needs Python <= 3.12 (it does not import on 3.14). On this machine:
    py -3.11 -m pip install -r requirements-monitoring.txt
    py -3.11 src/mlops/evidently_report.py --domain floorplan     (plain file: the src.mlops package imports torch)
The production Docker image (python:3.11) runs it directly.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

MONITORING_DIR = Path(__file__).resolve().parents[2] / "data" / "monitoring"


def load_frames(domain: str, max_current: int = 200):
    profile = json.loads((MONITORING_DIR / "reference_profile.json").read_text(encoding="utf-8"))
    reference = pd.DataFrame(profile[f"{domain}_reference"])
    rows = []
    buffer = MONITORING_DIR / "inference_buffer.jsonl"
    if buffer.exists():
        for line in buffer.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("domain") == domain:
                rows.append(rec["features"])
    current = pd.DataFrame(rows[-max_current:])
    cols = [c for c in reference.columns if c in current.columns and c != "timestamp"]
    return reference[cols].astype(float), current[cols].astype(float)


def run_report(domain: str = "floorplan", max_current: int = 200) -> dict:
    reference, current = load_frames(domain, max_current)
    if len(current) < 5:
        raise SystemExit(f"Not enough production samples for '{domain}' ({len(current)}); use the studio or "
                         "POST /monitoring/drift/simulate first.")
    html = MONITORING_DIR / f"evidently_{domain}.html"
    try:                                             # Evidently >= 0.7
        from evidently import Report
        from evidently.presets import DataDriftPreset
        snapshot = Report([DataDriftPreset()]).run(current_data=current, reference_data=reference)
        snapshot.save_html(str(html))
        raw = snapshot.dict()
    except ImportError:                              # Evidently 0.4 - 0.6
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)
        report.save_html(str(html))
        raw = report.as_dict()
    summary = {"domain": domain, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "reference_samples": len(reference), "current_samples": len(current),
               "features": list(reference.columns), "html_report": str(html), "evidently_result": _drift_summary(raw)}
    (MONITORING_DIR / f"evidently_{domain}_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def _drift_summary(raw: dict) -> dict:
    """Dataset-level and per-feature drift from the Evidently result (0.7 'metrics' list or 0.4-0.6 format)."""
    out: dict = {"per_feature": {}}
    for m in raw.get("metrics", []):
        name = str(m.get("metric_name", m.get("metric", "")))
        value, config = m.get("value"), m.get("config", {})
        if name.startswith("DriftedColumnsCount") and isinstance(value, dict):        # Evidently >= 0.7
            out["drifted_features"] = int(value.get("count", 0))
            out["drifted_share"] = round(float(value.get("share", 0.0)), 4)
            out["dataset_drift"] = value.get("share", 0.0) >= config.get("drift_share", 0.5)
            out["dataset_drift_rule"] = f"share of drifted features >= {config.get('drift_share', 0.5)}"
        elif name.startswith("ValueDrift") and "column" in config:
            p, thr = float(value), float(config.get("threshold", 0.05))
            out["per_feature"][config["column"]] = {"method": config.get("method"), "score": p,
                                                    "drifted": p < thr if "p_value" in str(config.get("method")) else p > thr}
        elif isinstance(m.get("result"), dict) and "number_of_drifted_columns" in m["result"]:   # Evidently 0.4-0.6
            r = m["result"]
            out.update(drifted_features=r["number_of_drifted_columns"], drifted_share=r["share_of_drifted_columns"],
                       dataset_drift=r["dataset_drift"])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evidently data-drift report (reference vs production)")
    ap.add_argument("--domain", choices=["floorplan", "facade"], default="floorplan")
    ap.add_argument("--max-current", type=int, default=200)
    a = ap.parse_args()
    print(json.dumps(run_report(a.domain, a.max_current), indent=2, default=str))
