"""Generates monitoring/grafana/dashboards/archigen.json (run from the repo root)."""
import json
from pathlib import Path

DS = {"type": "prometheus", "uid": "prometheus"}
panels, y = [], 0


def row(title):
    global y
    panels.append({"type": "row", "title": title, "collapsed": False, "gridPos": {"h": 1, "w": 24, "x": 0, "y": y}, "panels": []})
    y += 1


def panel(title, exprs, kind="timeseries", w=8, h=8, unit=None, x=None, thresholds=None, desc=""):
    global y
    pid = len(panels) + 1
    used = sum(p["gridPos"]["w"] for p in panels if p["gridPos"]["y"] == y and p["type"] != "row")
    if x is None:
        x = used
    if x + w > 24:
        y += h
        x = 0
    p = {"id": pid, "type": kind, "title": title, "description": desc, "datasource": DS,
         "gridPos": {"h": h, "w": w, "x": x, "y": y},
         "targets": [{"datasource": DS, "expr": e, "legendFormat": lf, "refId": chr(65 + i)} for i, (e, lf) in enumerate(exprs)],
         "fieldConfig": {"defaults": {}, "overrides": []}, "options": {}}
    if unit:
        p["fieldConfig"]["defaults"]["unit"] = unit
    if thresholds:
        p["fieldConfig"]["defaults"]["thresholds"] = {"mode": "absolute", "steps": thresholds}
    panels.append(p)
    if x + w >= 24:
        y += h


row("Technical: API")
panel("Requests / s by endpoint", [('sum by (endpoint) (rate(archigen_http_requests_total[5m]))', "{{endpoint}}")], unit="reqps")
panel("p95 latency by endpoint", [('histogram_quantile(0.95, sum by (le, endpoint) (rate(archigen_http_request_seconds_bucket[5m])))', "{{endpoint}}")], unit="s")
panel("Error rate (5xx)", [('sum(rate(archigen_http_requests_total{status=~"5.."}[5m])) / clamp_min(sum(rate(archigen_http_requests_total[5m])), 1e-9)', "5xx share")],
      kind="stat", unit="percentunit", thresholds=[{"color": "green", "value": None}, {"color": "orange", "value": 0.01}, {"color": "red", "value": 0.05}])
row("Model: A/B / canary (floor plans) and generators")
panel("Plans analysed per model variant", [('sum by (variant) (increase(archigen_plans_analysed_total[1h]))', "{{variant}}")], kind="barchart")
panel("Usable-plan rate per variant (business KPI)", [('sum by (variant) (increase(archigen_plans_usable_total[6h])) / clamp_min(sum by (variant) (increase(archigen_plans_analysed_total[6h])), 1)', "{{variant}}")],
      unit="percentunit", desc=">= 3 rooms and >= 25 m2. Compare champion vs challenger; decision at GET /ab/status.")
panel("Inference latency p95 per variant", [('histogram_quantile(0.95, sum by (le, task, variant) (rate(archigen_model_inference_seconds_bucket[15m])))', "{{task}} / {{variant}}")], unit="s")
row("Business")
panel("Closed 3D shell rate", [('sum(increase(archigen_plans_shell_closed_total[6h])) / clamp_min(sum(increase(archigen_plans_analysed_total[6h])), 1)', "closed shells")],
      kind="stat", unit="percentunit", thresholds=[{"color": "red", "value": None}, {"color": "orange", "value": 0.8}, {"color": "green", "value": 0.95}])
panel("Median rooms per plan", [('histogram_quantile(0.5, sum by (le) (rate(archigen_rooms_per_plan_bucket[1h])))', "median rooms")], kind="stat")
panel("Facades generated (24 h)", [('sum by (generator) (increase(archigen_facades_generated_total[24h]))', "{{generator}}")], kind="stat")
panel("User corrections saved for retraining (24 h)", [('sum(increase(archigen_feedback_corrections_total[24h]))', "corrections")], kind="stat", w=8)
panel("Median plan area (m2)", [('histogram_quantile(0.5, sum by (le) (rate(archigen_plan_area_m2_bucket[1h])))', "m2")], kind="stat", w=8)
panel("Plans analysed (24 h)", [('sum(increase(archigen_plans_analysed_total[24h]))', "plans")], kind="stat", w=8)
row("Data drift (thresholds: PSI 0.10 warning, 0.20 critical)")
panel("Max PSI by domain", [('archigen_drift_psi_max', "{{domain}}")], kind="gauge",
      thresholds=[{"color": "green", "value": None}, {"color": "orange", "value": 0.1}, {"color": "red", "value": 0.2}])
panel("Composite drift score", [('archigen_drift_score', "{{domain}}")], kind="timeseries")
panel("Drift detected", [('archigen_drift_detected', "{{domain}}")], kind="stat",
      thresholds=[{"color": "green", "value": None}, {"color": "red", "value": 1}])

dash = {"title": "ArchiGen 3D: production monitoring", "uid": "archigen-prod", "schemaVersion": 39, "version": 1,
        "time": {"from": "now-6h", "to": "now"}, "refresh": "30s", "tags": ["archigen", "mlops"], "panels": panels}
out = Path("monitoring/grafana/dashboards/archigen.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(dash, indent=2), encoding="utf-8")
print("panels:", sum(p["type"] != "row" for p in panels))
