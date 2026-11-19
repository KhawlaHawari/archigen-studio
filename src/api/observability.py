"""
Production observability for the ArchiGen API (Prometheus format, scraped at GET /metrics).

Three families of metrics, matching the course's monitoring checklist:
  * technical  : request count / errors / latency per endpoint (histogram -> p50 / p95 in Grafana);
  * model      : inference latency per model variant (A/B / canary), drift scores (PSI, composite);
  * business   : plans analysed, usable plans (>= 3 rooms and >= 25 m2), rooms and area per plan,
                 closed 3D shells, facades generated per generator, user corrections (feedback loop).

Alert rules on these metrics live in monitoring/prometheus/alerts.yml; the Grafana dashboard in
monitoring/grafana/dashboards/archigen.json.
"""
from __future__ import annotations

import time

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.requests import Request
from starlette.responses import Response

# ---- technical -----------------------------------------------------------------------------
HTTP_REQUESTS = Counter("archigen_http_requests_total", "HTTP requests", ["endpoint", "method", "status"])
HTTP_LATENCY = Histogram("archigen_http_request_seconds", "HTTP request latency", ["endpoint"],
                         buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 10, 30, 60, 180, 600))
# ---- model ---------------------------------------------------------------------------------
INFERENCE_LATENCY = Histogram("archigen_model_inference_seconds", "Model inference + post-processing latency",
                              ["task", "variant"], buckets=(0.1, 0.25, 0.5, 1, 2, 3, 5, 10, 30, 120, 600))
DRIFT_PSI_MAX = Gauge("archigen_drift_psi_max", "Max Population Stability Index over monitored features", ["domain"])
DRIFT_SCORE = Gauge("archigen_drift_score", "Composite drift score (0..1)", ["domain"])
DRIFT_DETECTED = Gauge("archigen_drift_detected", "1 if the last drift check flagged drift", ["domain"])
# ---- business ------------------------------------------------------------------------------
PLANS_ANALYSED = Counter("archigen_plans_analysed_total", "Floor plans turned into a 3D layout", ["variant"])
PLANS_USABLE = Counter("archigen_plans_usable_total", "Plans with >= 3 rooms and >= 25 m2", ["variant"])
PLANS_SHELL_CLOSED = Counter("archigen_plans_shell_closed_total", "Plans with a watertight 3D shell", ["variant"])
ROOMS_PER_PLAN = Histogram("archigen_rooms_per_plan", "Rooms detected per plan", ["variant"],
                           buckets=(0, 1, 2, 3, 5, 8, 12, 16, 24))
PLAN_AREA = Histogram("archigen_plan_area_m2", "Total room area per plan (m2)", ["variant"],
                      buckets=(0, 10, 25, 50, 75, 100, 150, 250))
FACADES_GENERATED = Counter("archigen_facades_generated_total", "Facade textures generated", ["generator"])
FEEDBACK_CORRECTIONS = Counter("archigen_feedback_corrections_total", "User-corrected layouts saved for retraining")

_SKIP = ("/metrics", "/static", "/docs", "/openapi.json", "/redoc", "/favicon")


def _endpoint(request: Request) -> str:
    route = request.scope.get("route")
    return getattr(route, "path", None) or "unmatched"


async def prometheus_middleware(request: Request, call_next):
    if request.url.path.startswith(_SKIP):
        return await call_next(request)
    t0 = time.perf_counter()
    status = "500"
    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    finally:
        endpoint = _endpoint(request)
        HTTP_REQUESTS.labels(endpoint, request.method, status).inc()
        HTTP_LATENCY.labels(endpoint).observe(time.perf_counter() - t0)


def record_plan(layout: dict, variant: str, seconds: float) -> None:
    rooms = layout.get("rooms", [])
    area = sum(r.get("surface_m2", 0.0) for r in rooms)
    INFERENCE_LATENCY.labels("floorplan", variant).observe(seconds)
    PLANS_ANALYSED.labels(variant).inc()
    ROOMS_PER_PLAN.labels(variant).observe(len(rooms))
    PLAN_AREA.labels(variant).observe(area)
    if len(rooms) >= 3 and area >= 25.0:
        PLANS_USABLE.labels(variant).inc()
    if layout.get("metadata", {}).get("is_building_shell_valid"):
        PLANS_SHELL_CLOSED.labels(variant).inc()


def record_facade(generator: str, seconds: float) -> None:
    INFERENCE_LATENCY.labels("facade", generator).observe(seconds)
    FACADES_GENERATED.labels(generator).inc()


def record_drift(report: dict, domain: str = "floorplan") -> None:
    DRIFT_PSI_MAX.labels(domain).set(float(report.get("max_psi", 0.0) or 0.0))
    DRIFT_SCORE.labels(domain).set(float(report.get("drift_score", 0.0) or 0.0))
    DRIFT_DETECTED.labels(domain).set(1.0 if report.get("drift_detected") else 0.0)


def metrics_response() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
