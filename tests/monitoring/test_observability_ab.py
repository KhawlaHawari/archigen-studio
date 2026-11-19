"""Prometheus metrics, business metrics and A/B / canary routing."""
import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from src.api.ab_testing import ABRouter
from src.api.main import app

client = TestClient(app)


def _plan_png(seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    img = np.full((256, 256, 3), 255, np.uint8)
    img[20:236, 20:28] = img[20:236, 228:236] = img[20:28, 20:236] = img[228:236, 20:236] = 0
    img[rng.integers(40, 200), 40:200] = 0
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def test_metrics_endpoint_exposes_prometheus_families():
    client.get("/health")
    text = client.get("/metrics").text
    for family in ("archigen_http_requests_total", "archigen_http_request_seconds_bucket",
                   "archigen_drift_score", "archigen_plans_analysed_total"):
        assert family in text, family


def test_layout_request_is_routed_and_counted_per_variant():
    router = ABRouter()
    variant = router.champion
    before = client.get("/metrics").text.count(f'variant="{variant}"')
    r = client.post("/predict/floorplan/layout", files={"file": ("p.png", _plan_png(), "image/png")},
                    headers={"X-Model-Variant": variant})
    assert r.status_code == 200
    assert r.json()["model_variant"] == variant and r.headers["X-Model-Variant"] == variant
    assert 'archigen_plans_analysed_total{variant="%s"}' % variant in client.get("/metrics").text
    assert client.get("/metrics").text.count(f'variant="{variant}"') >= before


def test_assignment_is_sticky_and_respects_the_canary_fraction(monkeypatch):
    monkeypatch.setenv("ARCHIGEN_AB_CHAMPION", "champ")
    monkeypatch.setenv("ARCHIGEN_AB_CHALLENGER", "chall")
    monkeypatch.setenv("ARCHIGEN_AB_CANARY_FRACTION", "0.2")
    monkeypatch.setattr(ABRouter, "checkpoint", staticmethod(lambda v: __import__("pathlib").Path(__file__)))
    router = ABRouter()
    plans = [f"plan-{i}".encode() for i in range(2000)]
    first = [router.assign(p) for p in plans]
    assert first == [router.assign(p) for p in plans]                  # sticky
    share = first.count("chall") / len(first)
    assert 0.15 < share < 0.25                                         # ~20 % canary
    assert router.assign(b"x", requested="chall") == "chall"           # explicit override


def test_promotion_decision_uses_a_two_proportion_z_test():
    few = ABRouter._decide({"requests": 10, "usable_rate": 0.5}, {"requests": 5, "usable_rate": 0.9})
    assert few["action"] == "keep_canary"
    better = ABRouter._decide({"requests": 400, "usable_rate": 0.60}, {"requests": 400, "usable_rate": 0.72})
    assert better["action"] == "promote_challenger" and better["p_value"] < 0.05
    worse = ABRouter._decide({"requests": 400, "usable_rate": 0.72}, {"requests": 400, "usable_rate": 0.60})
    assert worse["action"] == "rollback_challenger"
    same = ABRouter._decide({"requests": 100, "usable_rate": 0.70}, {"requests": 100, "usable_rate": 0.71})
    assert same["action"] == "keep_canary"


def test_ab_status_endpoint():
    data = client.get("/ab/status").json()
    assert data["champion"] and "variants" in data
