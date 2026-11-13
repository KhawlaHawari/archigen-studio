"""
Monitoring & health tests for ArchiGen 3D API
Validates system health, metrics freshness, and endpoint availability.
Run with: pytest tests/monitoring/ -v
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_endpoint_healthy():
    """GET /health must return 200 with healthy status and valid versions."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "torch_version" in data
    assert "onnxruntime_version" in data
    assert data["device"] in ("cpu", "cuda:0")
    assert data["uptime_seconds"] >= 0


def test_system_info_endpoint_complete():
    """GET /system/info returns academic framing + model export status."""
    response = client.get("/system/info")
    assert response.status_code == 200
    data = response.json()
    assert data["institution"] == "TEK-UP UNIVERSITÉ"
    assert data["academic_year"] == "2026/2027"
    assert "Khalil Ayeb" in data["team"]
    assert "model_1_floorplan" in data["models"]
    assert "model_2_facade" in data["models"]


def test_metrics_summary_endpoint_available():
    """GET /metrics/summary returns benchmark + parity + latency data."""
    response = client.get("/metrics/summary")
    assert response.status_code == 200
    data = response.json()
    assert "official_benchmark" in data


def test_api_responds_within_sla():
    """Basic responsiveness SLA: healthcheck < 500ms."""
    import time
    t0 = time.perf_counter()
    client.get("/health")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 500, f"Healthcheck too slow: {elapsed_ms:.0f}ms"


def test_dvc_remote_monitoring_endpoint():
    """GET /monitoring/dvc/status returns DVC initialization and remote configurations."""
    response = client.get("/monitoring/dvc/status")
    assert response.status_code == 200
    data = response.json()
    assert data["dvc_initialized"] is True
    assert "remotes" in data
    assert "artifacts" in data


def test_drift_status_monitoring_endpoint():
    """GET /monitoring/drift/status returns Level 3 drift state and metrics."""
    response = client.get("/monitoring/drift/status?domain=floorplan")
    assert response.status_code == 200
    data = response.json()
    assert data["domain"] == "floorplan"
    assert "drift_detected" in data
    assert "drift_score" in data
    assert "retraining_recommended" in data
