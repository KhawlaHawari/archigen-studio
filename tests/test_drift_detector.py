"""
Unit & Integration Tests for MLOps Level 3 Drift Monitoring & Retraining Engine
TEK-UP UNIVERSITÉ | Deep Learning Avancé & MLOps 2026/2027
Encadrant : Haythem Ghazouani
Équipe : Khalil Ayeb · Amine Ben Fadhel · Khawla Hawari
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient

from src.api.main import app
from src.mlops.drift_detector import (
    calculate_psi,
    calculate_ks_test,
    calculate_wasserstein,
    extract_floorplan_features,
    extract_facade_features,
    load_reference_profile,
    DriftDetector,
    simulate_production_drift
)
from src.mlops.continuous_retraining import evaluate_and_trigger_ct
from src.mlops.dvc_remote import get_full_dvc_status, get_dvc_remotes

client = TestClient(app)


def test_psi_stable_distribution():
    """Identical distributions must yield PSI close to 0 (< 0.10)."""
    np.random.seed(42)
    b = np.random.normal(100, 15, 1000)
    c = np.random.normal(100, 15, 1000)
    psi = calculate_psi(b, c)
    assert psi < 0.10, f"Expected stable PSI < 0.10, got {psi}"


def test_psi_drifted_distribution():
    """Significantly shifted distribution must yield PSI >= 0.20."""
    np.random.seed(42)
    b = np.random.normal(100, 15, 1000)
    c = np.random.normal(150, 25, 1000)  # Major shift
    psi = calculate_psi(b, c)
    assert psi >= 0.20, f"Expected drifted PSI >= 0.20, got {psi}"


def test_ks_test_sensitivity():
    """KS test must detect shift with p-value < 0.05."""
    np.random.seed(42)
    b = np.random.normal(0, 1, 500)
    c = np.random.normal(1.5, 1, 500)
    res = calculate_ks_test(b, c)
    assert res["drift_detected"] is True
    assert res["p_value"] < 0.05


def test_feature_extraction_floorplan():
    """Validates statistical feature extraction from 2D floorplan image and masks."""
    img = np.ones((512, 512, 3), dtype=np.uint8) * 240
    struct = np.zeros((512, 512), dtype=np.uint8)
    struct[100:150, 100:400] = 1  # exterior wall
    room = np.zeros((512, 512), dtype=np.uint8)
    room[150:350, 100:400] = 1  # living room

    feat = extract_floorplan_features(img, struct, room, total_area_m2=75.0)
    assert "mean_brightness" in feat
    assert "std_brightness" in feat
    assert "edge_density" in feat
    assert "struct_wall_ratio" in feat
    assert feat["total_area_m2"] == 75.0
    assert feat["mean_brightness"] > 200


def test_feature_extraction_facade():
    """Validates statistical feature extraction from facade semantic mask."""
    mask = np.zeros((256, 256, 3), dtype=np.uint8)
    mask[50:200, 50:200] = [0, 0, 255]  # facade blue
    photo = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)

    feat = extract_facade_features(mask, photo)
    assert "facade_wall_ratio" in feat
    assert "gen_brightness" in feat
    assert "gen_sharpness" in feat
    assert feat["facade_wall_ratio"] > 0.0


def test_reference_profile_loaded():
    """Reference profile must contain floorplan and facade feature distributions."""
    profile = load_reference_profile()
    assert "floorplan_reference" in profile
    assert "facade_reference" in profile
    assert "mean_brightness" in profile["floorplan_reference"]
    assert len(profile["floorplan_reference"]["mean_brightness"]) >= 10


def test_drift_detection_and_retraining_trigger():
    """Simulates drifted production traffic and verifies automated drift detection & CT recommendation."""
    simulate_production_drift(domain="floorplan", drift_type="inverted_dark_mode", n_samples=12)
    detector = DriftDetector()
    report = detector.detect_drift_on_buffer(domain="floorplan")

    assert report["drift_detected"] is True
    assert report["retraining_recommended"] is True
    assert report["status"] in ("DRIFT_DETECTED", "WARNING_DRIFT")
    assert report["max_psi"] >= 0.10

    # Test CT trigger simulation
    ct_res = evaluate_and_trigger_ct(domain="floorplan", force=False, dry_run=True)
    assert ct_res["retraining_triggered"] is True
    assert ct_res["status"] == "SIMULATED_TRIGGER"


def test_monitoring_api_endpoints():
    """Tests all FastAPI MLOps Level 3 monitoring endpoints."""
    # 1. DVC status
    res = client.get("/monitoring/dvc/status")
    assert res.status_code == 200
    dvc_data = res.json()
    assert dvc_data["dvc_initialized"] is True
    assert "minio-storage" in dvc_data["remotes"]

    # 2. Drift status
    res = client.get("/monitoring/drift/status?domain=floorplan")
    assert res.status_code == 200
    assert "drift_detected" in res.json()

    # 3. Drift full report
    res = client.get("/monitoring/drift/report?domain=floorplan")
    assert res.status_code == 200
    assert "feature_reports" in res.json()

    # 4. Drift simulation endpoint
    res = client.post("/monitoring/drift/simulate?domain=floorplan&drift_type=heavy_noise&samples=8")
    assert res.status_code == 200
    assert "drift_audit_result" in res.json()

    # 5. Dashboard HTML page
    res = client.get("/monitoring/drift/dashboard")
    assert res.status_code == 200
    assert "MLOps Niveau 3" in res.text
