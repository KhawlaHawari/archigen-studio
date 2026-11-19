"""Feedback loop: studio correction -> stored -> rasterised training example; retraining policy."""
import io
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import src.data.feedback_dataset as fd
import src.mlops.feedback as fb
from src.api.main import app

client = TestClient(app)
TEST_CACHE = Path("data/processed/cubicasa_cache/test")


@pytest.fixture()
def tmp_feedback(tmp_path, monkeypatch):
    monkeypatch.setattr(fb, "FEEDBACK_DIR", tmp_path / "feedback")
    monkeypatch.setattr(fd, "FEEDBACK_DIR", tmp_path / "feedback")
    return tmp_path


def _png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_feedback_is_saved_rasterised_and_aligned_with_the_plan(tmp_feedback):
    files = sorted(TEST_CACHE.glob("*.npz"))
    if not files:
        pytest.skip("CubiCasa test cache not available")
    data = np.load(files[0])
    image, struct_gt = data["image"], data["structure_mask"]
    png = _png(image)
    layout = client.post("/predict/floorplan/layout", files={"file": ("plan.png", png, "image/png")}).json()
    # the "correction": the user deletes the last room (any edit works for the pipeline)
    corrected = {**layout, "rooms": layout["rooms"][:-1]}
    r = client.post("/feedback/layout", files={"file": ("plan.png", png, "image/png")},
                    data={"layout": json.dumps(corrected), "model_variant": layout["model_variant"]})
    assert r.status_code == 200 and r.json()["status"] == "saved"
    assert client.get("/feedback/stats").json()["total"] >= 1
    assert "archigen_feedback_corrections_total" in client.get("/metrics").text

    summary = fd.build(out_dir=tmp_feedback / "cache")
    assert summary["converted"] == 1
    sample = np.load(next((tmp_feedback / "cache").glob("*.npz")))
    assert sample["image"].shape == (512, 512, 3) and sample["structure_mask"].shape == (512, 512)
    # geometry check: walls drawn back from the layout overlap the ground-truth walls of the plan
    pred_walls, gt_walls = np.isin(sample["structure_mask"], (1, 2)), np.isin(struct_gt, (1, 2))
    iou = (pred_walls & gt_walls).sum() / max((pred_walls | gt_walls).sum(), 1)
    assert iou > 0.4, f"rasterised walls do not line up with the plan (IoU {iou:.2f})"


def test_invalid_feedback_is_rejected(tmp_feedback):
    png = _png(np.full((64, 64, 3), 255, np.uint8))
    r = client.post("/feedback/layout", files={"file": ("p.png", png, "image/png")}, data={"layout": json.dumps({"rooms": []})})
    assert r.status_code == 422


def test_retraining_policy_reasons(monkeypatch, tmp_feedback):
    import time
    import src.mlops.retraining_policy as rp
    monkeypatch.setattr(rp, "DECISION", tmp_feedback / "decision.json")
    monkeypatch.setattr(rp, "load_policy", lambda: {"retrain_every_days": 30, "min_new_feedback": 2, "drift_triggers_retraining": True})
    now = time.time()
    monkeypatch.setattr(rp, "last_training_time", lambda: now - 3 * 86400)
    calm = rp.decide({"drift_detected": False}, now=now)
    assert calm["retrain"] is False
    assert rp.decide({"drift_detected": True, "max_psi": 0.4, "drift_score": 0.9}, now=now)["retrain"] is True
    old = rp.decide({"drift_detected": False}, now=now + 40 * 86400)
    assert old["retrain"] and "scheduled" in old["reasons"][0]
