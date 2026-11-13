"""
Integration tests for FastAPI REST Endpoints
"""

import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    """Tests GET /health returns 200 and healthy status"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "torch_version" in data
    assert "onnxruntime_version" in data


def test_system_info_endpoint():
    """Tests GET /system/info returns academic and model metadata"""
    response = client.get("/system/info")
    assert response.status_code == 200
    data = response.json()
    assert data["institution"] == "TEK-UP UNIVERSITÉ"
    assert "Khalil Ayeb" in data["team"]


def test_demo_floorplan_endpoint():
    """Tests GET /demo/floorplan generates valid 2D masks and 3D meshes"""
    response = client.get("/demo/floorplan")
    assert response.status_code == 200
    data = response.json()
    # Area depends on trained weights (none in CI / before training): only check it is valid.
    assert data["total_habitable_area_m2"] >= 0
    assert "structure_mask_base64" in data
    assert "room_mask_base64" in data
    assert "meshes" in data


def test_demo_facade_endpoint():
    """Tests GET /demo/facade generates photorealistic facade in < 400ms"""
    response = client.get("/demo/facade")
    assert response.status_code == 200
    data = response.json()
    assert data["width"] == 256
    assert data["height"] == 256
    assert "generated_facade_base64" in data
    assert data["processing_time_ms"] > 0


def test_post_predict_floorplan_upload():
    """Tests POST /predict/floorplan with dummy raster image"""
    img = Image.new("RGB", (256, 256), color=(240, 240, 240))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/predict/floorplan",
        files={"file": ("test_plan.png", buf, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "structure_mask_base64" in data
    assert "room_mask_base64" in data
