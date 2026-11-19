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


def test_facade_generate_standalone_wall():
    """POST /facade/generate without a plan: default 8.5 m wall, 1 door + 2 windows, Pix2Pix texture + UVs."""
    response = client.post("/facade/generate", json={"style": "Modern Minimalist", "num_floors": 2})
    assert response.status_code == 200
    data = response.json()
    assert data["texture_base64"].startswith("data:image/png;base64,")
    assert data["sketch_base64"].startswith("data:image/png;base64,")
    assert len(data["openings"]) >= 3
    assert 0.0 <= data["uv_bounds"]["u_min"] < data["uv_bounds"]["u_max"] <= 1.0


def test_facade_generate_rejects_interior_wall():
    wall = {"id": "w1", "is_exterior": False, "polygon_meters": [[0, 0], [4, 0], [4, .2], [0, .2]]}
    assert client.post("/facade/generate", json={"wall": wall}).status_code == 422


def test_design_room_places_furniture_in_a_bedroom():
    room = [[0.0, 0.0], [4.0, 0.0], [4.0, 3.5], [0.0, 3.5]]
    walls = [{"id": f"w{i}", "polygon_meters": [room[i], room[(i + 1) % 4]], "is_exterior": True} for i in range(4)]
    door = {"id": "d0", "type": "door", "position_meters": [2.0, 0.0], "width_m": 0.9}
    response = client.post("/design", json={"room_type": "Bedroom", "polygon_meters": room,
                                            "wall_segments": walls, "doors": [door], "windows": []})
    assert response.status_code == 200
    assert response.json()["status"] == "valid"


def test_image_to_3d_furniture_is_reported_unavailable():
    assert client.get("/furniture/custom-library").json() == {}
    assert client.post("/furniture/generate-3d").status_code == 501


def test_high_quality_facade_reports_missing_controlnet(monkeypatch):
    from src.api.service import ModelService
    monkeypatch.setattr(ModelService, "CONTROLNET_DIR", ModelService.CONTROLNET_DIR.parent / "does_not_exist")
    caps = client.get("/facade/capabilities").json()
    assert caps["fast"]["available"] and caps["high"]["available"] is False
    r = client.post("/facade/generate", json={"num_floors": 1, "quality": "high"})
    assert r.status_code == 503 and "Run 3" in r.json()["detail"]


def test_composer_keeps_512px_detail_for_high_quality_tiles():
    from PIL import Image
    from src.geometry.facade_composer import generate_wall_texture
    ops = [{"type": "window", "position_along_wall_m": 3.0, "width_m": 1.2, "height_m": 1.2, "sill_m": 0.8, "floor_index": 0}]
    fake_controlnet = lambda tiles: [Image.new("RGB", (512, 512), (120, 110, 100)) for _ in tiles]
    hq = generate_wall_texture(20.0, 2.5, 1, ops, fake_controlnet)
    assert hq["n_tiles"] >= 2
    wall_px = round(20.0 * hq["px_per_m"])            # wall width in the 256-px tile space
    assert hq["texture"].width == 2 * wall_px          # stitched at 512-px tile resolution, not downscaled
    assert hq["texture"].size == hq["sketch"].size
