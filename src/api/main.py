"""
ArchiGen 3D - Production FastAPI Application (OS 5)
TEK-UP UNIVERSITÉ | Deep Learning Avancé & MLOps 2026/2027
Encadrant : Haythem Ghazouani
Équipe : Khalil Ayeb · Amine Ben Fadhel · Khawla Hawari
"""

import time
from pathlib import Path
from PIL import Image
import io
import torch
import onnxruntime as ort
import json
import base64
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

from src.config import (
    BASE_DIR,
    FLOORPLAN_CFG,
    FACADE_CFG,
    MLOPS_CFG
)
from src.api.schemas import (
    HealthResponse,
    SystemInfoResponse,
    FloorplanPredictionResponse,
    Floorplan3DResponse,
    FacadePredictionResponse,
    FacadeGenerateRequest,
    RoomDesignRequest,
    FurnitureChoicesRequest,
    LayoutDesignRequest,
)
from src.api.service import ModelService
from src.mlops.mlflow_tracker import get_measured_metrics
from src.api.monitoring_endpoints import router as monitoring_router

# Initialize FastAPI application
app = FastAPI(
    title="ArchiGen 3D MLOps Platform",
    description=(
        "Plateforme Intégrée de Segmentation Sémantique Multi-Tâches de Plans 2D (CubiCasa5K) "
        "et de Synthèse Photoréaliste de Façades par Réseaux Antagonistes Conditionnels (Pix2Pix cGAN)."
    ),
    version="1.0.0",
    contact={
        "name": "Khalil Ayeb · Amine Ben Fadhel · Khawla Hawari",
        "email": "haythem.ghazouani@tekup.de",
    }
)

# CORS Middleware for WebGL / Three.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# MLOps Level 3 Monitoring & DVC Remote Router
app.include_router(monitoring_router)

# Prometheus observability (technical, model and business metrics) + A/B / canary router
from src.api import observability as obs          # noqa: E402
from src.api.ab_testing import ROUTER as AB_ROUTER  # noqa: E402
from fastapi import Header                         # noqa: E402

app.middleware("http")(obs.prometheus_middleware)
_DRIFT_REFRESH = {"t": 0.0}


@app.get("/metrics", tags=["MLOps"])
def prometheus_metrics():
    """Prometheus scrape endpoint. Drift gauges are refreshed from the telemetry buffer at most once a minute."""
    if time.time() - _DRIFT_REFRESH["t"] > 60:
        _DRIFT_REFRESH["t"] = time.time()
        try:
            from src.mlops.drift_detector import DriftDetector
            for domain in ("floorplan", "facade"):
                obs.record_drift(DriftDetector().detect_drift_on_buffer(domain=domain), domain)
        except Exception as exc:   # monitoring must never break the scrape
            print(f"[metrics] drift refresh skipped: {exc}")
    return obs.metrics_response()


@app.post("/feedback/layout", tags=["MLOps"])
async def feedback_layout(
    file: UploadFile = File(...),
    layout: str = Form(...),
    model_variant: Optional[str] = Form(default=None),
    scale_m_per_px: Optional[float] = Form(default=None),
):
    """Feedback loop: store a user-corrected layout + its plan image as a future training example."""
    from src.mlops.feedback import save_feedback
    try:
        meta = save_feedback(await file.read(), file.filename, json.loads(layout), model_variant, scale_m_per_px)
    except (ValueError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    obs.FEEDBACK_CORRECTIONS.inc()
    return {"status": "saved", "feedback": meta}


@app.get("/feedback/stats", tags=["MLOps"])
def feedback_statistics():
    from src.mlops.feedback import feedback_stats
    from src.mlops.retraining_policy import last_training_time
    return feedback_stats(since=last_training_time() or None)


@app.get("/retraining/decision", tags=["MLOps"])
def retraining_decision():
    """Should the floor-plan model be retrained? (drift, new corrections, or schedule; see params.yaml)"""
    from src.mlops.retraining_policy import decide
    return decide()


@app.get("/ab/status", tags=["MLOps"])
def ab_status():
    """A/B / canary comparison of the floor-plan model variants and the promotion decision."""
    return AB_ROUTER.status()

START_TIME = time.time()


@app.get("/", response_class=HTMLResponse, tags=["UI Viewer"])
async def serve_viewer():
    """Serves the interactive 3D WebGL Three.js visualizer"""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h2>ArchiGen 3D Viewer Initializing...</h2>")


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Healthcheck endpoint reporting ONNX Runtime and PyTorch runtime providers"""
    providers = ort.get_available_providers()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return HealthResponse(
        torch_version=torch.__version__,
        onnxruntime_version=ort.__version__,
        active_providers=providers,
        device=device,
        uptime_seconds=round(time.time() - START_TIME, 2)
    )


@app.get("/system/info", response_model=SystemInfoResponse, tags=["System"])
async def system_info():
    """Returns project framing metadata and model specifications"""
    return SystemInfoResponse(
        models={
            "model_1_floorplan": {
                "name": FLOORPLAN_CFG.name,
                "dataset": "CubiCasa5K (5 000 plans)",
                "task": "Multi-Task Semantic Segmentation (Structures & Rooms)",
                "onnx_exported": FLOORPLAN_CFG.onnx_path.exists(),
                "checkpoint_exists": FLOORPLAN_CFG.checkpoint_path.exists(),
            },
            "model_2_facade": {
                "name": FACADE_CFG.name,
                "dataset": "CMP Facades (606 paires)",
                "task": "Image-to-Image Conditional Translation (cGAN)",
                "onnx_exported": FACADE_CFG.onnx_path.exists(),
                "checkpoint_exists": FACADE_CFG.checkpoint_path.exists(),
            }
        }
    )


@app.get("/metrics/summary", tags=["MLOps"])
async def get_metrics_summary():
    """Returns metrics MEASURED by real training/evaluation runs (models/metrics/*.json)."""
    parity_file = BASE_DIR / "models" / "onnx" / "parity_report.json"
    latency_file = BASE_DIR / "models" / "onnx" / "latency_benchmark.json"

    parity_data = None
    if parity_file.exists():
        import json
        with open(parity_file, "r") as f:
            parity_data = json.load(f)

    latency_data = None
    if latency_file.exists():
        import json
        with open(latency_file, "r") as f:
            latency_data = json.load(f)

    return {
        # Key kept for API compatibility; values are measured, never hardcoded.
        "official_benchmark": get_measured_metrics(),
        "onnx_parity_status": parity_data,
        "latency_sla_status": latency_data
    }


@app.post("/predict/floorplan", response_model=FloorplanPredictionResponse, tags=["Inference"])
async def predict_floorplan_endpoint(file: UploadFile = File(...)):
    """
    Upload a 2D floorplan image (PNG/JPEG).
    Returns multi-task segmentation masks and room area calculations.
    """
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    service = ModelService.get_instance()
    result = service.predict_floorplan(pil_img)
    return FloorplanPredictionResponse(
        image_width=result["image_width"],
        image_height=result["image_height"],
        total_habitable_area_m2=result["total_habitable_area_m2"],
        num_doors_detected=result["num_doors_detected"],
        num_windows_detected=result["num_windows_detected"],
        rooms=result["rooms"],
        structure_mask_base64=result["structure_mask_base64"],
        room_mask_base64=result["room_mask_base64"],
        processing_time_ms=result["processing_time_ms"]
    )


@app.post("/predict/floorplan/layout", tags=["3D Extrusion"])
async def floorplan_layout_endpoint(
    file: UploadFile = File(...),
    scale_override: Optional[float] = Form(default=None),
    reference_rooms: Optional[str] = Form(default=None),
    x_model_variant: Optional[str] = Header(default=None),
):
    """
    Upload a 2D floor plan image. Returns the vector layout used by the web studio's 3D view:
    walls (axis, thickness, openings), rooms (type, polygon, area, confidence), doors, windows
    and the building envelope, all in meters.
    """
    raw = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
    variant = AB_ROUTER.assign(raw, x_model_variant)
    try:
        regions = json.loads(reference_rooms) if reference_rooms else None
        if regions is not None and not isinstance(regions, list):
            raise ValueError("expected a list")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid reference room polygons: {e}")
    if scale_override is not None and scale_override <= 0:
        scale_override = None

    service = ModelService.get_instance()
    try:
        result = service.analyze_floorplan_layout(pil_img, scale_override=scale_override, reference_rooms=regions,
                                                  variant=variant)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Floorplan vectorization failed: {e}")
    result["model_variant"] = variant
    obs.record_plan(result, variant, result["processing_time_ms"] / 1000.0)
    return JSONResponse(content=result, headers={"X-Inference-Latency-Ms": str(result["processing_time_ms"]),
                                                 "X-Model-Variant": variant})


@app.post("/facade/generate", tags=["Studio"])
def facade_generate_endpoint(request: FacadeGenerateRequest):   # sync: runs in the thread pool (ControlNet can take minutes on CPU)
    """
    Facade texture for one exterior wall of the analysed plan: the wall's real doors and windows
    are drawn as a CMP label map, then this project's Pix2Pix turns it into a photo-like texture
    with UV coordinates for the 3D wall.
    """
    from src.geometry.dimensions import DOOR_HEIGHT_M, STOREY_HEIGHT_M, WINDOW_HEIGHT_M
    service = ModelService.get_instance()
    bridge = service.facade_bridge()
    wall = request.wall
    if wall is None and request.layout:
        walls = request.layout.get("walls", [])
        if request.wall_id:
            wall = next((w for w in walls if w.get("id") == request.wall_id), None)
            if wall is None:
                raise HTTPException(status_code=404, detail="The selected exterior wall was not found.")
        else:
            wall = bridge.select_front_wall(walls)
            if wall is None:
                raise HTTPException(status_code=422, detail="The floorplan has no exterior wall to finish.")
    if wall is not None:
        if wall.get("is_exterior") is False:
            raise HTTPException(status_code=422, detail="Choose an exterior wall for the facade.")
        wall = {**wall, "geometry_fixed": True}
    else:  # standalone preview
        wall = {
            "id": request.wall_id or "wall_front_default", "is_exterior": True, "is_front": True,
            "height_m": STOREY_HEIGHT_M, "polygon_meters": [[0.0, 0.0], [8.5, 0.0], [8.5, 0.25], [0.0, 0.25]],
            "openings": [
                {"type": "door", "position_along_wall_m": 4.25, "width": 1.15, "height": DOOR_HEIGHT_M},
                {"type": "window", "position_along_wall_m": 1.8, "width": 1.4, "height": WINDOW_HEIGHT_M},
                {"type": "window", "position_along_wall_m": 6.7, "width": 1.4, "height": WINDOW_HEIGHT_M},
            ],
        }
    num_floors = 1 if wall.get("level_id") else (request.num_floors or 1)
    t0 = time.perf_counter()
    try:
        from src.bridge.facade_bridge import apply_facade_style
        from src.geometry.facade_composer import generate_wall_texture
        # Wall frame and exact opening positions (meters) from the bridge geometry ...
        _, _, frame, openings = bridge._build_facade_sketch(wall, num_floors=num_floors, is_front=True)
        floor_h = float(wall.get("height_m") or STOREY_HEIGHT_M)
        length = float(frame["length_m"])
        # ... then this repository's composer: CMP-like multi-storey tiles -> Pix2Pix -> crop the real wall.
        high = (request.quality or "fast") == "high"
        if high:
            status = service.controlnet_status()
            if not status["available"]:
                raise HTTPException(status_code=503, detail=f"High-quality facades unavailable. {status['reason']}")
        generate = service.controlnet_generate_batch if high else bridge.inference_engine.generate_batch
        from src.models import superres
        use_sr = bool(request.super_resolution) and superres.available()
        out = generate_wall_texture(length, floor_h, num_floors, openings, generate,
                                    upscaler=(lambda im: superres.upscale(im, device=service.device)) if use_sr else None)
        texture = apply_facade_style(out["texture"], request.style) if request.style else out["texture"]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Facade generation error: {e}")
    height = floor_h * num_floors
    for op in openings:   # UV rectangles relative to the new texture (which covers exactly the wall)
        left, right = op["position_along_wall_m"] - op["width_m"] / 2, op["position_along_wall_m"] + op["width_m"] / 2
        op["uv_rect"] = [left / length, op["sill_m"] / height, right / length, (op["sill_m"] + op["height_m"]) / height]

    def to_b64(img):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    obs.record_facade("controlnet" if high else "pix2pix", time.perf_counter() - t0)
    return {
        "status": "success", "wall_id": wall.get("id"), "style": request.style, "num_floors": num_floors,
        "wall_length_m": length, "wall_height_m": height,
        "uv_bounds": {"u_min": 0.0, "u_max": 1.0, "v_min": 0.0, "v_max": 1.0, "wall_length_m": length,
                      "wall_height_m": height, "aspect_ratio": length / height},
        "facade_frame": frame, "openings": openings,
        "quality": "high" if high else "fast",
        "super_resolution": "Real-ESRGAN x4 (realesr-general-x4v3), post-processing" if use_sr else None,
        "model_info": {**({"model": "ArchiGen ControlNet (SD 1.5, fine-tuned on CMP)", "resolution": 512,
                           "style_conditioned": False} if high else bridge.inference_engine.describe()),
                       "composer": "multi-storey CMP context + tiling + crop",
                       "tiles": out["n_tiles"], "context_storeys": out["context_floors"]},
        "texture_base64": to_b64(texture), "sketch_base64": to_b64(out["sketch"]),
        "context_sketch_base64": to_b64(out["context_sketch"]), "context_photo_base64": to_b64(out["context_photo"]),
        "device": bridge.device_name,
        "processing_time_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


# --- Rule-based furniture placement (src/design, ported from the original ArchiGen) ---
@app.get("/facade/capabilities", tags=["Studio"])
async def facade_capabilities_endpoint():
    """Which facade generators are available: fast Pix2Pix always; high-quality ControlNet after Run 3."""
    from src.models import superres
    return {"fast": {"available": True, "model": "Pix2Pix 256 px"},
            "high": ModelService.get_instance().controlnet_status(),
            "super_resolution": {"available": superres.available(),
                                 "reason": None if superres.available() else "python -m src.models.superres --download"}}


@app.post("/design", tags=["Studio"])
async def design_room_endpoint(request: RoomDesignRequest):
    """Deterministic furnishing of one room: no collisions, door clearances, wall anchoring, BFS access."""
    from src.design.engine import RoomDesignEngine
    try:
        result = RoomDesignEngine().design_room(
            room_polygon_meters=request.polygon_meters, room_type=request.room_type,
            room_id=request.room_id or "room_0", wall_segments=request.wall_segments,
            doors=request.doors, windows=request.windows, preset=request.preset,
            seed=request.seed if request.seed is not None else 42,
            scale_m_per_px=request.scale_m_per_px, force_furnish=bool(request.force_furnish),
        )
        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Room design failed: {e}")


@app.post("/design/furniture-choices", tags=["Studio"])
async def furniture_choices_endpoint(request: FurnitureChoicesRequest):
    """Alternative catalog models that fit a placed item's location."""
    from src.design.engine import RoomDesignEngine
    try:
        choices = RoomDesignEngine().furniture_choices(
            room_polygon_meters=request.polygon_meters, room_type=request.room_type,
            items=[item.model_dump() for item in request.items], item_id=request.item_id,
            wall_segments=request.wall_segments, doors=request.doors, windows=request.windows,
            scale_m_per_px=request.scale_m_per_px, force_furnish=bool(request.force_furnish),
            custom_catalog={},
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"choices": choices}


@app.post("/design/layout", tags=["Studio"])
async def design_layout_endpoint(request: LayoutDesignRequest):
    """Furnishes every eligible room of a full layout."""
    from src.design.engine import RoomDesignEngine
    try:
        return RoomDesignEngine().design_layout(layout=request.layout, preset=request.preset,
                                                seed=request.seed if request.seed is not None else 42)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Layout design failed: {e}")


@app.get("/furniture/custom-library", tags=["Studio"])
async def custom_furniture_library_endpoint():
    """Image-to-3D furniture (TripoSR) needs a CUDA worker; this deployment has no custom library."""
    return {}


@app.post("/furniture/generate-3d", tags=["Studio"])
@app.post("/furniture/save-to-library", tags=["Studio"])
async def furniture_image_to_3d_unavailable():
    raise HTTPException(status_code=501, detail="Image-to-3D furniture (TripoSR) needs an NVIDIA GPU worker, "
                                                "which this deployment does not run. Use the furniture catalog instead.")


@app.post("/vectorize/floorplan", response_model=Floorplan3DResponse, tags=["3D Extrusion"])
async def vectorize_floorplan_endpoint(file: UploadFile = File(...)):
    """
    Upload a 2D floorplan image.
    Returns 3D meshes (exterior walls, interior walls, room floor slabs) ready for Three.js.
    """
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    service = ModelService.get_instance()
    result = service.predict_floorplan(pil_img)
    return Floorplan3DResponse(
        total_habitable_area_m2=result["total_habitable_area_m2"],
        num_doors_detected=result["num_doors_detected"],
        num_windows_detected=result["num_windows_detected"],
        rooms=result["rooms"],
        meshes=result["meshes"],
        processing_time_ms=result["processing_time_ms"]
    )


@app.post("/predict/facade", response_model=FacadePredictionResponse, tags=["Inference"])
async def predict_facade_endpoint(file: UploadFile = File(...)):
    """
    Upload a semantic elevation mask (256x256 RGB).
    Generates a photorealistic facade image via Pix2Pix cGAN.
    """
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    service = ModelService.get_instance()
    result = service.predict_facade(pil_img)
    return FacadePredictionResponse(
        width=result["width"],
        height=result["height"],
        input_mask_base64=result["input_mask_base64"],
        generated_facade_base64=result["generated_facade_base64"],
        processing_time_ms=result["processing_time_ms"]
    )


@app.get("/demo/floorplan", tags=["Demonstration"])
async def demo_floorplan_endpoint():
    """Runs segmentation and 3D vectorization on built-in synthetic floorplan"""
    service = ModelService.get_instance()
    return service.get_synthetic_floorplan()


@app.get("/demo/facade", tags=["Demonstration"])
async def demo_facade_endpoint():
    """Runs facade synthesis on built-in synthetic elevation mask"""
    service = ModelService.get_instance()
    return service.get_synthetic_facade()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
