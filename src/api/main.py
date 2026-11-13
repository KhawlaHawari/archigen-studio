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
from fastapi import FastAPI, UploadFile, File, HTTPException
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
    FacadePredictionResponse
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
