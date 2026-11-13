"""
Pydantic Schemas for ArchiGen 3D Production REST API
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple


class HealthResponse(BaseModel):
    status: str = "healthy"
    app_name: str = "ArchiGen 3D REST API"
    version: str = "1.0.0"
    torch_version: str
    onnxruntime_version: str
    active_providers: List[str]
    device: str
    uptime_seconds: float


class SystemInfoResponse(BaseModel):
    app: str = "ArchiGen 3D"
    institution: str = "TEK-UP UNIVERSITÉ"
    academic_year: str = "2026/2027"
    module: str = "Deep Learning Avancé & MLOps"
    prof: str = "Haythem Ghazouani"
    team: List[str] = ["Khalil Ayeb", "Amine Ben Fadhel", "Khawla Hawari"]
    models: Dict[str, Any]


class RoomSummaryItem(BaseModel):
    room_id: int
    name: str
    area_m2: float
    color_rgb: List[int]
    color_hex: str


class FloorplanPredictionResponse(BaseModel):
    image_width: int
    image_height: int
    total_habitable_area_m2: float
    num_doors_detected: int
    num_windows_detected: int
    rooms: List[RoomSummaryItem]
    structure_mask_base64: str
    room_mask_base64: str
    processing_time_ms: float


class MeshGeometry(BaseModel):
    vertices: List[float]
    indices: List[int]
    normals: List[float]
    color: List[float]
    height: float


class RoomMeshItem(BaseModel):
    room_name: str
    area_m2: float
    mesh: MeshGeometry


class MeshCollection(BaseModel):
    exterior_walls: Optional[MeshGeometry] = None
    interior_walls: Optional[MeshGeometry] = None
    room_floors: List[RoomMeshItem] = Field(default_factory=list)


class Floorplan3DResponse(BaseModel):
    total_habitable_area_m2: float
    num_doors_detected: int
    num_windows_detected: int
    rooms: List[RoomSummaryItem]
    meshes: MeshCollection
    processing_time_ms: float


class FacadePredictionResponse(BaseModel):
    width: int
    height: int
    input_mask_base64: str
    generated_facade_base64: str
    processing_time_ms: float
