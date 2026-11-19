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


# --- Studio: facade on a wall + rule-based furniture (request formats of the original ArchiGen API) ---
from pydantic import FiniteFloat  # noqa: E402


class FacadeGenerateRequest(BaseModel):
    wall_id: Optional[str] = None
    style: Optional[str] = "Modern Minimalist"
    num_floors: Optional[int] = 1
    wall: Optional[Dict[str, Any]] = None
    layout: Optional[Dict[str, Any]] = None
    quality: Optional[str] = "fast"   # "fast" = Pix2Pix 256 px, "high" = fine-tuned ControlNet 512 px
    super_resolution: Optional[bool] = True   # 4x Real-ESRGAN post-processing of the texture (if weights exist)


class RoomDesignRequest(BaseModel):
    room_id: Optional[str] = Field(default="room_0")
    room_type: str = Field(...)
    polygon_meters: List[List[float]] = Field(...)
    doors: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    windows: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    wall_segments: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    preset: str = Field(default="contemporary_minimal")
    seed: Optional[int] = Field(default=42)
    room_edges: Optional[List[Dict[str, Any]]] = Field(default=None)
    force_furnish: Optional[bool] = Field(default=False)
    scale_m_per_px: float = Field(default=0.03, gt=0)


class PlacedFurnitureSnapshot(BaseModel):
    item_id: str
    catalog_id: str
    category: Optional[str] = None
    position_m: Tuple[FiniteFloat, FiniteFloat]
    rotation_deg: FiniteFloat
    dimensions_m: Tuple[FiniteFloat, FiniteFloat, FiniteFloat]
    scale: Tuple[FiniteFloat, FiniteFloat, FiniteFloat]
    footprint_polygon: List[Tuple[FiniteFloat, FiniteFloat]] = Field(min_length=4, max_length=4)


class FurnitureChoicesRequest(RoomDesignRequest):
    item_id: str
    items: List[PlacedFurnitureSnapshot] = Field(min_length=1, max_length=64)


class LayoutDesignRequest(BaseModel):
    layout: Dict[str, Any] = Field(...)
    preset: str = Field(default="contemporary_minimal")
    seed: Optional[int] = Field(default=42)
