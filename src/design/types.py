"""
src/design/types.py
-------------------
Data models and type definitions for the deterministic rule-based placement engine.
All spatial coordinates are in metric meters in 2D floorplan space (plan coordinates).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from shapely.geometry import Polygon, Point, LineString

# Minimum plausible physical room areas (in m²) for residential floorplans.
# Polygons below these thresholds represent neural over-segmentation fragments
# and receive status "not a room (fragment)" without placement.
MIN_PLAUSIBLE_ROOM_AREAS: Dict[str, float] = {
    "bathroom": 2.0,      # Compact powder room / toilet + sink
    "bedroom": 5.0,       # Minimum single bedroom code standard
    "kitchen": 3.0,       # Compact kitchenette
    "living_room": 6.0,   # Compact studio living space
}

@dataclass
class WallSegment:
    """A linear wall segment or snapped wall face in plan meters."""
    start: Tuple[float, float]
    end: Tuple[float, float]
    length_m: float
    is_exterior: bool = False
    wall_id: Optional[str] = None
    normal: Tuple[float, float] = (0.0, 0.0)  # Inward unit normal pointing into room

@dataclass
class OpeningSpan:
    """A door or window opening located along an edge."""
    opening_type: str  # "door" or "window"
    start_dist: float  # distance along edge from edge start in meters
    end_dist: float    # distance along edge from edge start in meters
    clearance_depth: float = 0.85  # Door swing clearance box depth into room

@dataclass
class UsableWallSegment:
    """An unobstructed sub-segment of a wall-backed room edge, after subtracting openings."""
    start: Tuple[float, float]
    end: Tuple[float, float]
    length_m: float
    normal: Tuple[float, float]  # Inward normal pointing into room (dx, dy)
    yaw_deg: float               # Clockwise angle in degrees from North for back-to-wall items
    edge_idx: int
    wall_distance_m: float

@dataclass
class PlacedItem:
    """A placed 3D furniture item represented in 2D plan space."""
    item_id: str                # Unique instance ID, e.g. "bed_double_0"
    catalog_id: str            # Key in catalog.json, e.g. "bed_double"
    name: str                  # Human-readable name
    category: str              # Category, e.g. "bed", "sofa", "nightstand"
    position_m: Tuple[float, float]   # Base center position [x, y] in plan meters
    rotation_deg: float        # Rotation angle in degrees clockwise from North [0, 360)
    dimensions_m: Tuple[float, float, float]  # [width_x, depth_z, height_y]
    glb: str                   # Asset path, e.g. "/models/furniture/bed_double.glb"
    polygon_coords: List[Tuple[float, float]]  # Footprint polygon vertices in plan meters
    placement_rule: str        # "back_to_wall", "in_front_of_anchor", "center", etc.
    wall_distance_cm: float = 0.0  # Distance from back edge to wall face in cm (target <= 2.0 cm)
    wall_contact_ratio: float = 1.0  # Share of back edge within 0.10m of a wall pixel (target >= 0.80)
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # 3D proportional scale factor (sx, sy, sz)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "catalog_id": self.catalog_id,
            "name": self.name,
            "category": self.category,
            "position_m": [round(value, 3) for value in self.position_m],
            "rotation_deg": round(self.rotation_deg, 1),
            "dimensions_m": [round(value, 3) for value in self.dimensions_m],
            "scale": [round(value, 3) for value in self.scale],
            "glb": self.glb,
            "placement_rule": self.placement_rule,
            "wall_distance_cm": round(self.wall_distance_cm, 1),
            "wall_contact_ratio": round(self.wall_contact_ratio, 3),
            "footprint_polygon": [[round(p[0], 3), round(p[1], 3)] for p in self.polygon_coords],
        }

@dataclass
class ValidationReport:
    """Strict geometric validation report for a furnished room layout."""
    is_valid: bool
    overlap_count: int = 0
    max_overlap_area_m2: float = 0.0
    door_clearance_violation: bool = False
    out_of_bounds_count: int = 0
    max_wall_gap_cm: float = 0.0
    bfs_reachable: bool = True
    failure_reasons: List[str] = field(default_factory=list)

@dataclass
class RoomDesignResult:
    """Complete output of the rule-based room design engine for a single room."""
    room_id: str
    room_type: str
    preset: str
    seed: int
    is_valid: bool
    failure_reason: Optional[str]
    status: Optional[str] = None
    items: List[PlacedItem] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation: Optional[ValidationReport] = None

    def to_dict(self) -> Dict[str, Any]:
        computed_status = self.status
        if not computed_status:
            computed_status = "valid" if self.is_valid else (self.failure_reason or "failed")

        return {
            "room_id": self.room_id,
            "room_type": self.room_type,
            "preset": self.preset,
            "seed": self.seed,
            "is_valid": self.is_valid,
            "status": computed_status,
            "failure_reason": self.failure_reason,
            "items": [item.to_dict() for item in self.items],
            "metadata": self.metadata,
            "validation": {
                "is_valid": self.validation.is_valid if self.validation else self.is_valid,
                "overlap_count": self.validation.overlap_count if self.validation else 0,
                "door_clearance_violation": self.validation.door_clearance_violation if self.validation else False,
                "out_of_bounds_count": self.validation.out_of_bounds_count if self.validation else 0,
                "max_wall_gap_cm": round(self.validation.max_wall_gap_cm, 1) if self.validation else 0.0,
                "bfs_reachable": self.validation.bfs_reachable if self.validation else True,
                "failure_reasons": self.validation.failure_reasons if self.validation else []
            } if self.validation else {}
        }
