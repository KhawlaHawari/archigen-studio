"""
Geometric and 3D vectorization module
"""
from src.geometry.topological_closing import bridge_wall_apertures, isolate_closed_rooms
from src.geometry.vectorizer_3d import (
    mask_to_shapely_polygon,
    extrude_polygon_to_3d_mesh,
    vectorize_floorplan_pipeline
)

__all__ = [
    "bridge_wall_apertures",
    "isolate_closed_rooms",
    "mask_to_shapely_polygon",
    "extrude_polygon_to_3d_mesh",
    "vectorize_floorplan_pipeline",
]
