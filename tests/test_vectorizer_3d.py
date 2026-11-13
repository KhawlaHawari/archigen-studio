"""
Unit tests for Topological Closing & 3D Extrusion Engine (OS 4)
"""

import numpy as np
import pytest
from src.geometry.topological_closing import bridge_wall_apertures, isolate_closed_rooms
from src.geometry.vectorizer_3d import (
    mask_to_shapely_polygon,
    extrude_polygon_to_3d_mesh,
    vectorize_floorplan_pipeline
)
from src.data.svg_preprocessor import generate_synthetic_cubicasa_sample


def test_topological_bridging():
    """Tests that door gaps in perimeter walls are bridged to prevent room bleeding"""
    # Create a 100x100 wall with a 10px door gap
    struct_mask = np.zeros((100, 100), dtype=np.uint8)
    struct_mask[10:15, 10:90] = 1  # Top wall
    struct_mask[10:15, 40:50] = 3  # Door in top wall

    barrier = bridge_wall_apertures(struct_mask, door_class=3, kernel_size=7)
    # Check that door gap is closed in barrier
    assert barrier[12, 45] == 1, "Door gap should be bridged by morphological closing"


def test_polygon_extrusion_and_buffers():
    """Tests converting raster mask to 3D buffer geometry"""
    _, struct_mask, room_mask = generate_synthetic_cubicasa_sample(256, 256)
    
    pipeline_res = vectorize_floorplan_pipeline(struct_mask, room_mask, scale_m_per_px=0.05)
    
    assert pipeline_res["total_habitable_area_m2"] > 0
    assert len(pipeline_res["rooms"]) > 0
    assert "exterior_walls" in pipeline_res["meshes"]

    ext_mesh = pipeline_res["meshes"]["exterior_walls"]
    assert ext_mesh is not None
    assert len(ext_mesh["vertices"]) > 0
    assert len(ext_mesh["indices"]) > 0
    # Vertices count must be multiple of 3
    assert len(ext_mesh["vertices"]) % 3 == 0
    # Indices count must be multiple of 3 (triangles)
    assert len(ext_mesh["indices"]) % 3 == 0
