"""
3D Vectorization & Polygonal Extrusion Engine (OS 4)
Transforms 2D raster segmentation predictions into closed Shapely polygons
and generates 3D volumetric meshes ready for WebGL / Three.js rendering.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from scipy import ndimage

from src.config import (
    STRUCTURE_CLASSES,
    STRUCTURE_COLORS,
    ROOM_CLASSES,
    ROOM_COLORS,
    GEOMETRY_CFG
)
from src.geometry.topological_closing import bridge_wall_apertures, isolate_closed_rooms


def extract_contours_from_binary_mask(binary_mask: np.ndarray) -> List[List[Tuple[float, float]]]:
    """
    Extracts outer boundary polylines from a 2D binary mask.
    Returns list of contour coordinate loops [[(x0, y0), (x1, y1), ...], ...].
    """
    contours = []
    # Find boundary pixels via morphological gradient
    eroded = ndimage.binary_erosion(binary_mask)
    boundary = (binary_mask.astype(bool) ^ eroded).astype(np.uint8)

    # Label distinct connected components
    labeled, num_features = ndimage.label(binary_mask)
    for comp_idx in range(1, num_features + 1):
        comp_mask = (labeled == comp_idx).astype(np.uint8)
        if np.sum(comp_mask) < 30:  # Skip tiny stray noise
            continue
        
        # Extract ordered perimeter via boundary tracing
        comp_boundary = comp_mask ^ ndimage.binary_erosion(comp_mask)
        pts = np.argwhere(comp_boundary)  # [y, x]
        if len(pts) < 4:
            continue
        
        # Sort points in approximate perimeter order (convex hull / angular sort)
        cy, cx = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 0] - cy, pts[:, 1] - cx)
        sort_idx = np.argsort(angles)
        sorted_pts = pts[sort_idx]
        
        # Downsample contour points for clean polyline representation
        step = max(1, len(sorted_pts) // 32)
        sampled_pts = sorted_pts[::step]
        # Return as (x, y) coordinates
        poly_coords = [(float(pt[1]), float(pt[0])) for pt in sampled_pts]
        if len(poly_coords) >= 3:
            poly_coords.append(poly_coords[0])  # Close loop
            contours.append(poly_coords)

    return contours


def mask_to_shapely_polygon(
    binary_mask: np.ndarray,
    scale: float = 0.05,
    simplify_tol: float = 1.0
) -> Optional[Polygon]:
    """Converts a binary raster mask into a clean, simplified Shapely Polygon in metric space."""
    contours = extract_contours_from_binary_mask(binary_mask)
    if not contours:
        return None

    polys = []
    for loop in contours:
        if len(loop) < 4:
            continue
        # Scale to metric units (meters) and flip Y for 3D coordinate space
        metric_loop = [(x * scale, -y * scale) for x, y in loop]
        try:
            poly = Polygon(metric_loop)
            if poly.is_valid and poly.area > 0.05:
                polys.append(poly)
            elif not poly.is_valid:
                fixed = poly.buffer(0)
                if fixed.is_valid and fixed.area > 0.05:
                    if isinstance(fixed, MultiPolygon):
                        polys.extend([p for p in fixed.geoms if p.area > 0.05])
                    else:
                        polys.append(fixed)
        except Exception:
            continue

    if not polys:
        return None

    unified = unary_union(polys)
    simplified = unified.simplify(simplify_tol * scale, preserve_topology=True)
    return simplified


def extrude_polygon_to_3d_mesh(
    polygon: Polygon,
    height: float = 3.0,
    color_rgb: Tuple[int, int, int] = (200, 200, 200)
) -> Dict[str, Any]:
    """
    Extrudes a 2D planar polygon into a 3D volumetric solid mesh.
    Produces Three.js-compatible BufferGeometry vertex array and face indices.
    """
    if isinstance(polygon, MultiPolygon):
        # Merge individual meshes if multipolygon
        all_verts = []
        all_indices = []
        all_normals = []
        offset = 0
        for sub_poly in polygon.geoms:
            sub_mesh = extrude_polygon_to_3d_mesh(sub_poly, height, color_rgb)
            all_verts.extend(sub_mesh["vertices"])
            all_normals.extend(sub_mesh["normals"])
            all_indices.extend([i + offset for i in sub_mesh["indices"]])
            offset += len(sub_mesh["vertices"]) // 3
        return {
            "vertices": all_verts,
            "indices": all_indices,
            "normals": all_normals,
            "color": [c / 255.0 for c in color_rgb],
            "height": height
        }

    exterior_coords = list(polygon.exterior.coords)
    if exterior_coords[0] == exterior_coords[-1]:
        exterior_coords = exterior_coords[:-1]

    N = len(exterior_coords)
    if N < 3:
        return {"vertices": [], "indices": [], "normals": [], "color": [c / 255.0 for c in color_rgb], "height": height}

    vertices = []
    normals = []
    indices = []

    # 1. Base vertices (z = 0) and Top vertices (z = height)
    # Bottom vertices indices: 0 to N-1
    for x, y in exterior_coords:
        vertices.extend([float(x), float(y), 0.0])
        normals.extend([0.0, 0.0, -1.0])

    # Top vertices indices: N to 2N-1
    for x, y in exterior_coords:
        vertices.extend([float(x), float(y), float(height)])
        normals.extend([0.0, 0.0, 1.0])

    # 2. Side Quad Walls (each side is two triangles)
    side_vert_offset = 2 * N
    for i in range(N):
        next_i = (i + 1) % N
        p0_bottom = exterior_coords[i]
        p1_bottom = exterior_coords[next_i]
        
        # Calculate outward horizontal normal
        dx = p1_bottom[0] - p0_bottom[0]
        dy = p1_bottom[1] - p0_bottom[1]
        length = max(1e-6, np.hypot(dx, dy))
        nx = -dy / length
        ny = dx / length

        # 4 vertices for side quad
        v_idx = len(vertices) // 3
        # Quad corners: bottom_curr, bottom_next, top_next, top_curr
        vertices.extend([p0_bottom[0], p0_bottom[1], 0.0])
        vertices.extend([p1_bottom[0], p1_bottom[1], 0.0])
        vertices.extend([p1_bottom[0], p1_bottom[1], float(height)])
        vertices.extend([p0_bottom[0], p0_bottom[1], float(height)])

        for _ in range(4):
            normals.extend([float(nx), float(ny), 0.0])

        # Triangles for quad
        indices.extend([v_idx, v_idx + 1, v_idx + 2])
        indices.extend([v_idx, v_idx + 2, v_idx + 3])

    # 3. Top Face Triangulation (simple fan from first vertex)
    for i in range(1, N - 1):
        indices.extend([N, N + i, N + i + 1])

    # 4. Bottom Face Triangulation (reverse winding)
    for i in range(1, N - 1):
        indices.extend([0, i + 1, i])

    return {
        "vertices": vertices,
        "indices": indices,
        "normals": normals,
        "color": [c / 255.0 for c in color_rgb],
        "height": height
    }


def vectorize_floorplan_pipeline(
    structure_mask: np.ndarray,
    room_mask: np.ndarray,
    scale_m_per_px: float = 0.05
) -> Dict[str, Any]:
    """
    Full end-to-end vectorization & 3D extrusion pipeline.
    
    Returns structured payload with:
    - total_habitable_area_m2
    - room_summary (list of rooms with surfaces in m²)
    - 3D mesh components for walls (exterior and interior)
    - 3D floor slabs for rooms
    """
    # 1. Topological bridging to prevent room leaks
    wall_barrier = bridge_wall_apertures(
        structure_mask,
        door_class=3,
        window_class=4,
        wall_classes=(1, 2),
        kernel_size=GEOMETRY_CFG.closing_kernel_size
    )

    # 2. Constrain rooms inside closed barriers
    cleaned_rooms = isolate_closed_rooms(room_mask, wall_barrier)

    # 3. Vectorize Exterior Walls (Height: 3.0m)
    ext_wall_mask = (structure_mask == 1).astype(np.uint8)
    ext_wall_poly = mask_to_shapely_polygon(ext_wall_mask, scale=scale_m_per_px)
    ext_wall_mesh = (
        extrude_polygon_to_3d_mesh(ext_wall_poly, height=GEOMETRY_CFG.exterior_wall_height, color_rgb=STRUCTURE_COLORS[1])
        if ext_wall_poly and not ext_wall_poly.is_empty else None
    )

    # 4. Vectorize Interior Walls (Height: 2.8m)
    int_wall_mask = (structure_mask == 2).astype(np.uint8)
    int_wall_poly = mask_to_shapely_polygon(int_wall_mask, scale=scale_m_per_px)
    int_wall_mesh = (
        extrude_polygon_to_3d_mesh(int_wall_poly, height=GEOMETRY_CFG.interior_wall_height, color_rgb=STRUCTURE_COLORS[2])
        if int_wall_poly and not int_wall_poly.is_empty else None
    )

    # 5. Vectorize Rooms and calculate surface areas
    room_summary = []
    room_meshes = []
    total_area_m2 = 0.0

    for room_id in range(1, len(ROOM_CLASSES)):
        r_mask = (cleaned_rooms == room_id).astype(np.uint8)
        if np.sum(r_mask) < 40:
            continue
        
        r_poly = mask_to_shapely_polygon(r_mask, scale=scale_m_per_px)
        if r_poly and not r_poly.is_empty:
            area_m2 = round(float(r_poly.area), 2)
            total_area_m2 += area_m2
            room_name = ROOM_CLASSES[room_id]
            color_rgb = ROOM_COLORS.get(room_id, (200, 200, 200))
            
            # Floor slab mesh (extruded by 0.05m to form floor tile)
            floor_mesh = extrude_polygon_to_3d_mesh(r_poly, height=0.05, color_rgb=color_rgb)
            
            room_summary.append({
                "room_id": room_id,
                "name": room_name,
                "area_m2": area_m2,
                "color_rgb": list(color_rgb),
                "color_hex": "#{:02x}{:02x}{:02x}".format(*color_rgb)
            })
            room_meshes.append({
                "room_name": room_name,
                "area_m2": area_m2,
                "mesh": floor_mesh
            })

    # Count detected structural elements
    num_doors = int(ndimage.label(structure_mask == 3)[1])
    num_windows = int(ndimage.label(structure_mask == 4)[1])

    return {
        "total_habitable_area_m2": round(total_area_m2, 2),
        "num_doors_detected": num_doors,
        "num_windows_detected": num_windows,
        "rooms": room_summary,
        "meshes": {
            "exterior_walls": ext_wall_mesh,
            "interior_walls": int_wall_mesh,
            "room_floors": room_meshes,
        }
    }
