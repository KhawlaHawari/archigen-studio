"""
src/design/edge_classifier.py
-----------------------------
Snaps room polygon edges to wall faces, detects door/window spans,
and computes unobstructed usable wall segments for furniture placement.
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from shapely.geometry import Polygon, LineString, Point, box
from src.design.types import WallSegment, OpeningSpan, UsableWallSegment

def opening_position(opening: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    for key in ("position_meters", "center_meters", "position_m"):
        coords = opening.get(key)
        if coords is not None and len(coords) >= 2:
            return tuple(coords[:2])
    polygon = opening.get("polygon_meters")
    if polygon:
        return tuple(np.mean(polygon, axis=0)[:2])
    return None

def compute_inward_normal(p1: Tuple[float, float], p2: Tuple[float, float], room_poly: Polygon) -> Tuple[float, float]:
    """
    Computes the unit normal vector to edge (p1 -> p2) pointing INWARD into the room.
    In plan space: North is -Y, South is +Y, East is +X, West is -X.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return (0.0, 1.0)

    # Two possible perpendicular unit normals
    n1 = (-dy / length, dx / length)
    n2 = (dy / length, -dx / length)

    # Test midpoint + small offset epsilon
    mid_x = (p1[0] + p2[0]) / 2.0
    mid_y = (p1[1] + p2[1]) / 2.0
    eps = 0.05  # 5 cm into room

    test_pt1 = Point(mid_x + n1[0] * eps, mid_y + n1[1] * eps)
    if room_poly.contains(test_pt1) or room_poly.touches(test_pt1):
        return n1
    return n2

def normal_to_yaw_deg(nx: float, ny: float) -> float:
    """
    Converts a 2D inward normal (nx, ny) in plan space to a clockwise angle in degrees from North [0, 360).
    North is (0, -1) -> 0 deg
    East is (1, 0)   -> 90 deg
    South is (0, 1)  -> 180 deg
    West is (-1, 0)  -> 270 deg
    """
    # atan2(x, -y) gives clockwise angle from (0, -1)
    rad = math.atan2(nx, -ny)
    deg = math.degrees(rad)
    return (deg + 360.0) % 360.0

class EdgeClassifier:
    """Classifies room edges, snaps to walls, and extracts usable segments."""

    def __init__(self, wall_tolerance_m: float = 0.15, door_snap_tolerance_m: float = 0.40):
        self.wall_tolerance_m = wall_tolerance_m
        self.door_snap_tolerance_m = door_snap_tolerance_m

    @staticmethod
    def rasterize_wall_segments(
        wall_segments: List[Dict[str, Any]],
        scale_m_per_px: float = 0.03,
        shape: Tuple[int, int] = (512, 512)
    ) -> np.ndarray:
        """
        Rasterizes vector wall polygons to a binary wall mask raster (Method B standard).
        Ensures Method B raster distance transform is the default everywhere, even when
        input consists of JSON polygon definitions.
        """
        import cv2
        mask = np.zeros(shape, dtype=np.uint8)
        for w in wall_segments:
            pts_px = w.get("polygon_px", [])
            if len(pts_px) >= 3:
                cv2.fillPoly(mask, [np.array(pts_px, dtype=np.int32)], 1)
            else:
                pts_m = w.get("polygon_meters", [])
                if len(pts_m) >= 3:
                    pts_px = [
                        [int(round(p[0] / max(scale_m_per_px, 1e-4))), int(round(p[1] / max(scale_m_per_px, 1e-4)))]
                        for p in pts_m
                    ]
                    cv2.fillPoly(mask, [np.array(pts_px, dtype=np.int32)], 1)
        return mask

    def classify_and_snap_edges(
        self,
        room_poly: Polygon,
        wall_mask: Optional[np.ndarray] = None,
        scale_m_per_px: float = 0.03,
        wall_segments: Optional[List[Dict[str, Any]]] = None,
        doors: Optional[List[Dict[str, Any]]] = None,
        windows: Optional[List[Dict[str, Any]]] = None,
        force_furnish: bool = False
    ) -> Tuple[Polygon, List[UsableWallSegment], List[Polygon]]:
        """
        Takes raw room polygon and wall evidence.
        Returns:
          1. Snapped room polygon (flush against wall inner face)
          2. List of unobstructed UsableWallSegments (with door/window cutouts subtracted)
          3. List of door clearance zone polygons (reserved areas that must stay clear)
        """
        pts = list(room_poly.exterior.coords)[:-1]
        num_pts = len(pts)
        if num_pts < 3:
            return room_poly, [], []

        # 1. Classify each edge: wall-backed vs open
        edge_info = []
        door_clearance_zones = []

        # Method B default: if wall_mask is None but wall_segments is provided, rasterize to binary wall mask
        if wall_mask is None and wall_segments:
            wall_mask = self.rasterize_wall_segments(wall_segments, scale_m_per_px)

        # Precompute distance transform if wall_mask is provided
        dist_map_m = None
        if wall_mask is not None and wall_mask.max() > 0:
            import scipy.ndimage as ndi
            wall_bin = (wall_mask > 0).astype(np.uint8)
            dist_px = ndi.distance_transform_edt(1 - wall_bin)
            dist_map_m = dist_px * scale_m_per_px

        snapped_pts = []

        for i in range(num_pts):
            p1 = pts[i]
            p2 = pts[(i + 1) % num_pts]
            edge = LineString([p1, p2])
            edge_len = edge.length

            # Measure distance to nearest wall
            med_dist = self._measure_edge_wall_distance(edge, dist_map_m, scale_m_per_px, wall_segments)
            is_wall_backed = (med_dist <= self.wall_tolerance_m)

            inward_n = compute_inward_normal(p1, p2, room_poly)
            yaw_deg = normal_to_yaw_deg(inward_n[0], inward_n[1])

            edge_info.append({
                "edge_idx": i,
                "p1": p1,
                "p2": p2,
                "length_m": edge_len,
                "is_wall_backed": is_wall_backed,
                "wall_dist_m": med_dist,
                "inward_normal": inward_n,
                "yaw_deg": yaw_deg,
                "openings": []
            })
            snapped_pts.append(p1)

        snapped_room_poly = room_poly
        walls_by_id = {wall.get("id"): wall for wall in (wall_segments or []) if wall.get("id")}

        def locate_opening(opening):
            position = opening_position(opening)
            if position is None:
                return None, 0.0
            wall = walls_by_id.get(opening.get("associated_wall"), {})
            idx, distance, projection = self._find_nearest_edge(position, edge_info, wall.get("axis_meters"))
            tolerance = self.door_snap_tolerance_m
            if wall.get("thickness_m"):
                tolerance = min(tolerance, wall["thickness_m"] / 2 + .20)
            return (idx, projection) if distance <= tolerance else (None, 0.0)

        # 2. Snap doors onto edges & generate door clearance zones
        if doors:
            for door in doors:
                best_edge_idx, best_proj = locate_opening(door)
                if best_edge_idx is not None:
                    door_width_m = door.get("width_m") or 0.90
                    e = edge_info[best_edge_idx]
                    e_len = e["length_m"]

                    # Project door span along edge
                    start_s = max(0.0, best_proj - door_width_m / 2.0)
                    end_s = min(e_len, best_proj + door_width_m / 2.0)

                    # Determine door clearance depth (adapt for compact rooms)
                    min_room_dim = min(room_poly.bounds[2] - room_poly.bounds[0], room_poly.bounds[3] - room_poly.bounds[1])
                    door_depth = 0.85
                    if force_furnish or room_poly.area < 2.5 or min_room_dim < 1.5:
                        door_depth = float(np.clip(min_room_dim * 0.35, 0.25, 0.85))

                    e["openings"].append(OpeningSpan(
                        opening_type="door",
                        start_dist=start_s,
                        end_dist=end_s,
                        clearance_depth=door_depth
                    ))

                    # Construct 2D door clearance polygon (swing box)
                    cz = self._make_opening_clearance_polygon(
                        e["p1"], e["p2"], start_s, end_s, e["inward_normal"], depth_m=door_depth
                    )
                    if cz is not None and cz.is_valid:
                        door_clearance_zones.append(cz)

        # 3. Snap windows onto edges
        if windows:
            for win in windows:
                best_edge_idx, best_proj = locate_opening(win)
                if best_edge_idx is not None:
                    win_width_m = win.get("width_m") or 1.20
                    e = edge_info[best_edge_idx]
                    e_len = e["length_m"]

                    start_s = max(0.0, best_proj - win_width_m / 2.0)
                    end_s = min(e_len, best_proj + win_width_m / 2.0)

                    e["openings"].append(OpeningSpan(
                        opening_type="window",
                        start_dist=start_s,
                        end_dist=end_s,
                        clearance_depth=0.10
                    ))

        # 3.5 Fallback for enclosed room partitions
        # If wall_segments evidence omitted interior partitions (resulting in 0 wall-backed edges),
        # treat the room boundary edges as partition wall faces.
        if not any(e["is_wall_backed"] for e in edge_info):
            for e in edge_info:
                e["is_wall_backed"] = True
                e["wall_dist_m"] = 0.05

        # 4. Extract continuous unobstructed usable wall segments
        usable_segments = []
        for e in edge_info:
            if not e["is_wall_backed"]:
                continue

            e_len = e["length_m"]
            p1 = np.array(e["p1"])
            p2 = np.array(e["p2"])
            v_dir = (p2 - p1) / max(e_len, 1e-6)

            # Sort openings along edge
            openings = sorted(e["openings"], key=lambda o: o.start_dist)

            if not openings:
                # Full edge is unobstructed
                usable_segments.append(UsableWallSegment(
                    start=tuple(p1),
                    end=tuple(p2),
                    length_m=round(float(e_len), 3),
                    normal=e["inward_normal"],
                    yaw_deg=e["yaw_deg"],
                    edge_idx=e["edge_idx"],
                    wall_distance_m=round(e["wall_dist_m"], 3)
                ))
            else:
                # Split edge around openings
                curr_s = 0.0
                for op in openings:
                    seg_len = op.start_dist - curr_s
                    if seg_len >= 0.50:  # Minimum 50cm to be usable for any item
                        seg_p1 = p1 + v_dir * curr_s
                        seg_p2 = p1 + v_dir * op.start_dist
                        usable_segments.append(UsableWallSegment(
                            start=(float(seg_p1[0]), float(seg_p1[1])),
                            end=(float(seg_p2[0]), float(seg_p2[1])),
                            length_m=round(float(seg_len), 3),
                            normal=e["inward_normal"],
                            yaw_deg=e["yaw_deg"],
                            edge_idx=e["edge_idx"],
                            wall_distance_m=round(e["wall_dist_m"], 3)
                        ))
                    curr_s = max(curr_s, op.end_dist)

                # Trailing segment
                if (e_len - curr_s) >= 0.50:
                    seg_p1 = p1 + v_dir * curr_s
                    seg_p2 = p2
                    usable_segments.append(UsableWallSegment(
                        start=(float(seg_p1[0]), float(seg_p1[1])),
                        end=(float(seg_p2[0]), float(seg_p2[1])),
                        length_m=round(float(e_len - curr_s), 3),
                        normal=e["inward_normal"],
                        yaw_deg=e["yaw_deg"],
                        edge_idx=e["edge_idx"],
                        wall_distance_m=round(e["wall_dist_m"], 3)
                    ))

        return snapped_room_poly, usable_segments, door_clearance_zones

    def _measure_edge_wall_distance(
        self,
        edge: LineString,
        dist_map_m: Optional[np.ndarray],
        scale_m: float,
        wall_segments: Optional[List[Dict[str, Any]]]
    ) -> float:
        """Samples points along edge to compute robust median distance to wall."""
        if dist_map_m is not None:
            h, w = dist_map_m.shape
            dists = []
            for frac in [0.2, 0.4, 0.5, 0.6, 0.8]:
                pt = edge.interpolate(frac, normalized=True)
                ix = int(round(pt.x / scale_m))
                iy = int(round(pt.y / scale_m))
                if 0 <= ix < w and 0 <= iy < h:
                    dists.append(float(dist_map_m[iy, ix]))
            if dists:
                return float(np.median(dists))

        if wall_segments:
            # Measure to list of wall segments / polygons
            dists = []
            for frac in [0.2, 0.4, 0.5, 0.6, 0.8]:
                pt = edge.interpolate(frac, normalized=True)
                min_d = 999.0
                for ws in wall_segments:
                    pts = ws.get("polygon_meters", [])
                    if len(pts) >= 3:
                        poly = Polygon(pts)
                        if poly.is_valid:
                            min_d = min(min_d, poly.distance(pt))
                dists.append(min_d)
            if dists:
                return float(np.median(dists))

        return 0.05  # Default assumed close if no evidence

    def _find_nearest_edge(self, pt: Tuple[float, float], edges: List[Dict[str, Any]], wall_axis=None) -> Tuple[Optional[int], float, float]:
        p = Point(pt)
        best_idx = None
        best_dist = 999.0
        best_proj = 0.0
        axis = np.subtract(wall_axis[1], wall_axis[0]) if wall_axis else None
        axis_length = np.linalg.norm(axis) if axis is not None else 0

        for idx, e in enumerate(edges):
            ls = LineString([e["p1"], e["p2"]])
            if axis_length > .01 and ls.length > .01:
                direction = np.subtract(e["p2"], e["p1"]) / ls.length
                along = np.dot(np.subtract(pt, e["p1"]), direction)
                if abs(np.dot(direction, axis) / axis_length) < .93 or not -.15 <= along <= ls.length + .15:
                    continue
            d = ls.distance(p)
            if d < best_dist:
                best_dist = d
                best_idx = idx
                best_proj = ls.project(p)

        return best_idx, best_dist, best_proj

    def _make_opening_clearance_polygon(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        start_s: float,
        end_s: float,
        inward_n: Tuple[float, float],
        depth_m: float = 0.85
    ) -> Optional[Polygon]:
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return None

        ux = dx / length
        uy = dy / length

        # 4 corners of clearance box projecting inward
        c1 = (p1[0] + ux * start_s, p1[1] + uy * start_s)
        c2 = (p1[0] + ux * end_s, p1[1] + uy * end_s)
        c3 = (c2[0] + inward_n[0] * depth_m, c2[1] + inward_n[1] * depth_m)
        c4 = (c1[0] + inward_n[0] * depth_m, c1[1] + inward_n[1] * depth_m)

        try:
            return Polygon([c1, c2, c3, c4])
        except Exception:
            return None
