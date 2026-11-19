"""
src/design/validator.py
-----------------------
Strict geometric validation suite for furnished room layouts:
1. Zero overlaps between furniture items
2. Door clearance zones completely clear
3. All items inside room polygon
4. Back-to-wall items within 2 cm of wall face
5. Connected walking space from doors to furniture access points
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional
from shapely import get_parts
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
from src.design.edge_classifier import opening_position
from src.design.types import PlacedItem, ValidationReport

class LayoutValidator:
    """Performs rigorous spatial verification on a generated room layout."""

    def __init__(
        self,
        wall_contact_tolerance_m: float = 0.02,
        min_corridor_width_m: float = 0.50,
        min_wall_contact_ratio: float = 0.80,
        wall_contact_dist_threshold_m: float = 0.10
    ):
        self.wall_contact_tolerance_m = wall_contact_tolerance_m
        self.min_corridor_width_m = min_corridor_width_m
        self.min_wall_contact_ratio = min_wall_contact_ratio
        self.wall_contact_dist_threshold_m = wall_contact_dist_threshold_m

    def validate(
        self,
        room_poly: Polygon,
        items: List[PlacedItem],
        door_clearance_zones: List[Polygon],
        doors: Optional[List[Dict[str, Any]]] = None,
        wall_dist_map_m: Optional[np.ndarray] = None,
        scale_m_per_px: float = 0.03,
        corridor_width_m: Optional[float] = None,
    ) -> ValidationReport:
        """
        Validates all constraints on the furnished layout.
        Returns a ValidationReport with complete pass/fail diagnostics.
        """
        reasons = []
        is_valid = True

        if not items:
            return ValidationReport(is_valid=True, failure_reasons=[])

        # 1. Zero Overlaps between furniture items
        overlap_count = 0
        max_overlap_area = 0.0
        n_items = len(items)

        item_polys = []
        for it in items:
            p = Polygon(it.polygon_coords)
            if not p.is_valid:
                p = p.buffer(0)
            item_polys.append(p)

        for i in range(n_items):
            for j in range(i + 1, n_items):
                # Small exception: rug is an under-anchor item placed under sofa/table
                if items[i].category == "rug" or items[j].category == "rug":
                    continue

                if item_polys[i].intersects(item_polys[j]):
                    inter = item_polys[i].intersection(item_polys[j])
                    area = inter.area
                    if area > 1e-4:  # > 1 cm²
                        overlap_count += 1
                        max_overlap_area = max(max_overlap_area, area)
                        reasons.append(f"OVERLAP: {items[i].item_id} overlaps {items[j].item_id} by {area:.4f} m²")

        if overlap_count > 0:
            is_valid = False

        # 2. Door clearance zones completely clear
        door_violation = False
        for dz in door_clearance_zones:
            if not dz or not dz.is_valid:
                continue
            for idx, p in enumerate(item_polys):
                if items[idx].category == "rug":
                    continue  # Rugs flat on floor may be walked on
                if dz.intersects(p):
                    inter = dz.intersection(p)
                    if inter.area > 1e-4:
                        door_violation = True
                        reasons.append(f"DOOR_BLOCKED: {items[idx].item_id} encroaches into door clear zone by {inter.area:.4f} m²")

        if door_violation:
            is_valid = False

        # 3. Items inside the room polygon
        out_of_bounds_count = 0
        # Allow small numerical buffer of 2cm for flush wall placement
        buffered_room = room_poly.buffer(0.025)

        for idx, p in enumerate(item_polys):
            diff = p.difference(buffered_room)
            if diff.area > 0.01:  # More than 100 cm² outside
                out_of_bounds_count += 1
                reasons.append(f"OUT_OF_BOUNDS: {items[idx].item_id} protrudes {diff.area:.4f} m² outside room")

        if out_of_bounds_count > 0:
            is_valid = False

        # 4. Back-to-wall contact (within 2 cm of wall face)
        max_wall_gap_cm = 0.0
        room_boundary = room_poly.exterior

        for it in items:
            if it.placement_rule == "back_to_wall":
                # Back edge distance to room boundary
                p = Polygon(it.polygon_coords)
                d = room_boundary.distance(p)
                d_cm = d * 100.0
                it.wall_distance_cm = d_cm
                max_wall_gap_cm = max(max_wall_gap_cm, d_cm)
                if d > (self.wall_contact_tolerance_m + 0.015):  # > 3.5 cm
                    reasons.append(f"WALL_GAP: {it.item_id} back is {d_cm:.1f} cm from wall face (limit 2.0 cm)")
                    is_valid = False

                # Strict wall raster contact check: >= 80% within 0.10 m (measured on predicted wall mask)
                if wall_dist_map_m is not None and len(it.polygon_coords) >= 4:
                    c_br = it.polygon_coords[1]
                    c_bl = it.polygon_coords[2]
                    back_line = LineString([c_br, c_bl])
                    if back_line.length > 1e-4:
                        pts_sample = [back_line.interpolate(frac, normalized=True) for frac in np.linspace(0.05, 0.95, 15)]
                        h, w = wall_dist_map_m.shape
                        contact_count = 0
                        for pt in pts_sample:
                            ix = int(round(pt.x / scale_m_per_px))
                            iy = int(round(pt.y / scale_m_per_px))
                            if 0 <= ix < w and 0 <= iy < h:
                                if wall_dist_map_m[iy, ix] <= self.wall_contact_dist_threshold_m:
                                    contact_count += 1
                        ratio = contact_count / len(pts_sample)
                        it.wall_contact_ratio = round(ratio, 3)
                        if ratio < self.min_wall_contact_ratio:
                            reasons.append(
                                f"WALL_CONTACT_VIOLATION: {it.item_id} has only {ratio*100:.1f}% "
                                f"of back edge within {self.wall_contact_dist_threshold_m*100:.0f}cm of wall pixel (minimum required: {self.min_wall_contact_ratio*100:.0f}%)"
                            )
                            is_valid = False

        # 5. Walking reachability from door to usable sides of the furniture
        reachable = self._check_reachability(
            room_poly, items, item_polys, door_clearance_zones, doors,
            corridor_width_m or self.min_corridor_width_m,
        )
        if not reachable:
            is_valid = False
            reasons.append("NO_PATH: Placed furniture blocks walkable path from entrance door to room items")

        return ValidationReport(
            is_valid=is_valid,
            overlap_count=overlap_count,
            max_overlap_area_m2=max_overlap_area,
            door_clearance_violation=door_violation,
            out_of_bounds_count=out_of_bounds_count,
            max_wall_gap_cm=max_wall_gap_cm,
            bfs_reachable=reachable,
            failure_reasons=reasons
        )

    def _check_reachability(self, room, items, footprints, door_zones, doors, width):
        # A centimeter of tolerance keeps a nominal half-meter aisle connected
        # after polygon and model dimensions have been rounded.
        radius = width / 2 - .005
        obstacles = unary_union([poly for item, poly in zip(items, footprints) if item.category != "rug"])
        free = room.buffer(-radius).difference(obstacles.buffer(radius))
        components = [part for part in get_parts(free) if part.area > .001]
        entrances = door_zones
        if not entrances and doors:
            for door in doors:
                position = opening_position(door)
                if position and room.boundary.distance(Point(position)) <= .4:
                    entrances = [*entrances, Point(position).buffer(radius + .45)]

        access_points = []
        has_desk_chair = any(item.category == "office_chair" for item in items)
        has_dining_chair = any(item.category == "dining_chair" for item in items)
        for item in items:
            if item.category in {"rug", "plant", "lamp", "coffee_table", "nightstand"}:
                continue
            if item.category == "desk" and has_desk_chair:
                # The chair occupies the working position at the desk.
                continue
            if item.category == "dining_table" and has_dining_chair:
                continue
            access_points.append(self._access_zone(item, radius))

        return any(
            all(component.intersection(zone).area > .001 for zone in entrances)
            and all(component.intersects(zone) for zone in access_points)
            for component in components
        )

    @staticmethod
    def _access_zone(item, radius):
        """Space where a person can stand to use the furniture."""
        x, z = item.position_m
        width, depth, _ = item.dimensions_m
        angle = math.radians(item.rotation_deg)
        forward = (math.sin(angle), -math.cos(angle))
        lateral = (math.cos(angle), math.sin(angle))
        front = (forward, lateral, depth, width)
        left = ((-lateral[0], -lateral[1]), forward, width, depth)
        right = (lateral, forward, width, depth)
        back = ((-forward[0], -forward[1]), lateral, depth, width)
        if item.category in {"dining_chair", "office_chair"}:
            sides = (left, right, back)
        elif item.category in {"bed", "dining_table", "bathtub"}:
            sides = (front, left, right)
        else:
            sides = (front,)
        zones = []
        for normal, tangent, extent, span in sides:
            offset = extent / 2 + radius + .05
            center = (x + normal[0] * offset, z + normal[1] * offset)
            half = span * .38
            zones.append(LineString([
                (center[0] - tangent[0] * half, center[1] - tangent[1] * half),
                (center[0] + tangent[0] * half, center[1] + tangent[1] * half),
            ]).buffer(.07))
        return unary_union(zones)
