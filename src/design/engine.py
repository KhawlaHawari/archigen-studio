"""
src/design/engine.py
--------------------
Orchestrates the deterministic rule-based room design placement engine:
1. Loads catalog.json
2. Snaps edges and extracts usable wall segments
3. Applies room-specific furnishing rules
4. Strictly validates spatial constraints
5. Returns plan coordinates (meters) + rotation (degrees clockwise from North)
"""

import os
import json
import time
import math
import numpy as np
from typing import List, Dict, Any, Optional
from shapely.geometry import LineString, Polygon

from src.design.types import PlacedItem, RoomDesignResult, MIN_PLAUSIBLE_ROOM_AREAS
from src.design.edge_classifier import EdgeClassifier
from src.design.rules import RoomRuleEngine, compute_footprint_polygon
from src.design.validator import LayoutValidator

CATALOG_DEFAULT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "catalog.json"))

class RoomDesignEngine:
    """Main entrypoint for offline deterministic room design synthesis."""

    ITEM_WEIGHTS = {
        "sofa": 6, "bed": 6, "sideboard": 3, "wardrobe": 3,
        "bathroom_sink": 3, "kitchen_counter": 3, "refrigerator": 3,
        "office_chair": 3, "dining_chair": 2, "coffee_table": 2,
        "tv_unit": 2, "armchair": 1.5, "lamp": .75, "rug": .5,
    }

    def __init__(self, catalog_path: str = CATALOG_DEFAULT_PATH):
        if not os.path.exists(catalog_path):
            raise FileNotFoundError(f"Catalog file not found at: {catalog_path}")
        with open(catalog_path, "r", encoding="utf-8") as f:
            self.catalog = json.load(f)

        self.classifier = EdgeClassifier(wall_tolerance_m=0.15, door_snap_tolerance_m=0.40)
        self.rules = RoomRuleEngine(self.catalog)
        self.validator = LayoutValidator(wall_contact_tolerance_m=0.02, min_corridor_width_m=0.50)

    def design_room(
        self,
        room_polygon_meters: List[List[float]],
        room_type: str,
        room_id: str = "room_0",
        wall_mask: Optional[np.ndarray] = None,
        scale_m_per_px: float = 0.03,
        wall_segments: Optional[List[Dict[str, Any]]] = None,
        doors: Optional[List[Dict[str, Any]]] = None,
        windows: Optional[List[Dict[str, Any]]] = None,
        preset: str = "contemporary_minimal",
        seed: int = 42,
        is_scale_uncalibrated: bool = False,
        force_furnish: bool = False
    ) -> RoomDesignResult:
        """
        Designs a furnished layout for a single room.
        All coordinates returned are in 2D plan meters.
        """
        t0 = time.perf_counter()

        # Handle preset aliases
        aliases = self.catalog.get("preset_aliases", {})
        norm_preset = aliases.get(preset, preset)
        if norm_preset not in self.catalog.get("preset_support", {}):
            norm_preset = "contemporary_minimal"

        if len(room_polygon_meters) < 3:
            return RoomDesignResult(
                room_id=room_id,
                room_type=room_type,
                preset=norm_preset,
                seed=seed,
                is_valid=False,
                failure_reason="INVALID_POLYGON",
                metadata={"runtime_ms": 0.0}
            )

        room_poly = Polygon(room_polygon_meters)
        if not room_poly.is_valid:
            room_poly = room_poly.buffer(0)
        if room_poly.is_empty or not isinstance(room_poly, Polygon):
            return RoomDesignResult(room_id, room_type, norm_preset, seed, False, "INVALID_POLYGON")

        # Compute room dimensions
        minx, miny, maxx, maxy = room_poly.bounds
        room_width = round(maxx - minx, 2)
        room_depth = round(maxy - miny, 2)
        room_area = round(room_poly.area, 2)

        # Check minimum plausible room area (exclude neural over-segmentation fragments)
        rule_type = self.rules._normalize_room_type(room_type)
        norm_type = "living_room" if rule_type == "living" else rule_type
        min_plausible_area = MIN_PLAUSIBLE_ROOM_AREAS.get(norm_type, 1.50)
        if room_area < min_plausible_area and not force_furnish:
            runtime_ms = (time.perf_counter() - t0) * 1000.0
            return RoomDesignResult(
                room_id=room_id,
                room_type=room_type,
                preset=norm_preset,
                seed=seed,
                is_valid=False,
                failure_reason="ROOM_FRAGMENT",
                status="not a room (fragment)",
                items=[],
                metadata={
                    "room_width_m": room_width,
                    "room_depth_m": room_depth,
                    "room_area_m2": room_area,
                    "min_plausible_area_m2": min_plausible_area,
                    "is_fragment": True,
                    "room_status": "not a room (fragment)",
                    "usable_wall_segments_count": 0,
                    "scale_m_per_px": round(scale_m_per_px, 4),
                    "scale_source": "2-Pt Calibrated" if not is_scale_uncalibrated else "Model Estimated",
                    "scale_uncalibrated": is_scale_uncalibrated,
                    "runtime_ms": round(runtime_ms, 2)
                }
            )

        snapped_poly, usable_segs, door_zones, dist_map_m = self._room_geometry(
            room_poly, wall_segments, doors, windows, scale_m_per_px, force_furnish, wall_mask,
        )

        # 2. Try several anchor walls and rank the layouts that pass validation.
        placed_items, val_report, fail_reason, option_count = self._choose_room_layout(
            room_type, rule_type, snapped_poly, usable_segs, door_zones,
            norm_preset, seed, dist_map_m, scale_m_per_px, force_furnish,
        )

        if fail_reason:
            runtime_ms = (time.perf_counter() - t0) * 1000.0
            return RoomDesignResult(
                room_id=room_id,
                room_type=room_type,
                preset=norm_preset,
                seed=seed,
                is_valid=False,
                failure_reason=fail_reason,
                items=[],
                metadata={
                    "room_width_m": room_width,
                    "room_depth_m": room_depth,
                    "room_area_m2": room_area,
                    "usable_wall_segments_count": len(usable_segs),
                    "scale_m_per_px": round(scale_m_per_px, 4),
                    "scale_source": "2-Pt Calibrated" if not is_scale_uncalibrated else "Model Estimated",
                    "scale_uncalibrated": is_scale_uncalibrated,
                    "runtime_ms": round(runtime_ms, 2)
                }
            )

        runtime_ms = (time.perf_counter() - t0) * 1000.0

        return RoomDesignResult(
            room_id=room_id,
            room_type=room_type,
            preset=norm_preset,
            seed=seed,
            is_valid=val_report.is_valid,
            failure_reason="VALIDATION_FAILED" if not val_report.is_valid else None,
            items=placed_items,
            metadata={
                "compact_mode": force_furnish,
                "layout_options": option_count,
                "room_width_m": room_width,
                "room_depth_m": room_depth,
                "room_area_m2": room_area,
                "usable_wall_segments_count": len(usable_segs),
                "scale_m_per_px": round(scale_m_per_px, 4),
                "scale_source": "2-Pt Calibrated" if not is_scale_uncalibrated else "Model Estimated",
                "scale_uncalibrated": is_scale_uncalibrated,
                "runtime_ms": round(runtime_ms, 2)
            },
            validation=val_report
        )

    def _choose_room_layout(self, room_type, rule_type, room, segments, door_zones,
                            preset, seed, wall_map, scale_m, compact):
        primary = self.rules.PRIMARY_CATEGORIES.get(rule_type)
        if not primary:
            return [], None, "UNSUPPORTED_ROOM_TYPE", 0
        width = self._corridor_width(rule_type, compact)
        walls = sorted(segments, key=lambda seg: -seg.length_m)
        options, seen = [], set()
        first_invalid = None
        failure = "NO_USABLE_WALL"

        for anchor in walls[:7] or [None]:
            items, reason = self.rules.furnish_room(
                room_type, room, walls, door_zones, preset, seed, wall_map, scale_m,
                force_furnish=compact, preferred_wall=anchor,
            )
            if reason or not any(item.category in primary for item in items):
                failure = reason or failure
                continue
            self.rules.complete_room(room_type, room, walls, door_zones, items, preset, wall_map, scale_m)
            while items:
                # Deduplicate proposals that pick the same furniture and positions.
                key = tuple(sorted((item.catalog_id, *(round(v, 2) for v in item.position_m),
                                    round(item.rotation_deg, 1)) for item in items))
                if key in seen:
                    break
                seen.add(key)
                report = self.validator.validate(room, items, door_zones, wall_dist_map_m=wall_map,
                                                 scale_m_per_px=scale_m, corridor_width_m=width)
                if report.is_valid:
                    score = sum(self.ITEM_WEIGHTS.get(item.category, 1) for item in items)
                    options.append((score, items, report))
                    break
                if first_invalid is None:
                    first_invalid = (items, report)
                optional = [item for item in items if item.category not in primary]
                if report.out_of_bounds_count == 0:
                    optional = [item for item in optional if item.category != "rug"]
                if rule_type == "dining" and sum(item.category == "dining_chair" for item in items) == 1:
                    optional = [item for item in optional if item.category != "dining_chair"]
                if not optional:
                    break
                drop = min(reversed(optional), key=lambda item: self.ITEM_WEIGHTS.get(item.category, 1))
                items = [item for item in items if item is not drop]

        if not options:
            if first_invalid:
                items, report = first_invalid
                return items, report, None, 0
            return [], None, failure, 0
        options.sort(key=lambda entry: -entry[0])
        best = [option for option in options[:3] if option[0] >= options[0][0] - 1]
        _, items, report = best[seed % len(best)]
        return items, report, None, len(best)

    @staticmethod
    def _corridor_width(rule_type, compact):
        return .70 if rule_type == "entry" else .30 if compact else .50

    def _room_geometry(self, room, walls, doors, windows, scale, compact, wall_mask=None):
        if wall_mask is None and walls:
            wall_mask = self.classifier.rasterize_wall_segments(walls, scale)
        distance_map = None
        if wall_mask is not None and wall_mask.max() > 0:
            from scipy.ndimage import distance_transform_edt
            distance_map = distance_transform_edt(wall_mask == 0) * scale
        room, segments, door_zones = self.classifier.classify_and_snap_edges(
            room_poly=room, wall_mask=wall_mask, scale_m_per_px=scale, wall_segments=walls,
            doors=doors, windows=windows, force_furnish=compact,
        )
        return room, segments, door_zones, distance_map

    def furniture_choices(self, room_polygon_meters, room_type, items, item_id, wall_segments=None,
                          doors=None, windows=None, scale_m_per_px=.03, force_furnish=False, custom_catalog=None):
        room = Polygon(room_polygon_meters)
        if not room.is_valid:
            room = room.buffer(0)
        if room.is_empty or not isinstance(room, Polygon):
            raise ValueError("The room needs a valid polygon.")
        catalog = {**(custom_catalog or {}), **self.catalog["items"]}

        def types(spec):
            if spec.get("source") == "custom":
                return spec["types"]
            return ("bed" if spec["category"] == "bed_single" else spec["category"],)

        placed = []
        for item in items:
            spec = catalog.get(item["catalog_id"])
            if spec is None:
                raise ValueError("A placed item is missing from the catalog.")
            slot_type = types(spec)[0]
            if spec.get("source") == "custom" and item.get("category") in types(spec):
                slot_type = item["category"]
            placed.append(PlacedItem(item["item_id"], item["catalog_id"], spec["name"], slot_type,
                                     item["position_m"], item["rotation_deg"], item["dimensions_m"], spec["glb"],
                                     item["footprint_polygon"], item["placement_rule"], scale=item["scale"]))
        original = next((item for item in placed if item.item_id == item_id), None)
        if original is None:
            raise ValueError("The selected furniture is no longer in this room.")
        others = [item for item in placed if item is not original]
        choices = []
        rule_type = self.rules._normalize_room_type(room_type)
        compatible = [(key, spec) for key, spec in catalog.items()
                      if key != original.catalog_id and original.category in types(spec)]
        if not compatible:
            return choices

        room, segments, door_zones, wall_map = self._room_geometry(
            room, wall_segments, doors, windows, scale_m_per_px, force_furnish,
        )
        scale = self.rules.room_scale(rule_type, room, force_furnish)
        minimum = .45 if force_furnish else .82
        scales = sorted({scale, max(minimum, original.scale[0]), max(minimum, round(scale * .92, 3))}, reverse=True)
        yaw = original.rotation_deg
        direction = (math.sin(math.radians(yaw)), -math.cos(math.radians(yaw)))
        anchored = original.placement_rule in {"back_to_wall", "corner_or_wall"}
        for key, spec in compatible:
            for fit in scales:
                w, d, h = (value * fit for value in spec["dimensions_m"])
                shift = (d - original.dimensions_m[1]) / 2 if anchored else 0
                position = tuple(value + axis * shift for value, axis in zip(original.position_m, direction))
                coords = compute_footprint_polygon(*position, w, d, yaw)
                if anchored and w > original.dimensions_m[0] + .01:
                    back = LineString(coords[1:3])
                    if not any(LineString([seg.start, seg.end]).buffer(.035).covers(back) for seg in segments):
                        continue
                candidate = PlacedItem(original.item_id, key, spec["name"], original.category, position, yaw,
                                       (w, d, h), spec["glb"], coords, original.placement_rule, scale=(fit,) * 3)
                report = self.validator.validate(
                    room, [*others, candidate], door_zones, wall_dist_map_m=wall_map, scale_m_per_px=scale_m_per_px,
                    corridor_width_m=self._corridor_width(rule_type, force_furnish),
                )
                if report.is_valid:
                    choice = candidate.to_dict()
                    if spec.get("source") == "custom":
                        choice["source"] = "custom"
                    choices.append(choice)
                    break
        return choices

    def design_layout(
        self,
        layout: Dict[str, Any],
        preset: str = "contemporary_minimal",
        seed: int = 42,
        wall_mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Designs all eligible rooms in a floorplan layout.
        """
        rooms = layout.get("rooms", [])
        doors = layout.get("doors", [])
        windows = layout.get("windows", [])
        walls = layout.get("walls", [])
        scale_m = layout.get("metadata", {}).get("scale_meters_per_pixel", 0.03)

        results = {}
        for r in rooms:
            rid = r.get("id", "room_0")
            rtype = r.get("room_type", "other")
            pts = r.get("polygon_meters", [])
            level = r.get("level_id")

            res = self.design_room(
                room_polygon_meters=pts,
                room_type=rtype,
                room_id=rid,
                wall_mask=wall_mask,
                scale_m_per_px=scale_m,
                wall_segments=[wall for wall in walls if level is None or wall.get("level_id") in (level, None)],
                doors=[door for door in doors if level is None or door.get("level_id") in (level, None)],
                windows=[window for window in windows if level is None or window.get("level_id") in (level, None)],
                preset=preset,
                seed=seed
            )
            results[rid] = res.to_dict()

        return {
            "version": "1.0.0",
            "preset": preset,
            "seed": seed,
            "num_rooms_furnished": sum(1 for r in results.values() if r["is_valid"] and len(r["items"]) > 0),
            "rooms": results
        }
