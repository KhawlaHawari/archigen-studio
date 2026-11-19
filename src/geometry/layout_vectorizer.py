"""
Ported from the original ArchiGen backend (src/floorplan/vectorizer.py); room ids follow src.config.ROOM_CLASSES.

Geometric Vectorization and Rectilinear Polygon Post-Processing Pipeline.
Converts raw 2D pixel segmentation masks (walls, doors, windows, rooms) into
clean, simplified vector geometry and structured 3D extrusion JSON specifications.
"""

import math
from collections import Counter
from typing import Dict, List, Any, Tuple
import numpy as np
import cv2
from shapely.geometry import Polygon, LineString, Point
from shapely.validation import make_valid
from src.geometry.dimensions import STOREY_HEIGHT_M, DOOR_HEIGHT_M, WINDOW_HEIGHT_M, WINDOW_SILL_M
from src.geometry.wall_reconstruction import solid_stroke_at_opening, structural_support, trace_wall_runs

# Display names for the ArchiGen room head (same order as src.config.ROOM_CLASSES).
ROOM_NAMES = {
    0: "Background", 1: "Living Room", 2: "Bedroom", 3: "Kitchen",
    4: "Bathroom", 5: "Entry / Hall", 6: "Closet", 7: "Balcony / Terrace",
    8: "Dining", 9: "Office", 10: "Storage", 11: "Utility"
}
# Small rooms that may stay below the habitable-room minimum area: bathroom, closet, storage.
SERVICE_ROOM_IDS = (4, 6, 10)

def score_room_polygon(probabilities: np.ndarray, class_id: int, polygon_px) -> float | None:
    """Average the requested semantic class over a room in model canvas coordinates."""
    region = np.zeros(probabilities.shape[1:], dtype=np.uint8)
    cv2.fillPoly(region, [np.rint(polygon_px).astype(np.int32)], 1)
    scores = probabilities[class_id][region > 0]
    if not scores.size:
        return None
    mean_score = float(scores.mean())
    return round(mean_score, 4) if math.isfinite(mean_score) else None

def repair_polygon_to_single_polygon(poly: Polygon, max_area_change_pct: float = 10.0) -> Tuple[Polygon, bool, float]:
    """
    Repairs self-intersecting or invalid polygons using Shapely's make_valid.
    Extracts the dominant single Polygon if a MultiPolygon or GeometryCollection is produced.
    Ensures that the area does not collapse or explode (must remain within max_area_change_pct of original area).
    
    Returns:
        (clean_polygon, was_repaired, area_change_pct)
    """
    if poly is None or poly.is_empty:
        return poly, False, 0.0

    if poly.is_valid:
        return poly, False, 0.0

    orig_area = poly.area
    try:
        repaired = make_valid(poly)
        if repaired.is_empty:
            return poly, False, 0.0

        if repaired.geom_type == "Polygon":
            clean_poly = repaired
        elif repaired.geom_type in ["MultiPolygon", "GeometryCollection"]:
            polys = [g for g in repaired.geoms if g.geom_type == "Polygon" and not g.is_empty]
            if not polys:
                return poly, False, 0.0
            clean_poly = max(polys, key=lambda p: p.area)
        else:
            return poly, False, 0.0

        if not clean_poly.is_valid:
            return poly, False, 0.0

        if orig_area > 1e-4:
            delta_area = abs(clean_poly.area - orig_area) / orig_area * 100.0
            if delta_area <= max_area_change_pct:
                return clean_poly, True, delta_area
            else:
                # Area changed too drastically (> max_area_change_pct collapse/expansion); treat as failed repair
                return poly, False, delta_area
        else:
            # Original geometry had cancelled winding area (e.g. figure-8)
            return clean_poly, True, 0.0
    except Exception:
        return poly, False, 0.0

class FloorplanVectorizer:


    """
    Transforms multi-class floorplan probability masks into vector geometry.
    Classes:
      0: Background / Exterior
      1: Wall
      2: Door
      3: Window
    """
    def __init__(
        self,
        min_wall_area: int = 40,
        polygon_epsilon_ratio: float = 0.015,
        snap_angle_deg_tolerance: float = 12.0,
        default_door_width_m: float = 0.85,
        wall_height_m: float = STOREY_HEIGHT_M,
        min_area_habitable_m2: float = 2.0,
        min_area_service_m2: float = 0.8
    ):
        self.min_wall_area = min_wall_area
        self.polygon_epsilon_ratio = polygon_epsilon_ratio
        self.snap_angle_deg_tolerance = snap_angle_deg_tolerance
        self.default_door_width_m = default_door_width_m
        self.wall_height_m = wall_height_m
        self.min_area_habitable_m2 = min_area_habitable_m2
        self.min_area_service_m2 = min_area_service_m2

    def snap_angle_rectilinear(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
        """Snaps an edge to rectilinear axes (0, 90, 180, 270 deg) if within tolerance."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-4:
            return p2

        angle = math.degrees(math.atan2(dy, dx)) % 360

        # Snap to horizontal (0 / 180 deg)
        if abs(angle - 0) <= self.snap_angle_deg_tolerance or abs(angle - 360) <= self.snap_angle_deg_tolerance:
            return (p1[0] + dist, p1[1])
        elif abs(angle - 180) <= self.snap_angle_deg_tolerance:
            return (p1[0] - dist, p1[1])
        # Snap to vertical (90 / 270 deg)
        elif abs(angle - 90) <= self.snap_angle_deg_tolerance:
            return (p1[0], p1[1] + dist)
        elif abs(angle - 270) <= self.snap_angle_deg_tolerance:
            return (p1[0], p1[1] - dist)

        return p2

    def estimate_scale(self, door_mask: np.ndarray) -> float:
        """
        Estimates meters per pixel scale factor using median door clearance width.
        The door opening clearance corresponds to the longer dimension of the door bounding rectangle.
        Falls back to 0.030 m/pixel (~33 px per meter) if no doors are detected.
        """
        contours, _ = cv2.findContours(door_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        door_widths = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 15:
                continue
            rect = cv2.minAreaRect(cnt)
            w, h = rect[1]
            door_dim = max(w, h)
            # Standard residential door slot in 512x512 plans is between 8 and 60 px
            if 8 <= door_dim <= 60:
                door_widths.append(door_dim)

        if len(door_widths) > 0:
            median_door_px = float(np.median(door_widths))
            return self.default_door_width_m / median_door_px
        return 0.030  # Standard fallback: 1 pixel ~ 3.0 cm

    def vectorize(

        self,
        mask: np.ndarray,
        room_mask: Any = None,
        struct_mask: Any = None,
        scale_override_m_per_px: Any = None,
        room_probabilities: np.ndarray | None = None,
        source_gray: np.ndarray | None = None,
    ) -> Dict[str, Any]:
        """
        Main pipeline: Takes (H, W) class index mask and extracts structured vector layout.
        Supports optional semantic room_mask and struct_mask (with Wall_External).
        room_probabilities contains the semantic class probabilities in (C, H, W) order.
        """
        h, w = mask.shape
        wall_binary = (mask == 1).astype(np.uint8)
        door_binary = (mask == 2).astype(np.uint8)
        window_binary = (mask == 3).astype(np.uint8)
        # Ink is reliable negative evidence on sparse line drawings. Dense scans
        # and textured/color plans continue to use semantic opening evidence.
        source_ink = (source_gray < 140).astype(np.uint8) if source_gray is not None and np.mean(source_gray > 235) >= .90 else None

        # 2. Scale estimation with manual override support
        if scale_override_m_per_px is not None and scale_override_m_per_px > 0:
            scale_m_per_px = float(scale_override_m_per_px)
            scale_source = "user_override"
        else:
            scale_m_per_px = self.estimate_scale(door_binary)
            scale_source = "door_width_estimation"

        # 3. Outer Building Shell (Watertight Exterior Envelope from Filled Footprint)
        # Seal all structural boundaries (walls, doors, windows) with morphological closure
        # so windows and exterior doors do not breach the exterior envelope.
        all_structure = structural_support(wall_binary, door_binary | window_binary)
        runs = [run for run in trace_wall_runs(all_structure, scale_m_per_px, source_ink)
                if run.length * run.thickness >= self.min_wall_area or run.length * scale_m_per_px >= .35]
        if len(runs) > 1:
            bodies = [Polygon(run.outline()) for run in runs]
            runs = [
                run for index, run in enumerate(runs) if run.length * scale_m_per_px >= .75
                or any(index != other and bodies[index].distance(body) * scale_m_per_px <= .15
                       for other, body in enumerate(bodies))
            ]
        # Doors and windows close partitions in plan. Their 3D cutouts are applied later.
        wall_clean = all_structure.copy()
        for run in runs:
            cv2.fillConvexPoly(wall_clean, np.rint(run.outline()).astype(np.int32), 1)
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        sealed_structure = cv2.morphologyEx(all_structure, cv2.MORPH_CLOSE, k_close)

        if room_mask is not None and np.max(room_mask) > 0:
            envelope_raw = ((sealed_structure > 0) | (room_mask > 0)).astype(np.uint8)
        else:
            envelope_raw = sealed_structure.copy()

        # Fill internal courtyard/corridor holes to construct watertight exterior footprint
        env_cnts, _ = cv2.findContours(envelope_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_envelope = np.zeros_like(envelope_raw)
        if env_cnts:
            cv2.drawContours(filled_envelope, env_cnts, -1, 1, -1)

        # Spur pruning: strip high-frequency perimeter whiskers, fences, and thin spurs (< 3px)
        kernel_spur = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        pruned_envelope = cv2.morphologyEx(filled_envelope, cv2.MORPH_OPEN, kernel_spur)
        if pruned_envelope.sum() >= 0.75 * max(filled_envelope.sum(), 1):
            clean_envelope = pruned_envelope
        else:
            clean_envelope = filled_envelope

        footprint_contours, _ = cv2.findContours(clean_envelope, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shell_valid = False
        was_shell_repaired = False
        shell_failure_code = None
        building_perimeter_m = 0.0

        if len(footprint_contours) > 0:
            largest_ext = max(footprint_contours, key=cv2.contourArea)
            if cv2.contourArea(largest_ext) > 100:
                ext_poly_approx = cv2.approxPolyDP(largest_ext, 2.0, True).reshape(-1, 2)
                if len(ext_poly_approx) >= 3:
                    try:
                        raw_shell_poly = Polygon(ext_poly_approx * scale_m_per_px)
                        if raw_shell_poly.is_valid:
                            shell_poly = raw_shell_poly
                            shell_valid = True
                            building_perimeter_m = round(float(shell_poly.length), 2)
                        else:
                            rep_shell_poly, rep_ok, delta_area = repair_polygon_to_single_polygon(raw_shell_poly)
                            if rep_ok and rep_shell_poly.is_valid:
                                shell_poly = rep_shell_poly
                                shell_valid = True
                                was_shell_repaired = True
                                building_perimeter_m = round(float(shell_poly.length), 2)
                            else:
                                shell_poly = raw_shell_poly
                                shell_valid = False
                                building_perimeter_m = round(float(raw_shell_poly.length), 2)
                                if delta_area > 10.0:
                                    shell_failure_code = "REPAIR_REJECTED_AREA_DISTORTION"
                                else:
                                    shell_failure_code = "INVALID_GEOMETRY"
                    except Exception:
                        shell_valid = False
                        shell_failure_code = "INVALID_GEOMETRY"
                        building_perimeter_m = 0.0

        # 4. Export explicit wall axes and strip footprints. Contour holes are
        # room interiors and must never be serialized as independent walls.
        walls = []
        wall_segments = []
        for index, run in enumerate(runs):
            direction = run.direction
            normal = np.array([-direction[1], direction[0]])
            midpoint = (run.start + run.end) / 2
            outline = run.outline()
            if struct_mask is not None:
                patch = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(patch, np.rint(outline).astype(np.int32), 1)
                ext_count = np.count_nonzero((patch > 0) & (struct_mask == 2))
                int_count = np.count_nonzero((patch > 0) & (struct_mask == 1))
                is_exterior = bool(ext_count > int_count)
            else:
                is_exterior = bool(footprint_contours) and abs(cv2.pointPolygonTest(
                    largest_ext, tuple(midpoint), True)) <= run.thickness / 2 + 3

            polygon_meters = np.round(outline * scale_m_per_px, 3).tolist()
            axis_meters = np.round(np.array([run.start, run.end]) * scale_m_per_px, 3).tolist()
            wall = {
                "id": f"wall_{index}",
                "is_exterior": is_exterior,
                "polygon_px": np.round(outline, 2).tolist(),
                "polygon_meters": polygon_meters,
                "axis_meters": axis_meters,
                "thickness_m": round(run.thickness * scale_m_per_px, 3),
                "area_m2": round(run.length * run.thickness * scale_m_per_px ** 2, 2),
                "height_m": self.wall_height_m,
                "openings": [],
            }
            if is_exterior:
                def side_coverage(sign):
                    points = np.linspace(run.start, run.end, 5)[1:-1]
                    points = np.rint(points + sign * normal * (run.thickness / 2 + 5)).astype(int)
                    return sum(clean_envelope[y, x] for x, y in points if 0 <= x < w and 0 <= y < h)

                outward = normal if side_coverage(1) < side_coverage(-1) else -normal
                forward = np.array([outward[1], -outward[0]])
                anchor = run.start if direction @ forward > 0 else run.end
                wall["facade_frame"] = {
                    "origin": ((anchor + outward * run.thickness / 2) * scale_m_per_px).tolist(),
                    "direction": forward.tolist(),
                    "outward": outward.tolist(),
                    "length_m": run.length * scale_m_per_px,
                }
            walls.append(wall)
            wall_segments.append((run, LineString(axis_meters)))

        # 5. Doors & Windows extraction with wall gating, dimensional filtering, and fragment clustering
        def extract_openings(bin_mask: np.ndarray, opening_type: str):
            raw_candidates = []
            cnts, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if cv2.contourArea(c) < 10:
                    continue
                M = cv2.moments(c)
                if M["m00"] == 0:
                    continue
                center_px_x = M["m10"] / M["m00"]
                center_px_y = M["m01"] / M["m00"]
                cx = center_px_x * scale_m_per_px
                cy = center_px_y * scale_m_per_px
                center_pt = Point(cx, cy)

                rect = cv2.minAreaRect(c)
                dim_m = max(rect[1]) * scale_m_per_px

                # Gate 2: Dimension filter (reject fragments < 0.50m or oversized gaps > 2.50m for doors)
                if opening_type == "door":
                    if dim_m < 0.50 or dim_m > 2.50:
                        continue
                    clamped_width = round(max(0.65, min(dim_m, 2.20)), 2)
                else:
                    if dim_m < 0.40 or dim_m > 3.50:
                        continue
                    clamped_width = round(max(0.50, min(dim_m, 3.00)), 2)

                # Prefer a nearby axis with enough wall on each side for the whole opening.
                hosts = []
                for index, (_, axis) in enumerate(wall_segments):
                    distance = axis.distance(center_pt)
                    if distance > .85:
                        continue
                    along = axis.project(center_pt)
                    margin = clamped_width / 2 - .08
                    fits = distance <= .28 and margin <= along <= axis.length - margin
                    hosts.append((not fits, distance, index))
                if not hosts:
                    continue
                _, min_dist, host = min(hosts)
                best_wall_id = walls[host]["id"]
                if source_ink is not None and solid_stroke_at_opening(
                    source_ink, wall_segments[host][0], np.array([center_px_x, center_px_y]),
                    clamped_width / scale_m_per_px,
                ):
                    continue

                raw_candidates.append({
                    "position_px": [round(center_px_x, 1), round(center_px_y, 1)],
                    "position_meters": [round(cx, 3), round(cy, 3)],
                    "width_m": clamped_width,
                    "associated_wall": best_wall_id,
                    "distance_to_wall_m": round(min_dist, 3)
                })

            # Gate 3: Spatial clustering of contiguous/overlapping opening fragments
            clusters = []
            used = set()
            for i, op1 in enumerate(raw_candidates):
                if i in used:
                    continue
                p1 = Point(op1["position_meters"])
                w1 = op1["associated_wall"]
                cluster = [op1]
                used.add(i)
                for j, op2 in enumerate(raw_candidates):
                    if j in used:
                        continue
                    p2 = Point(op2["position_meters"])
                    w2 = op2["associated_wall"]
                    dist = p1.distance(p2)
                    # Merge if on the same wall within 0.65m, or close within 0.50m
                    if (w1 is not None and w1 == w2 and dist <= 0.65) or (dist <= 0.50):
                        cluster.append(op2)
                        used.add(j)
                clusters.append(cluster)

            openings = []
            for op_idx, cl in enumerate(clusters):
                avg_px_x = sum(item["position_px"][0] for item in cl) / len(cl)
                avg_px_y = sum(item["position_px"][1] for item in cl) / len(cl)
                avg_x = sum(item["position_meters"][0] for item in cl) / len(cl)
                avg_y = sum(item["position_meters"][1] for item in cl) / len(cl)
                avg_w = sum(item["width_m"] for item in cl) / len(cl)
                best_wall = cl[0]["associated_wall"]
                min_dw = min(item["distance_to_wall_m"] for item in cl)

                op_data = {
                    "id": f"{opening_type}_{op_idx}",
                    "type": opening_type,
                    "position_px": [round(avg_px_x, 1), round(avg_px_y, 1)],
                    "position_meters": [round(avg_x, 3), round(avg_y, 3)],
                    "width_m": round(avg_w, 2),
                    "height_m": DOOR_HEIGHT_M if opening_type == "door" else WINDOW_HEIGHT_M,
                    "sill_m": 0.0 if opening_type == "door" else WINDOW_SILL_M,
                    "associated_wall": best_wall,
                    "distance_to_wall_m": round(min_dw, 3)
                }
                openings.append(op_data)

                # Attach to wall openings list
                if best_wall is not None and min_dw < 1.0:
                    for w_item in walls:
                        if w_item["id"] == best_wall:
                            w_item["openings"].append(op_data)
                            break

            return openings

        doors = extract_openings(door_binary, "door")
        windows = extract_openings(window_binary, "window")

        # Precompute wall distance transform map for edge classification
        dist_px = cv2.distanceTransform((1 - wall_clean).astype(np.uint8), cv2.DIST_L2, 3)
        wall_dist_map_m = dist_px * scale_m_per_px

        # 6. Extract Room Interior Spaces with Semantic Room Typing
        rooms = []
        room_idx = 0
        total_interior_area_m2 = 0.0
        num_rooms_repaired = 0
        partition_count, partitions, partition_stats, _ = cv2.connectedComponentsWithStats(
            1 - wall_clean, connectivity=4)
        outside_ids = set(np.concatenate([partitions[0], partitions[-1], partitions[:, 0], partitions[:, -1]]))
        min_room_pixels = self.min_area_service_m2 / scale_m_per_px ** 2
        enclosed_ids = [
            index for index in range(1, partition_count) if index not in outside_ids
            and partition_stats[index, cv2.CC_STAT_AREA] >= min_room_pixels
            and min(partition_stats[index, cv2.CC_STAT_WIDTH], partition_stats[index, cv2.CC_STAT_HEIGHT])
            * scale_m_per_px >= .35
        ]

        if room_mask is not None and (np.max(room_mask) > 0 or len(enclosed_ids) >= 2):
            # Hybrid Room Partitioning:
            # 1. Mask room predictions with walls to split same-class adjacent rooms
            clean_room = room_mask.copy()
            clean_room[wall_clean > 0] = 0

            px_area_habitable = int(round(self.min_area_habitable_m2 / (scale_m_per_px ** 2)))
            px_area_service = int(round(self.min_area_service_m2 / (scale_m_per_px ** 2)))

            # 2. Connected components per class
            labeled_map = np.zeros((h, w), dtype=np.int32)
            region_info = {}
            current_label = 1

            # A physically closed chamber defines one room, even if the semantic
            # head predicts several labels or drops pixels near its boundaries.
            for chamber in enclosed_ids:
                region = partitions == chamber
                area = int(partition_stats[chamber, cv2.CC_STAT_AREA])
                votes = np.bincount(clean_room[region], minlength=len(ROOM_NAMES))
                class_id = int(np.argmax(votes[1:])) + 1
                geometry_only = votes[class_id] < .1 * area
                labeled_map[region] = current_label
                region_info[current_label] = {
                    "class_id": 0 if geometry_only else class_id,
                    "area_px": area,
                    "min_thresh": px_area_service,
                    "geometry_only": geometry_only,
                    "enclosed": True,
                }
                clean_room[region] = 0
                current_label += 1

            for c_id in range(1, len(ROOM_NAMES)):
                c_bin = (clean_room == c_id).astype(np.uint8)
                num_c, labels_c = cv2.connectedComponents(c_bin)
                for lbl in range(1, num_c):
                    r_mask = (labels_c == lbl)
                    area = int(r_mask.sum())
                    if area < 10:
                        continue
                    labeled_map[r_mask] = current_label
                    is_service = c_id in SERVICE_ROOM_IDS
                    min_thresh = px_area_service if is_service else px_area_habitable
                    region_info[current_label] = {
                        "class_id": c_id,
                        "area_px": area,
                        "min_thresh": min_thresh
                    }
                    current_label += 1

            # 3. Sub-threshold fragment absorption into longest-boundary neighbor
            kernel_absorb = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            small_labels = [lbl for lbl, info in region_info.items() if info["area_px"] < info["min_thresh"]]

            for s_lbl in small_labels:
                if s_lbl not in region_info:
                    continue
                s_mask = (labeled_map == s_lbl).astype(np.uint8)
                dilated = cv2.dilate(s_mask, kernel_absorb, iterations=1)
                border = dilated - s_mask

                neighbor_pixels = labeled_map[border > 0]
                neighbor_candidates = [n for n in neighbor_pixels if n > 0 and n != s_lbl]

                if neighbor_candidates:
                    counts = Counter(neighbor_candidates)
                    best_neighbor = counts.most_common(1)[0][0]
                    labeled_map[s_mask > 0] = best_neighbor
                    region_info[best_neighbor]["area_px"] += region_info[s_lbl]["area_px"]
                    del region_info[s_lbl]
                else:
                    # Isolated noise without neighboring room: clear it
                    labeled_map[s_mask > 0] = 0
                    del region_info[s_lbl]

            # 3b. Iteratively absorb unassigned interior space bounded by walls into adjacent rooms
            mask_interior = (filled_envelope > 0) & (wall_clean == 0)
            k_abs = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

            for _ in range(12):
                unassigned_interior = mask_interior & (labeled_map == 0)
                if not unassigned_interior.any():
                    break
                num_u, labels_u = cv2.connectedComponents(unassigned_interior.astype(np.uint8))
                if num_u <= 1:
                    break
                absorbed_any = False
                for u_lbl in range(1, num_u):
                    comp = (labels_u == u_lbl).astype(np.uint8)
                    comp_area = int(comp.sum())
                    if comp_area < 8:
                        continue
                    dilated_comp = cv2.dilate(comp, k_abs, iterations=2)
                    partition = partitions[comp > 0][0]
                    touching = labeled_map[(dilated_comp > 0) & (labeled_map > 0) & (partitions == partition)]
                    if len(touching) > 0:
                        best_lbl = Counter(touching).most_common(1)[0][0]
                        labeled_map[comp > 0] = best_lbl
                        if best_lbl in region_info:
                            region_info[best_lbl]["area_px"] += comp_area
                        absorbed_any = True
                if not absorbed_any:
                    # Isolated interior space without neighboring room: register as common space
                    for u_lbl in range(1, num_u):
                        comp = (labels_u == u_lbl).astype(np.uint8)
                        comp_area = int(comp.sum())
                        if comp_area >= px_area_service:
                            labeled_map[comp > 0] = current_label
                            region_info[current_label] = {
                                "class_id": 1,  # Living Room / Common space
                                "area_px": comp_area,
                                "min_thresh": px_area_habitable,
                                "geometry_only": True,
                            }
                            current_label += 1
                    break

            # 4. Construct vectorized room instances
            for lbl, info in region_info.items():
                r_mask = (labeled_map == lbl).astype(np.uint8)
                area_px = int(r_mask.sum())
                if area_px < 20:
                    continue

                # Fill any internal holes/notches inside room polygon (e.g. from text or furniture symbols)
                cnts_ext, _ = cv2.findContours(r_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not cnts_ext:
                    continue
                filled_rm = np.zeros_like(r_mask)
                cv2.drawContours(filled_rm, cnts_ext, -1, 1, -1)

                # Smooth out jagged text edges with morphological close
                k_smooth = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                smoothed_rm = cv2.morphologyEx(filled_rm, cv2.MORPH_CLOSE, k_smooth)

                r_cnts, _ = cv2.findContours(smoothed_rm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not r_cnts:
                    continue
                largest_c = max(r_cnts, key=cv2.contourArea)
                if cv2.contourArea(largest_c) < 15:
                    continue
                epsilon = 1.0 if info.get("enclosed") else min(2.5, max(.75, self.polygon_epsilon_ratio * cv2.arcLength(largest_c, True)))
                r_approx = cv2.approxPolyDP(largest_c, epsilon, True).reshape(-1, 2)
                if len(r_approx) < 3:
                    continue
                area_m2 = round(int(smoothed_rm.sum()) * (scale_m_per_px ** 2), 2)
                r_poly_m = [[round(pt[0] * scale_m_per_px, 3), round(pt[1] * scale_m_per_px, 3)] for pt in r_approx]

                # Validate and repair room polygon with Shapely make_valid
                r_repaired = False
                try:
                    r_raw_poly = Polygon(r_poly_m)
                    if not r_raw_poly.is_valid:
                        rep_r_poly, r_ok, _ = repair_polygon_to_single_polygon(r_raw_poly)
                        if r_ok and rep_r_poly.is_valid:
                            r_poly_m = [[round(pt[0], 3), round(pt[1], 3)] for pt in rep_r_poly.exterior.coords[:-1]]
                            r_repaired = True
                            num_rooms_repaired += 1
                except Exception:
                    pass

                M = cv2.moments(largest_c)
                cx = (M["m10"] / max(M["m00"], 1e-4)) * scale_m_per_px
                cy = (M["m01"] / max(M["m00"], 1e-4)) * scale_m_per_px
                c_id = info["class_id"]
                confidence = None
                if room_probabilities is not None and not info.get("geometry_only"):
                    confidence = score_room_polygon(room_probabilities, c_id, np.asarray(r_poly_m) / scale_m_per_px)
                total_interior_area_m2 += area_m2

                # Classify room edges against wall mask
                room_edges = []
                num_r_pts = len(r_poly_m)
                for ei in range(num_r_pts):
                    p1 = r_poly_m[ei]
                    p2 = r_poly_m[(ei + 1) % num_r_pts]
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    e_len = math.hypot(dx, dy)
                    e_dists = []
                    for ef in [0.2, 0.4, 0.5, 0.6, 0.8]:
                        sx = p1[0] + dx * ef
                        sy = p1[1] + dy * ef
                        ix = int(round(sx / scale_m_per_px))
                        iy = int(round(sy / scale_m_per_px))
                        if 0 <= ix < w and 0 <= iy < h:
                            e_dists.append(float(wall_dist_map_m[iy, ix]))
                    e_med_d = float(np.median(e_dists)) if e_dists else 0.05
                    room_edges.append({
                        "start": [round(p1[0], 3), round(p1[1], 3)],
                        "end": [round(p2[0], 3), round(p2[1], 3)],
                        "length_m": round(e_len, 3),
                        "is_wall_backed": bool(e_med_d <= 0.15),
                        "wall_distance_m": round(e_med_d, 3)
                    })

                wb_count = sum(1 for e in room_edges if e["is_wall_backed"])

                rooms.append({
                    "id": f"room_{room_idx}",
                    "room_type": ROOM_NAMES.get(c_id, "Other Room") if c_id else "Unclassified Room",
                    "surface_m2": area_m2,
                    "confidence": confidence,
                    "centroid_meters": [round(cx, 3), round(cy, 3)],
                    "polygon_meters": r_poly_m,
                    "was_repaired": r_repaired,
                    "edges": room_edges,
                    "wall_backed_edge_count": wb_count
                })
                room_idx += 1
        elif room_mask is not None:
            # Explicit room_mask provided with zero detected rooms
            rooms = []
        else:
            # Geometric Connected Components fallback (only when room_mask is None)
            interior_mask = (wall_clean == 0).astype(np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(interior_mask)
            for lbl in range(1, num_labels):
                area_px = stats[lbl, cv2.CC_STAT_AREA]
                if area_px < 200 or area_px > (h * w * 0.7):
                    continue

                area_m2 = round(area_px * (scale_m_per_px ** 2), 2)
                total_interior_area_m2 += area_m2
                room_bin = (labels == lbl).astype(np.uint8)
                r_cnts, _ = cv2.findContours(room_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(r_cnts) > 0:
                    largest_c = max(r_cnts, key=cv2.contourArea)
                    r_approx = cv2.approxPolyDP(largest_c, 3.0, True).reshape(-1, 2)
                    r_poly_m = [[round(pt[0] * scale_m_per_px, 3), round(pt[1] * scale_m_per_px, 3)] for pt in r_approx]
                    r_type = "Living Room / Common" if area_m2 > 25 else "Bedroom" if area_m2 > 12 else "Bathroom / Storage"

                    # Validate and repair room polygon with Shapely make_valid
                    r_repaired = False
                    try:
                        r_raw_poly = Polygon(r_poly_m)
                        if not r_raw_poly.is_valid:
                            rep_r_poly, r_ok, _ = repair_polygon_to_single_polygon(r_raw_poly)
                            if r_ok and rep_r_poly.is_valid:
                                r_poly_m = [[round(pt[0], 3), round(pt[1], 3)] for pt in rep_r_poly.exterior.coords[:-1]]
                                r_repaired = True
                                num_rooms_repaired += 1
                    except Exception:
                        pass

                    rooms.append({
                        "id": f"room_{room_idx}",
                        "room_type": r_type,
                        "surface_m2": area_m2,
                        "centroid_meters": [round(centroids[lbl][0] * scale_m_per_px, 3), round(centroids[lbl][1] * scale_m_per_px, 3)],
                        "polygon_meters": r_poly_m,
                        "was_repaired": r_repaired
                    })
                    room_idx += 1

        # 7. Refine Exterior Walls using Room Union Outer Boundary (fallback only when struct_mask is None)
        if struct_mask is None:
            try:
                from shapely.ops import unary_union
                valid_room_polys = [
                    Polygon(r["polygon_meters"])
                    for r in rooms
                    if len(r.get("polygon_meters", [])) >= 3 and Polygon(r["polygon_meters"]).is_valid
                ]
                if len(valid_room_polys) > 0:
                    room_envelope = unary_union(valid_room_polys)
                    envelope_boundary = room_envelope.boundary
                    for w_item in walls:
                        poly_pts = w_item.get("polygon_meters", [])
                        if len(poly_pts) >= 3:
                            w_poly = Polygon(poly_pts)
                            if w_poly.is_valid:
                                if envelope_boundary.distance(w_poly) < 0.45:
                                    w_item["is_exterior"] = True
            except Exception:
                pass

        # 8. Keep floors on the wall-supported building outline. Buffering
        # disconnected room labels alone can create extra slabs inside one house.
        envelope_polygons = []
        room_centers = [
            Polygon(room["polygon_meters"]).representative_point()
            for room in rooms
            if len(room["polygon_meters"]) >= 3
        ]
        for contour in footprint_contours:
            if cv2.contourArea(contour) * scale_m_per_px ** 2 < 2.0:
                continue
            outline = cv2.approxPolyDP(contour, 1.5, True).reshape(-1, 2) * scale_m_per_px
            if len(outline) < 3:
                continue
            polygon = Polygon(outline)
            if not polygon.is_valid:
                polygon, repaired, _ = repair_polygon_to_single_polygon(polygon)
                if not repaired:
                    continue
            if any(polygon.covers(center) for center in room_centers):
                envelope_polygons.append(np.round(np.asarray(polygon.exterior.coords[:-1]), 3).tolist())

        try:
            from shapely.ops import unary_union
            valid_r_polys = [
                Polygon(r["polygon_meters"])
                for r in rooms
                if len(r.get("polygon_meters", [])) >= 3 and Polygon(r["polygon_meters"]).is_valid
            ]
            if not envelope_polygons and valid_r_polys:
                # Buffer rooms by 0.22m to bridge walls and corridors
                buffered = [p.buffer(0.22, join_style=2) for p in valid_r_polys]
                merged_env = unary_union(buffered)
                if merged_env.geom_type == "Polygon" and merged_env.area > 5.0:
                    outer = Polygon(merged_env.exterior)
                    s = outer.simplify(0.08, preserve_topology=True)
                    envelope_polygons.append([[round(pt[0], 3), round(pt[1], 3)] for pt in s.exterior.coords[:-1]])
                elif merged_env.geom_type == "MultiPolygon":
                    for g in merged_env.geoms:
                        if g.area > 5.0:
                            outer = Polygon(g.exterior)
                            s = outer.simplify(0.08, preserve_topology=True)
                            envelope_polygons.append([[round(pt[0], 3), round(pt[1], 3)] for pt in s.exterior.coords[:-1]])
        except Exception:
            pass

        # Fallback to footprint_contours if room envelope was empty
        if not envelope_polygons and len(footprint_contours) > 0:
            for cnt in footprint_contours:
                if cv2.contourArea(cnt) > 80:
                    approx = cv2.approxPolyDP(cnt, 2.5, True).reshape(-1, 2)
                    if len(approx) >= 3:
                        envelope_polygons.append([[round(pt[0] * scale_m_per_px, 3), round(pt[1] * scale_m_per_px, 3)] for pt in approx])

        # Strict Building Shell Validation:
        # A valid residential shell must enclose at least 1 room and have at least 4 walls or segments
        envelope_segments = sum(len(w.get("polygon_meters", [])) for w in walls)
        has_enough_walls = (len(walls) >= 4) or (envelope_segments >= 4 and len(walls) >= 1)

        warnings = []
        if len(rooms) == 0:
            shell_status = "NO_ROOMS_DETECTED"
            strict_shell_valid = False
            warnings.append("NO_ROOMS_DETECTED")
        elif not has_enough_walls:
            shell_status = "INCOMPLETE_ENVELOPE"
            strict_shell_valid = False
            warnings.append("INCOMPLETE_ENVELOPE")
        elif not shell_valid:
            shell_status = shell_failure_code if shell_failure_code else "INVALID_GEOMETRY"
            strict_shell_valid = False
            warnings.append(shell_status)
        else:
            shell_status = "VALID"
            strict_shell_valid = True

        if was_shell_repaired:
            warnings.append("SHELL_REPAIRED_TOPOLOGY")
        if len(doors) > max(4, int(2.5 * len(rooms))):
            warnings.append("DOOR_COUNT_IMPLAUSIBLE")

        return {
            "metadata": {
                "canvas_size": [w, h],
                "scale_meters_per_pixel": round(scale_m_per_px, 5),
                "scale_source": scale_source,
                "building_perimeter_m": building_perimeter_m,
                "is_building_shell_valid": strict_shell_valid,
                "shell_status": shell_status,
                "was_shell_repaired": was_shell_repaired,
                "warnings": warnings,
                "total_surface_m2": round(total_interior_area_m2, 2),
                "num_walls": len(walls),
                "num_exterior_walls": sum(1 for w in walls if w.get("is_exterior", False)),
                "num_envelope_segments": envelope_segments,
                "num_rooms": len(rooms),
                "num_rooms_repaired": num_rooms_repaired,
                "num_doors": len(doors),
                "num_windows": len(windows),
                "envelope": envelope_polygons
            },
            "envelope": envelope_polygons,
            "walls": walls,
            "rooms": rooms,
            "doors": doors,
            "windows": windows
        }

