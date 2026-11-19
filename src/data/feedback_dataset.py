"""
Turn studio corrections (data/feedback/<id>/) into training examples in the CubiCasa cache format.

For each correction: the plan image is letterboxed to 512 px exactly as the API does, and the
corrected layout (meters, px = m / scale) is drawn back into the two label masks:
  structure mask: 1 exterior wall, 2 interior wall, 3 door, 4 window (openings drawn as rectangles
                  across their wall), 0 background
  room mask:      room polygons with the ArchiGen room class ids; unknown room types -> 255 (ignored)
Output: data/processed/feedback_cache/train/feedback_<id>.npz with keys image, structure_mask, room_mask
(same keys as data/processed/cubicasa_cache), used by `train_floorplan --include-feedback`.

    python -m src.data.feedback_dataset
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
from PIL import Image

from src.config import DATA_DIR
from src.data.svg_preprocessor import isotropic_letterbox
from src.geometry.layout_vectorizer import ROOM_NAMES
from src.mlops.feedback import FEEDBACK_DIR

OUT_DIR = DATA_DIR / "processed" / "feedback_cache" / "train"
ROOM_IDS = {name.lower(): idx for idx, name in ROOM_NAMES.items()}
IGNORE = 255


def _px(points, scale: float) -> np.ndarray:
    return np.rint(np.asarray(points, dtype=np.float64) / scale).astype(np.int32)


def rasterize_layout(layout: Dict[str, Any], scale: float, size: int = 512):
    struct = np.zeros((size, size), np.uint8)
    rooms = np.zeros((size, size), np.uint8)
    for room in layout.get("rooms", []):
        poly = room.get("polygon_meters") or room.get("polygon_points") or []
        if len(poly) >= 3:
            cls = ROOM_IDS.get(str(room.get("room_type", "")).lower(), IGNORE)
            cv2.fillPoly(rooms, [_px(poly, scale)], int(cls))
    walls = {w.get("id"): w for w in layout.get("walls", [])}
    for wall in walls.values():
        poly = wall.get("polygon_meters") or []
        if len(poly) >= 3:
            cv2.fillPoly(struct, [_px(poly, scale)], 1 if wall.get("is_exterior") else 2)
    for kind, cls in (("doors", 3), ("windows", 4)):
        for op in layout.get(kind, []):
            centre, width = op.get("position_meters"), float(op.get("width_m") or 0)
            wall = walls.get(op.get("associated_wall"))
            if not centre or width <= 0 or not wall or not wall.get("axis_meters"):
                continue
            (x0, y0), (x1, y1) = wall["axis_meters"]
            d = np.array([x1 - x0, y1 - y0], np.float64)
            d /= max(np.linalg.norm(d), 1e-9)
            n = np.array([-d[1], d[0]])
            half_t = max(float(wall.get("thickness_m") or 0.2), 0.1) / 2 + 0.02
            c = np.asarray(centre, np.float64)
            corners = [c - d * width / 2 - n * half_t, c + d * width / 2 - n * half_t,
                       c + d * width / 2 + n * half_t, c - d * width / 2 + n * half_t]
            cv2.fillPoly(struct, [_px(corners, scale)], cls)
    rooms[struct > 0] = 0          # walls and openings are not room pixels (as in CubiCasa)
    return struct, rooms


def build(out_dir: Path = OUT_DIR) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    done, skipped = [], []
    for meta_path in sorted(FEEDBACK_DIR.glob("*/meta.json")) if FEEDBACK_DIR.exists() else []:
        folder = meta_path.parent
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        layout = json.loads((folder / "layout.json").read_text(encoding="utf-8"))
        scale = float(meta.get("scale_meters_per_pixel") or layout.get("metadata", {}).get("scale_meters_per_pixel") or 0)
        if scale <= 0:
            skipped.append((folder.name, "no scale"))
            continue
        rgb = np.array(Image.open(folder / meta["image_file"]).convert("RGB"))
        image, _ = isotropic_letterbox(rgb, target_size=(512, 512), is_mask=False, fill_value=255)
        struct, rooms = rasterize_layout(layout, scale)
        if (struct > 0).sum() < 200:
            skipped.append((folder.name, "almost no walls"))
            continue
        np.savez_compressed(out_dir / f"feedback_{folder.name}.npz", image=image.astype(np.uint8),
                            structure_mask=struct, room_mask=rooms)
        done.append(folder.name)
    summary = {"converted": len(done), "skipped": skipped, "output": str(out_dir)}
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    build()
