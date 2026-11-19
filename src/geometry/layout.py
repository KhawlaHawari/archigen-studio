"""
Floor-plan layout builder shared by the API and the layout evaluation.

Turns the UNet++ class probabilities into the vector layout the web studio extrudes in 3D:
walls (axis, thickness, openings), rooms (type, polygon, area, confidence), doors, windows
and the building envelope, all in meters.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from src.geometry.layout_vectorizer import FloorplanVectorizer

_VECTORIZER: Optional[FloorplanVectorizer] = None


# Vectorizer hyperparameters tuned by Optuna (src/mlops/hpo_vectorizer.py) on the validation split;
# the defaults of FloorplanVectorizer are used when the file does not exist.
TUNED_PARAMS = Path(__file__).resolve().parents[2] / "models" / "vectorizer_params.json"


def tuned_vectorizer_params() -> Dict[str, Any]:
    try:
        return json.loads(TUNED_PARAMS.read_text(encoding="utf-8"))["params"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return {}


def _vectorizer() -> FloorplanVectorizer:
    global _VECTORIZER
    if _VECTORIZER is None:
        _VECTORIZER = FloorplanVectorizer(**tuned_vectorizer_params())
    return _VECTORIZER


def layout_from_probabilities(
    prob_s: np.ndarray,
    prob_r: np.ndarray,
    image_rgb: np.ndarray,
    scale_override: Optional[float] = None,
    vectorizer: Optional[FloorplanVectorizer] = None,
) -> Dict[str, Any]:
    """
    prob_s: (5, H, W) structure probabilities (0 bg, 1 exterior wall, 2 interior wall, 3 door, 4 window).
    prob_r: (12, H, W) room probabilities (src.config.ROOM_CLASSES order).
    image_rgb: (H, W, 3) letterboxed input, used as ink evidence on sparse line drawings.
    """
    struct_mask = np.argmax(prob_s, axis=0).astype(np.uint8)
    room_mask = np.argmax(prob_r, axis=0).astype(np.uint8)

    # The vectorizer expects 1 = any wall, 2 = door, 3 = window ...
    opening_mask = np.zeros_like(struct_mask)
    opening_mask[(struct_mask == 1) | (struct_mask == 2)] = 1
    opening_mask[struct_mask == 3] = 2
    opening_mask[struct_mask == 4] = 3
    # ... and a structure mask with 1 = interior wall, 2 = exterior wall (ArchiGen uses the reverse).
    legacy_struct = struct_mask.copy()
    legacy_struct[struct_mask == 1] = 2
    legacy_struct[struct_mask == 2] = 1

    layout = (vectorizer or _vectorizer()).vectorize(
        mask=opening_mask,
        struct_mask=legacy_struct,
        room_mask=room_mask,
        scale_override_m_per_px=scale_override,
        room_probabilities=prob_r,
        source_gray=np.asarray(Image.fromarray(image_rgb.astype(np.uint8)).convert("L")),
    )

    # Default front facade (the user can pick another one in the studio): the longest exterior wall
    # that has an entrance door among walls at least 65 % as long as the longest; otherwise the one
    # with the most openings, so the generated facade is not a blank wall.
    def wall_length(wall: Dict[str, Any]) -> float:
        axis = wall.get("axis_meters")
        return float(np.hypot(axis[1][0] - axis[0][0], axis[1][1] - axis[0][1])) if axis else 0.0

    exterior = [w for w in layout.get("walls", []) if w.get("is_exterior")]
    front = None
    if exterior:
        longest = max(wall_length(w) for w in exterior)
        long_walls = [w for w in exterior if wall_length(w) >= max(1.8, 0.65 * longest)] or exterior
        with_door = [w for w in long_walls if any(o.get("type") == "door" for o in w.get("openings", []))]
        front = (max(with_door, key=wall_length) if with_door
                 else max(long_walls, key=lambda w: (len(w.get("openings", [])), wall_length(w))))
    for wall in layout.get("walls", []):
        wall["is_front"] = wall is front
    return layout
