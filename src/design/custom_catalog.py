"""Adapt saved image-to-3D assets to the room catalog's furniture types."""

import json
import math
import re
import struct


FURNITURE_TYPES = {
    "chair": ("dining_chair", "armchair", "office_chair"),
    "table": ("coffee_table", "dining_table", "desk"),
    "storage": ("wardrobe", "bookcase", "sideboard", "nightstand"),
    "sofa": ("sofa",),
    "bed": ("bed",),
    "lighting": ("lamp",),
    "decor": ("plant",),
}


def compatible_types(category, furniture_type, name):
    types = FURNITURE_TYPES.get(category)
    if types is None:
        return (category,) if any(category in group for group in FURNITURE_TYPES.values()) else ()
    if furniture_type in types:
        return (furniture_type,)
    words = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    for candidate in types:
        labels = ("armchair", "arm chair") if candidate == "armchair" else (candidate.replace("_", " "),)
        if any(f" {label} " in f" {words} " for label in labels):
            return (candidate,)
    return types


def valid_glb_header(header, size):
    if len(header) < 20 or header[:4] != b"glTF":
        return False
    version, declared_size, json_size, chunk_type = struct.unpack("<III4s", header[4:20])
    return version == 2 and declared_size == size and chunk_type == b"JSON" and 20 + json_size <= size


def load_custom_catalog(catalog_path, assets_dir):
    if not catalog_path.is_file():
        return {}
    try:
        saved = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    if not isinstance(saved, dict):
        return {}

    catalog = {}
    for key, spec in saved.items():
        if not isinstance(spec, dict) or not re.fullmatch(r"[A-Za-z0-9_-]+", key):
            continue
        asset = assets_dir / f"{key}.glb"
        try:
            with asset.open("rb") as stream:
                if not valid_glb_header(stream.read(20), asset.stat().st_size):
                    continue
        except OSError:
            continue
        size = spec.get("dimensions_m")
        if not isinstance(size, list) or len(size) != 3 or not all(
            isinstance(dim, (int, float)) and math.isfinite(dim) and dim > 0 for dim in size
        ):
            continue
        name = str(spec.get("name") or "Custom furniture")
        types = compatible_types(str(spec.get("category", "")).lower(), spec.get("furniture_type"), name)
        if types:
            catalog[key] = {
                "name": name, "types": types, "dimensions_m": size,
                "glb": f"/models/custom/{key}.glb", "source": "custom",
            }
    return catalog
