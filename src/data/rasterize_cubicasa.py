"""
ArchiGen 3D - CubiCasa5K SVG -> compressed .NPZ rasterizer
TEK-UP UNIVERSITÉ | Deep Learning Avancé & MLOps 2026/2027

Converts raw CubiCasa5K houses (F1_scaled.png + model.svg) into .npz files
using the OFFICIAL train/val/test split files shipped with the dataset:

    data/processed/cubicasa_cache/{train,val,test}/<category>_<house_id>.npz

Each .npz holds:
  - 'image':          (S, S, 3) uint8, letterboxed on a white canvas
  - 'structure_mask': (S, S) uint8  0=background 1=ext wall 2=int wall 3=door 4=window
  - 'room_mask':      (S, S) uint8  0..11 (see src.config.ROOM_CLASSES),
                                    255 = IGNORE (unknown / unmapped room type)

Verified facts about the raw data (checked on the downloaded archive):
  * SVG polygon coordinates are pixel coordinates of F1_scaled.png.
  * The SVG width/height cover only the drawn plan; the rest of F1_scaled.png is
    blank paper, so the image is cropped to the SVG extent before resizing.
  * 'Space Undefined' is the most frequent room label (~11k polygons). It is mapped
    to IGNORE rather than to a real class, so it never pollutes the room labels.

Requires only Pillow + numpy (no OpenCV).
"""

from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

SVG_NS = "{http://www.w3.org/2000/svg}"
IGNORE_INDEX = 255
DEFAULT_SOURCE = RAW_DATA_DIR / "cubicasa5k"
DEFAULT_OUT = PROCESSED_DATA_DIR / "cubicasa_cache"
SPLITS = ("train", "val", "test")

# CubiCasa5K "Space <Type>" -> ArchiGen 12-class room taxonomy (src.config.ROOM_CLASSES).
# Every type observed in the dataset is listed explicitly; anything else -> IGNORE.
ROOM_TYPE_TO_ID: Dict[str, int] = {
    # 1 living_room
    "LivingRoom": 1, "Den": 1, "RecreationRoom": 1, "Alcove": 1, "Bar": 1,
    # 2 bedroom
    "Bedroom": 2,
    # 3 kitchen
    "Kitchen": 3,
    # 4 bathroom
    "Bath": 4, "Sauna": 4,
    # 5 hallway
    "Entry": 5, "Hall": 5, "DraughtLobby": 5, "Elevator": 5,
    # 6 closet
    "Closet": 6, "DressingRoom": 6,
    # 7 balcony / outdoor
    "Outdoor": 7,
    # 8 dining
    "Dining": 8,
    # 9 office
    "Office": 9, "Library": 9,
    # 10 storage
    "Storage": 10, "Garage": 10, "Attic": 10, "Basement": 10, "Garbage": 10,
    # 11 technical
    "Utility": 11, "TechnicalRoom": 11,
    # Explicitly ignored (ambiguous / not in taxonomy)
    "Undefined": IGNORE_INDEX, "UserDefined": IGNORE_INDEX, "Room": IGNORE_INDEX,
    "Elevated": IGNORE_INDEX, "CarPort": IGNORE_INDEX, "SwimmingPool": IGNORE_INDEX,
    "RetailSpace": IGNORE_INDEX, "OpenToBelow": IGNORE_INDEX, "ExerciseRoom": IGNORE_INDEX,
}


def _parse_points(points: str) -> List[Tuple[float, float]]:
    vals = points.replace(",", " ").split()
    return [(float(vals[i]), float(vals[i + 1])) for i in range(0, len(vals) - 1, 2)]


def _own_polygons(g: ET.Element) -> List[List[Tuple[float, float]]]:
    """Polygons that are DIRECT children of a <g> (not those of nested doors/windows)."""
    polys = []
    for p in g.findall(f"{SVG_NS}polygon"):
        pts = _parse_points(p.get("points", ""))
        if len(pts) >= 3:
            polys.append(pts)
    return polys


def read_split_file(path: Path) -> List[str]:
    """Split files contain lines like '/high_quality_architectural/6044/'."""
    return [ln.strip().strip("/") for ln in path.read_text().splitlines() if ln.strip()]


def rasterize_house(house_dir: Path, target_size: int = 512) -> Optional[Dict[str, np.ndarray]]:
    """Rasterize one house. Returns arrays, or None if files are missing."""
    img_path = house_dir / "F1_scaled.png"
    svg_path = house_dir / "model.svg"
    if not img_path.exists() or not svg_path.exists():
        return None

    root = ET.parse(svg_path).getroot()
    img = Image.open(img_path).convert("RGB")
    iw, ih = img.size

    # Crop to the plan extent declared by the SVG (coordinates == image pixels).
    cw = min(iw, int(np.ceil(float(root.get("width", iw)))))
    ch = min(ih, int(np.ceil(float(root.get("height", ih)))))
    img = img.crop((0, 0, cw, ch))

    scale = min(target_size / cw, target_size / ch)
    nw, nh = max(1, round(cw * scale)), max(1, round(ch * scale))
    pad_x, pad_y = (target_size - nw) // 2, (target_size - nh) // 2

    canvas = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    canvas.paste(img.resize((nw, nh), Image.Resampling.BILINEAR), (pad_x, pad_y))

    # Masks are drawn with nearest semantics (PIL polygon fill on a 'L' image).
    room_img = Image.new("L", (target_size, target_size), 0)
    struct_img = Image.new("L", (target_size, target_size), 0)
    room_draw, struct_draw = ImageDraw.Draw(room_img), ImageDraw.Draw(struct_img)

    def tf(pts):
        return [(x * scale + pad_x, y * scale + pad_y) for x, y in pts]

    unknown_types: Dict[str, int] = {}
    # Document order: rooms first, then walls, then the doors/windows nested in walls,
    # so openings overwrite the wall pixels they sit in.
    for g in root.iter(f"{SVG_NS}g"):
        tokens = g.get("class", "").split()
        if not tokens:
            continue
        kind = tokens[0]
        if kind == "Space":
            rtype = tokens[1] if len(tokens) > 1 else "Undefined"
            rid = ROOM_TYPE_TO_ID.get(rtype)
            if rid is None:
                unknown_types[rtype] = unknown_types.get(rtype, 0) + 1
                rid = IGNORE_INDEX
            for pts in _own_polygons(g):
                room_draw.polygon(tf(pts), fill=rid)
        elif kind in ("Wall", "Door", "Window"):
            sid = {"Door": 3, "Window": 4}.get(kind, 1 if "External" in tokens else 2)
            for pts in _own_polygons(g):
                struct_draw.polygon(tf(pts), fill=sid)

    return {
        "image": np.asarray(canvas, dtype=np.uint8),
        "structure_mask": np.asarray(struct_img, dtype=np.uint8),
        "room_mask": np.asarray(room_img, dtype=np.uint8),
        "_unknown_types": unknown_types,
    }


def _worker(args) -> Tuple[str, str, Optional[str], Dict[str, int]]:
    rel, split, src_root, out_root, size, overwrite = args
    house_dir = Path(src_root) / rel
    out_file = Path(out_root) / split / (rel.replace("/", "_") + ".npz")
    if out_file.exists() and not overwrite:
        return rel, split, None, {}
    try:
        res = rasterize_house(house_dir, size)
        if res is None:
            return rel, split, "missing F1_scaled.png or model.svg", {}
        unknown = res.pop("_unknown_types")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_file, **res)
        return rel, split, None, unknown
    except Exception as exc:  # corrupted SVG / image
        return rel, split, f"{type(exc).__name__}: {exc}", {}


def find_dataset_root(source: Path) -> Path:
    """Locate the folder that holds train.txt (handles cubicasa5k/cubicasa5k nesting)."""
    if (source / "train.txt").exists():
        return source
    for p in sorted(source.rglob("train.txt")):
        return p.parent
    raise FileNotFoundError(f"No train.txt found under {source}. Is CubiCasa5K extracted there?")


def rasterize_cubicasa_dataset(
    source: Path = DEFAULT_SOURCE,
    out_dir: Path = DEFAULT_OUT,
    target_size: int = 512,
    max_per_split: Optional[int] = None,
    workers: int = 8,
    overwrite: bool = False,
) -> Dict[str, object]:
    root = find_dataset_root(Path(source))
    out_dir = Path(out_dir)
    jobs = []
    for split in SPLITS:
        rels = read_split_file(root / f"{split}.txt")
        if max_per_split:
            rels = rels[:max_per_split]
        jobs += [(r, split, str(root), str(out_dir), target_size, overwrite) for r in rels]

    print(f"==> CubiCasa5K root: {root}")
    print(f"==> Rasterizing {len(jobs)} houses at {target_size}x{target_size} with {workers} workers")
    t0 = time.time()
    failures: List[Dict[str, str]] = []
    unknown_total: Dict[str, int] = {}
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for rel, split, err, unknown in ex.map(_worker, jobs, chunksize=8):
            done += 1
            if err:
                failures.append({"house": rel, "split": split, "error": err})
            for k, v in unknown.items():
                unknown_total[k] = unknown_total.get(k, 0) + v
            if done % 250 == 0 or done == len(jobs):
                print(f"  [{done}/{len(jobs)}] {done / (time.time() - t0):.1f} houses/s, failures={len(failures)}")

    counts = {s: len(list((out_dir / s).glob("*.npz"))) for s in SPLITS}
    manifest = {
        "source": str(root),
        "target_size": target_size,
        "counts": counts,
        "failures": failures,
        "unmapped_room_types": unknown_total,
        "room_type_mapping": ROOM_TYPE_TO_ID,
        "ignore_index": IGNORE_INDEX,
        "seconds": round(time.time() - t0, 1),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"==> Done: {counts} | failures={len(failures)} | manifest: {out_dir / 'manifest.json'}")
    return manifest


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Rasterize CubiCasa5K (official splits) to NPZ")
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--max-per-split", type=int, default=None, help="Limit houses per split (smoke tests)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    rasterize_cubicasa_dataset(a.source, a.out, a.size, a.max_per_split, a.workers, a.overwrite)
