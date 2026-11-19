"""
Facade composer: makes a studio wall look like a CMP training facade before Pix2Pix sees it.

Problem: a plan wall is often ONE storey and 8-20 m long. Drawn as is, it becomes a thin strip in a
256x256 canvas surrounded by "unknown" colour -- nothing like the CMP training images (square,
4-6 storey apartment fronts) -- and Pix2Pix produces smeared output.

Method (this repository):
  1. Scale: one storey = ~44-65 px, like CMP buildings.
  2. Tile: a long wall is covered by overlapping 256 px tiles, blended with feathered weights.
  3. Stack: above the real storeys we add "context" storeys (same window rhythm, a moulding band
     between storeys, a cornice on top), so every tile looks like a normal multi-storey facade.
  4. Generate all tiles in one batch, then CUT OUT only the real storeys of each segment and stitch
     them. The texture covers exactly the wall (UV 0..1); openings keep their exact positions.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List

import numpy as np
from PIL import Image, ImageDraw

from src.data.facade_palette import CMP_TRAINING_LUT

# Colours exactly as in the CMP training label maps (src/data/facade_palette.py).
C = {name: tuple(int(v) for v in CMP_TRAINING_LUT[i]) for name, i in {
    "background": 1, "facade": 2, "window": 3, "door": 4, "cornice": 5, "sill": 6,
    "balcony": 7, "deco": 9, "molding": 10, "pillar": 11, "shop": 12}.items()}
TILE = 256
TOP_SKY_PX = 14          # background strip above the cornice (sky / neighbours in CMP photos)
OVERLAP = 96             # px shared by neighbouring tiles on long walls (blended, no visible seam)


def _px_per_m(wall_len_m: float, floor_h_m: float) -> float:
    """Short walls fill the tile width; long walls are tiled. A storey stays ~44-65 px high."""
    lo, hi = 44 / floor_h_m, 65 / floor_h_m
    return float(np.clip(TILE / max(wall_len_m, 0.5), lo, hi))


def _context_windows(wall_len_m: float, real: List[Dict[str, Any]]) -> List[tuple]:
    """Window columns for the context storeys: the real windows/doors' columns, or a regular rhythm."""
    cols = sorted({(round(o["position_along_wall_m"], 2), round(min(o["width_m"], 1.6), 2)) for o in real})
    if cols:
        return [(x, max(0.8, w)) for x, w in cols]
    n = max(1, int(wall_len_m // 2.8))
    return [((i + 0.5) * wall_len_m / n, 1.2) for i in range(n)]


def compose_sketch(wall_len_m: float, floor_h_m: float, n_floors: int,
                   openings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Draws the full multi-storey label map (all tiles side by side) and records the crop of the real wall."""
    s = _px_per_m(wall_len_m, floor_h_m)
    floor_px = floor_h_m * s
    wall_w = max(1, int(round(wall_len_m * s)))
    if wall_w <= TILE:                               # short wall: one tile, wall centred
        width, x0, tile_x = TILE, (TILE - wall_w) // 2, [0]
    else:                                            # long wall: overlapping tiles across the wall
        width, x0 = wall_w, 0
        n = math.ceil((wall_w - OVERLAP) / (TILE - OVERLAP))
        tile_x = [round(i * (wall_w - TILE) / (n - 1)) for i in range(n)]
    n_tiles = len(tile_x)
    total_floors = max(n_floors, int((TILE - TOP_SKY_PX) // floor_px))
    canvas = Image.new("RGB", (width, TILE), C["background"])
    d = ImageDraw.Draw(canvas)
    ground_y = TILE                                  # building stands on the bottom edge
    top_y = ground_y - total_floors * floor_px
    d.rectangle([x0, top_y, x0 + wall_w - 1, ground_y - 1], fill=C["facade"])
    d.rectangle([x0, top_y - 6, x0 + wall_w - 1, top_y + 3], fill=C["cornice"])
    X = lambda m: x0 + m * s
    Y = lambda m: ground_y - m * s                   # meters above ground -> pixel row

    def rect(x_c, w, bottom, h, colour):
        d.rectangle([X(x_c - w / 2), Y(bottom + h), X(x_c + w / 2) - 1, Y(bottom) - 1], fill=colour)

    # Real storeys: the wall's actual openings (exact positions, they are cut in 3D).
    for op in openings:
        x, w, sill, h = op["position_along_wall_m"], op["width_m"], op["sill_m"], op["height_m"]
        if op.get("type") == "door":
            rect(x, w, sill, h, C["door"])
        else:
            rect(x, w, sill, h, C["window"])
            rect(x, w + 0.15, sill - 0.12, 0.12, C["sill"])
    # Context storeys above the real ones: same window columns, standard sill / height.
    cols = _context_windows(wall_len_m, openings)    # every opening column, incl. the one above the door
    for f in range(n_floors, total_floors):
        base = f * floor_h_m
        for x, w in cols:
            rect(x, w, base + 0.8, 1.25, C["window"])
            rect(x, w + 0.15, base + 0.68, 0.12, C["sill"])
    # Moulding bands between storeys (CMP facades almost always have them).
    for f in range(1, total_floors):
        y = Y(f * floor_h_m)
        d.rectangle([x0, y - 2, x0 + wall_w - 1, y + 1], fill=C["molding"])
    crop = (x0, int(round(Y(n_floors * floor_h_m))), x0 + wall_w, ground_y)
    return {"sketch": canvas, "n_tiles": n_tiles, "tile_x": tile_x, "crop": crop, "px_per_m": s,
            "total_floors": total_floors}


def _blend_weight(x: int, width: int) -> np.ndarray:
    """Per-column weight of a tile: linear feather on edges that overlap a neighbour tile."""
    w = np.ones(TILE, np.float32)
    ramp = np.linspace(0.05, 1.0, OVERLAP // 2, dtype=np.float32)
    if x > 0:
        w[: len(ramp)] = ramp
    if x + TILE < width:
        w[-len(ramp):] = ramp[::-1]
    return w


def generate_wall_texture(wall_len_m: float, floor_h_m: float, n_floors: int, openings: List[Dict[str, Any]],
                          generate_batch: Callable[[List[Image.Image]], List[Image.Image]],
                          upscaler: Callable[[Image.Image], Image.Image] | None = None) -> Dict[str, Any]:
    """Compose, generate every tile in one batch, stitch, and crop to the real wall.

    `generate_batch` may return tiles larger than 256 px (ControlNet: 512); the stitching then runs
    at that resolution, so the high-quality texture keeps its extra detail.
    """
    comp = compose_sketch(wall_len_m, floor_h_m, n_floors, openings)
    sketch = comp["sketch"]
    tiles = [sketch.crop((x, 0, x + TILE, TILE)) for x in comp["tile_x"]]
    photos = generate_batch(tiles)
    k = max(1, round(photos[0].width / TILE))       # 1 for Pix2Pix, 2 for ControlNet
    T = TILE * k
    acc = np.zeros((T, sketch.width * k, 3), np.float32)
    wsum = np.zeros((1, sketch.width * k, 1), np.float32)
    for x, ph in zip(comp["tile_x"], photos):    # feathered blend of overlapping tiles (no seams)
        w = np.repeat(_blend_weight(x, sketch.width), k)[None, :, None]
        acc[:, x * k:x * k + T] += np.asarray(ph.convert("RGB").resize((T, T)), np.float32) * w
        wsum[:, x * k:x * k + T] += w
    full = Image.fromarray(np.clip(acc / np.maximum(wsum, 1e-6), 0, 255).astype(np.uint8))
    texture = full.crop(tuple(v * k for v in comp["crop"]))
    sketch_crop = sketch.crop(comp["crop"]).resize(texture.size, Image.Resampling.NEAREST)
    if upscaler is not None:            # optional AI super-resolution of the native-resolution crop
        texture = upscaler(texture)
        sketch_crop = sketch_crop.resize(texture.size, Image.Resampling.NEAREST)
    # Upscale the wall strip so the 3D texture is not tiny (nearest keeps the label map crisp).
    scale = max(1, int(math.ceil(512 / max(texture.width, 1))))
    if scale > 1:
        texture = texture.resize((texture.width * scale, texture.height * scale), Image.Resampling.BICUBIC)
        sketch_crop = sketch_crop.resize((sketch_crop.width * scale, sketch_crop.height * scale), Image.Resampling.NEAREST)
    return {"texture": texture, "sketch": sketch_crop, "context_sketch": sketch, "context_photo": full,
            "n_tiles": comp["n_tiles"], "context_floors": comp["total_floors"], "px_per_m": comp["px_per_m"]}
