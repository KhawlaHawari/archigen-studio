"""
Drawing-style augmentation for floor-plan training (recipe v3).

96 % of the CubiCasa5K training plans draw walls in green on a light background, so a model can
learn "green = wall" and fail on black-ink CAD plans, scans, sketches or dark-mode drawings.
This module re-draws a training image in another style. Only the IMAGE changes; the structure
and room masks stay the same, so labels remain exact (no annotation leakage).

Styles (one is drawn at random):
  * ink       - grayscale, room fills removed, walls/furniture/text kept as dark ink
  * render    - clean CAD-like drawing rendered from the structure mask (solid or hatched walls,
                windows as double lines, doors as gaps with a swing arc) with the original
                furniture/text ink overlaid, so room-type cues survive
  * recolor   - original image with the wall pixels recoloured (black, grey, brown, blue ...)
followed by optional degradations: dark mode, blueprint, blur / low resolution, noise, JPEG.

All operations are numpy / OpenCV on 512x512 uint8 images and cost a few milliseconds.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

WALL_CLASSES = (1, 2)
DOOR, WINDOW = 3, 4
STYLES = ("ink", "render", "recolor")


def _paper(rng: np.random.Generator) -> np.ndarray:
    return np.array([rng.integers(235, 256)] * 3, dtype=np.uint8) - rng.integers(0, 8, 3).astype(np.uint8)


def _ink_color(rng: np.random.Generator) -> np.ndarray:
    palette = [(20, 20, 20), (60, 60, 60), (110, 110, 110), (90, 50, 30), (30, 40, 110), (0, 0, 0)]
    return np.array(palette[rng.integers(len(palette))], dtype=np.uint8)


def ink_layer(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask of dark strokes (walls, furniture, text) with the light room fills dropped."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray < rng.integers(110, 175)


def _ink(img: np.ndarray, struct: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.empty_like(img)
    out[:] = _paper(rng)
    out[ink_layer(img, rng) | np.isin(struct, WALL_CLASSES)] = _ink_color(rng)
    return out


def _hatch(shape, rng: np.random.Generator) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    period = int(rng.integers(4, 9))
    return ((xx + yy) % period) < max(1, period // 3)


def _render(img: np.ndarray, struct: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    paper, ink = _paper(rng), _ink_color(rng)
    out = np.empty_like(img)
    out[:] = paper
    if rng.random() < 0.7:                         # keep furniture / text / fixtures as cues
        out[ink_layer(img, rng) & ~np.isin(struct, (1, 2, 3, 4))] = ink
    walls = np.isin(struct, WALL_CLASSES)
    if rng.random() < 0.3:                         # hatched walls with a solid outline
        edge = walls & ~cv2.erode(walls.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        out[(walls & _hatch(struct.shape, rng)) | edge] = ink
    else:
        out[walls] = ink
    # Windows: paper with two thin lines along the opening.
    win = (struct == WINDOW).astype(np.uint8)
    out[win.astype(bool)] = paper
    out[(cv2.morphologyEx(win, cv2.MORPH_GRADIENT, np.ones((2, 2), np.uint8)) > 0)] = ink
    # Doors: a gap in the wall plus a quarter-circle swing arc.
    door = (struct == DOOR).astype(np.uint8)
    out[door.astype(bool)] = paper
    n, _, stats, _ = cv2.connectedComponentsWithStats(door)
    for x, y, w, h, area in stats[1:]:
        if area < 6:
            continue
        r = int(max(w, h))
        if w >= h:   # horizontal opening: hinge at one end, arc above or below
            cx, cy = (x, y) if rng.random() < 0.5 else (x + w, y)
            start = 0 if cx == x else 90
            start += 0 if rng.random() < 0.5 else 180
        else:        # vertical opening
            cx, cy = (x, y) if rng.random() < 0.5 else (x, y + h)
            start = 90 if cy == y else 270
            start += 0 if rng.random() < 0.5 else 180
        cv2.ellipse(out, (int(cx), int(cy)), (r, r), 0, start, start + 90, ink.tolist(), 1, cv2.LINE_AA)
    return out


def _recolor(img: np.ndarray, struct: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = img.copy()
    out[np.isin(struct, WALL_CLASSES)] = _ink_color(rng)
    if rng.random() < 0.5:                         # also wash out the coloured room fills
        gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
        light = gray > 170
        out[light] = _paper(rng)
    return out


def _degrade(out: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.15:                        # dark-mode CAD
        out = 255 - out
    elif rng.random() < 0.15:                      # blueprint: white lines on blue
        gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        blue = np.array([25, 60, 140], np.float32)
        out = (blue[None, None] * gray[..., None] + 240 * (1 - gray[..., None])).astype(np.uint8)
    if rng.random() < 0.25:                        # low resolution scan / screenshot
        f = float(rng.uniform(0.35, 0.7))
        h, w = out.shape[:2]
        small = cv2.resize(out, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    elif rng.random() < 0.2:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    if rng.random() < 0.2:
        noise = rng.normal(0, rng.uniform(3, 10), out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.2:
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(30, 80))])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED)
    return out


def stylize(img: np.ndarray, struct: np.ndarray, rng: Optional[np.random.Generator] = None,
            style: Optional[str] = None) -> np.ndarray:
    """Return `img` (H, W, 3 uint8) re-drawn in a random style consistent with `struct`."""
    rng = rng or np.random.default_rng()
    style = style or STYLES[rng.integers(len(STYLES))]
    out = {"ink": _ink, "render": _render, "recolor": _recolor}[style](img, struct, rng)
    return _degrade(out, rng)
