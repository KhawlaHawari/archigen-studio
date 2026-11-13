"""
Topological Closing & Leakage Prevention (OS 4)
Solves the topological room bleeding (fuite topologique) by morphologically
closing door and window apertures before spatial polygonization.
"""

import numpy as np
from scipy import ndimage
from typing import Tuple


def apply_morphological_closing(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Applies binary dilation followed by erosion using a circular footprint"""
    radius = max(1, kernel_size // 2)
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    footprint = (x * x + y * y) <= (radius * radius)
    return ndimage.binary_closing(mask, structure=footprint).astype(np.uint8)


def bridge_wall_apertures(
    structure_mask: np.ndarray,
    door_class: int = 3,
    window_class: int = 4,
    wall_classes: Tuple[int, ...] = (1, 2),
    kernel_size: int = 7
) -> np.ndarray:
    """
    Creates an impermeable structural barrier for room zoning.
    Treats doors and windows as temporary wall bridges so that room regions
    cannot bleed into each other during connected-component analysis.
    
    Args:
        structure_mask: [H, W] integer array with values in [0, 4]
        door_class: index for doors (3)
        window_class: index for windows (4)
        wall_classes: indices for exterior (1) and interior (2) walls
        kernel_size: dilation radius to close door gaps
    
    Returns:
        closed_barrier: [H, W] boolean array of continuous perimeter walls
    """
    # 1. Base walls
    walls_binary = np.isin(structure_mask, wall_classes).astype(np.uint8)

    # 2. Openings (doors + windows)
    openings_binary = np.isin(structure_mask, (door_class, window_class)).astype(np.uint8)

    # 3. Morphological bridging across openings
    combined_envelope = (walls_binary | openings_binary).astype(np.uint8)
    closed_envelope = apply_morphological_closing(combined_envelope, kernel_size=kernel_size)

    # 4. Filter out any isolated noise (keep only connected boundaries)
    labeled, num_features = ndimage.label(closed_envelope)
    if num_features > 0:
        sizes = ndimage.sum(closed_envelope, labeled, range(num_features + 1))
        # Keep components with significant spatial footprint (> 50 pixels)
        mask_size = sizes >= 50
        closed_envelope = mask_size[labeled].astype(np.uint8)

    return closed_envelope


def isolate_closed_rooms(
    room_mask: np.ndarray,
    wall_barrier: np.ndarray,
    min_area_pixels: int = 100
) -> np.ndarray:
    """
    Constrains predicted room spaces within impermeable wall barriers,
    removing boundary spillage and tiny stray artifacts.
    """
    cleaned_rooms = room_mask.copy()
    # Force wall locations to background in room map
    cleaned_rooms[wall_barrier > 0] = 0

    # Clean small isolated islands per room category
    for c in np.unique(cleaned_rooms):
        if c == 0:
            continue
        c_mask = (cleaned_rooms == c).astype(np.uint8)
        # Small opening to sever thin topological bridges
        opened = ndimage.binary_opening(c_mask, structure=np.ones((3, 3)))
        labeled, num_comp = ndimage.label(opened)
        if num_comp > 0:
            sizes = ndimage.sum(opened, labeled, range(num_comp + 1))
            keep_mask = sizes >= min_area_pixels
            cleaned_rooms[(c_mask == 1) & (~keep_mask[labeled])] = 0

    return cleaned_rooms
