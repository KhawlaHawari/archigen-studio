"""Original CMP class IDs in the *trained Pix2Pix* RGB encoding.

Recovered from aligned original index masks and eroded training-split JPEG mask
interiors. The original PNG display palette differs from the Berkeley encoding.
Do not confuse label 6 (sill) with label 10 (molding), or 5 (cornice) with 1 (sky).
JPEG medians differ by a few intensity levels; no validation/test labels used.
"""
import numpy as np

FACADE_STYLES = ("Modern Minimalist", "Mediterranean Stone", "Traditional Brick", "Lush Vegetation")
NEUTRAL_STYLE = 4

CMP_TRAINING_LUT = np.array([
    [0, 0, 222], [0, 0, 222], [0, 48, 255], [0, 128, 255], [1, 207, 255],
    [34, 255, 225], [112, 254, 144], [192, 255, 63], [253, 240, 6],
    [255, 159, 2], [254, 80, 0], [249, 2, 4], [174, 0, 1],
], dtype=np.uint8)

EXPERIMENT_CMP_PALETTE = {name: tuple(map(int, CMP_TRAINING_LUT[index])) for name, index in {
    'background': 1,  # Match the training semantics outside the physical facade.
    'facade': 2, 'window': 3, 'door': 4, 'cornice': 5, 'sill': 6,
    'balcony': 7, 'deco': 9, 'molding': 10, 'pillar': 11, 'base': 10,
}.items()}
