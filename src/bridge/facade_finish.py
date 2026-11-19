"""Apply material finishes to the wall pixels of a neutral facade prediction."""

import numpy as np
from PIL import Image

from src.data.facade_palette import EXPERIMENT_CMP_PALETTE


def apply_material_finish(image, sketch, bounds, style):
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    semantic = np.asarray(sketch.resize(image.size, Image.Resampling.NEAREST))
    wall = np.all(semantic == EXPERIMENT_CMP_PALETTE["facade"], axis=-1)
    height, width = wall.shape
    x_fraction = ((np.arange(width) + .5) / width - bounds["u_min"]) / (bounds["u_max"] - bounds["u_min"])
    y_fraction = ((np.arange(height) + .5) / height - (1 - bounds["v_max"])) / (bounds["v_max"] - bounds["v_min"])
    x = x_fraction[None, :] * bounds["wall_length_m"]
    y = y_fraction[:, None] * bounds["wall_height_m"]

    rng = np.random.default_rng(42)
    grain = rng.uniform(-5, 5, wall.shape)
    variation = Image.fromarray(rng.integers(0, 255, (16, 16), dtype=np.uint8)).resize(
        image.size, Image.Resampling.BILINEAR
    )
    mottling = (np.asarray(variation, dtype=np.float32) - 127.5) / 127.5
    # Retain low-frequency lighting from the generated texture.
    shading = np.clip((rgb.mean(axis=-1) - 180) * .28, -35, 20)
    detail = shading + grain + mottling * 9

    if style in ("Mediterranean Stone", "Traditional Brick"):
        brick = style == "Traditional Brick"
        block_width, course_height, joint = (.24, .078, .008) if brick else (.48, .23, .012)
        row = np.floor(y / course_height)
        staggered_x = x + (row % 2) * block_width / 2
        column = np.floor(staggered_x / block_width)
        across = staggered_x % block_width
        down = y % course_height
        pixel_width = bounds["wall_length_m"] / (width * (bounds["u_max"] - bounds["u_min"]))
        pixel_height = bounds["wall_height_m"] / (height * (bounds["v_max"] - bounds["v_min"]))
        mortar = np.clip(np.maximum(
            .5 + (joint / 2 - np.minimum(across, block_width - across)) / pixel_width,
            .5 + (joint / 2 - np.minimum(down, course_height - down)) / pixel_height,
        ), 0, 1)[..., None]
        block_variation = np.sin(row * 5.37 + column * 2.13) * (10 if brick else 15)
        color = np.array([157, 72, 49] if brick else [205, 169, 118])
        finish = (1 - mortar) * (color + block_variation[..., None]) + mortar * [184, 165, 143]
    elif style == "Lush Vegetation":
        growth = np.sin(x * 2.8 + y * 1.8) * .22 + np.cos(x * 5.3 - y * 2.7) * .12
        foliage = y_fraction[:, None] + growth + mottling * .2 > .53
        finish = np.where(foliage[..., None], [70, 105, 55], [197, 198, 176])
        detail += np.where(foliage, mottling * 25, 0)
    else:
        finish = np.array([226, 222, 211])

    rgb[wall] = np.broadcast_to(finish + detail[..., None], rgb.shape)[wall]
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
