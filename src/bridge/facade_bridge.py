"""
Facade Bridge Engine for ArchiGen 3D.

Ported from the original ArchiGen backend (src/bridge/facade_bridge.py, team project). Change in
this repository: the generator is THIS repository's Pix2Pix (src.models.pix2pix_cgan.FacadeGenerator,
ONNX or PyTorch), injected by src.api.service.ModelService through `ArchiGenFacadeInference`.

Bridges 2D Floorplan vectorization (walls, openings, dimensions) with
the Photorealistic Pix2Pix Facade Generator (CMP 12-class semantic canvas).

Features:
1. Real physical aspect ratio preservation (Length in meters x Floors * 2.8m).
2. Exact CMP 12-class RGB palette matching Pix2Pix training distribution.
3. Multi-floor vertical stacking: repeats window bays per floor; restricts doors to ground level.
4. Support for user-designated front facade (`is_front = True`).
5. Dual generation modes:
   - 'padded': Letterboxed 256x256 with exact UV sub-rectangle coordinates for Three.js.
   - 'tiled': Modular bay synthesis and seamless stitching for long exterior walls.
6. Frozen layout schema validation.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import io
import math
import base64
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from src.bridge.facade_finish import apply_material_finish
from src.data.facade_palette import FACADE_STYLES
from src.geometry.dimensions import DOOR_HEIGHT_M, WINDOW_HEIGHT_M, WINDOW_SILL_M

# Exact CMP 12-Class Colormap matching trained generator distribution
CMP_PALETTE = {
    "background": (0, 48, 255),    # Wall color for seamless padding (avoids border artifacts)
    "facade": (0, 48, 255),        # Main Wall
    "window": (0, 128, 255),       # Window Glass / Bay
    "door": (0, 207, 255),         # Entrance Door (true CMP Cyan, not dark red)
    "cornice": (0, 0, 222),        # Top Roof Trim / Cornice
    "sill": (255, 80, 0),          # Window Sill
    "balcony": (192, 255, 54),     # Balcony / Ledge
    "base": (0, 0, 222),           # Foundation / Bottom Base Molding
    "molding": (0, 0, 222),        # Architectural Horizontal Trim
    "pillar": (0, 128, 0),         # Pilasters / Vertical Columns
    "deco": (255, 255, 0)          # Window Lintels / Decorative Headers
}

def enhance_facade_details(
    img: Image.Image,
    radius: float = 1.2,
    percent: int = 125,
    threshold: int = 3,
    contrast_factor: float = 1.04
) -> Image.Image:
    """
    Applies contrast-adaptive unsharp masking and subtle micro-contrast pop to
    synthesized neural facade textures. Sharpens window frames, sill moldings,
    and stone/brick textures while preventing ringing artifacts or pixel clipping.
    """
    sharpened = img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))
    if contrast_factor != 1.0:
        enhancer = ImageEnhance.Contrast(sharpened)
        return enhancer.enhance(contrast_factor)
    return sharpened

def apply_facade_style(img: Image.Image, style: str) -> Image.Image:
    """
    Applies style-specific architectural color grading and textural enhancements
    to the generated photorealistic facade.
    """
    norm_style = (style or "").lower()
    arr = np.array(img, dtype=np.float32)
    h, w, _ = arr.shape

    if "brick" in norm_style:
        # Traditional Brick: rich warm terracotta/clay brick palette
        arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.15 + 10, 0, 255)
        arr[:, :, 1] = np.clip(arr[:, :, 1] * 0.94 - 4, 0, 255)
        arr[:, :, 2] = np.clip(arr[:, :, 2] * 0.86 - 8, 0, 255)
    elif "mediterranean" in norm_style or "stone" in norm_style:
        # Mediterranean Stone: warm sandstone, golden sunlight warmth
        arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.10 + 12, 0, 255)
        arr[:, :, 1] = np.clip(arr[:, :, 1] * 1.05 + 8, 0, 255)
        arr[:, :, 2] = np.clip(arr[:, :, 2] * 0.88 - 6, 0, 255)
    elif "vegetation" in norm_style or "lush" in norm_style:
        # Lush Vegetation: biophilic greening on lower facade and architectural columns
        y_grad = np.linspace(1.0, 0.0, h)[:, None, None]
        lower_mask = np.clip((y_grad - 0.25) * 1.6, 0.0, 1.0)
        arr[:, :, 1] = np.clip(arr[:, :, 1] + lower_mask[:, :, 0] * 28.0, 0, 255)
        arr[:, :, 0] = np.clip(arr[:, :, 0] - lower_mask[:, :, 0] * 10.0, 0, 255)
        arr[:, :, 2] = np.clip(arr[:, :, 2] - lower_mask[:, :, 0] * 14.0, 0, 255)
    else:
        # Modern Minimalist: clean white stucco, high contrast, crisp glass
        enhancer = ImageEnhance.Contrast(Image.fromarray(arr.astype(np.uint8)))
        enhanced = enhancer.enhance(1.08)
        arr = np.array(enhanced, dtype=np.float32)

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

class PyTorchFacadeInference:
    """
    Ultra-low-latency PyTorch CUDA Generator (~10ms per 256x256 facade).
    Provides native GPU execution when PyTorch with CUDA is available.
    """
    def __init__(self, generator: Any, device: str = "cuda", enhance: bool = True):
        self.generator = generator
        self.device = device
        self.enhance = enhance

    def generate(self, image_bytes: bytes) -> bytes:
        import torch
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((256, 256), Image.Resampling.BICUBIC)
        arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
        arr = np.transpose(arr, (2, 0, 1))
        tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.generator(tensor)
        out_arr = out[0].detach().cpu().numpy()
        out_arr = np.transpose(out_arr, (1, 2, 0))
        out_arr = np.clip((out_arr + 1.0) * 127.5, 0, 255).astype(np.uint8)
        out_img = Image.fromarray(out_arr)
        if self.enhance:
            out_img = enhance_facade_details(out_img)
        buf = io.BytesIO()
        out_img.save(buf, format="PNG")
        return buf.getvalue()

    def generate_batch(self, pil_images: List[Image.Image]) -> List[Image.Image]:
        """
        High-throughput tensor batching on GPU. Runs a single forward pass
        for all exterior wall sketches instead of looping sequentially.
        """
        import torch
        if not pil_images:
            return []
        tensors = []
        for img in pil_images:
            img_resized = img.resize((256, 256), Image.Resampling.BICUBIC)
            arr = np.array(img_resized, dtype=np.float32) / 127.5 - 1.0
            arr = np.transpose(arr, (2, 0, 1))
            tensors.append(torch.from_numpy(arr))

        batch_tensor = torch.stack(tensors, dim=0).to(self.device)
        with torch.no_grad():
            out = self.generator(batch_tensor)

        out_np = out.detach().cpu().numpy()
        out_np = np.transpose(out_np, (0, 2, 3, 1))
        out_np = np.clip((out_np + 1.0) * 127.5, 0, 255).astype(np.uint8)
        out_imgs = [Image.fromarray(out_np[i]) for i in range(len(pil_images))]
        if self.enhance:
            out_imgs = [enhance_facade_details(im) for im in out_imgs]
        return out_imgs

class ArchiGenFacadeInference:
    """Wraps any generator function mapping a [-1, 1] float32 batch (N, 3, 256, 256) to images in [-1, 1]."""
    resolution = 256

    def __init__(self, run, name: str = "ArchiGen Pix2Pix", enhance: bool = True):
        self.run = run
        self.name = name
        self.enhance = enhance

    def generate_batch(self, pil_images: List[Image.Image]) -> List[Image.Image]:
        if not pil_images:
            return []
        batch = np.stack([
            np.asarray(img.convert("RGB").resize((256, 256), Image.Resampling.NEAREST), dtype=np.float32) / 127.5 - 1.0
            for img in pil_images
        ]).transpose(0, 3, 1, 2)
        out = np.asarray(self.run(batch)).transpose(0, 2, 3, 1)
        imgs = [Image.fromarray(np.clip((o + 1.0) * 127.5, 0, 255).astype(np.uint8)) for o in out]
        return [enhance_facade_details(im) for im in imgs] if self.enhance else imgs

    def generate(self, image_bytes: bytes) -> bytes:
        img = self.generate_batch([Image.open(io.BytesIO(image_bytes))])[0]
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def describe(self, style=None) -> Dict[str, Any]:
        return {"model": self.name, "resolution": 256, "style_conditioned": False}


class FacadeBridge:
    def __init__(
        self,
        onnx_model_path: str = "models/saved/generator_perceptual_ema.onnx",
        pytorch_model_path: str = "models/saved/generator_perceptual_ema_best.pth",
        prefer_cuda: bool = True,
        enhance_textures: bool = True,
        inference_engine: Any = None,
        device_name: str = "CPU",
    ):
        self.onnx_model_path = Path(onnx_model_path)
        self.pytorch_model_path = Path(pytorch_model_path)
        self.prefer_cuda = prefer_cuda
        self.enhance_textures = enhance_textures
        self.inference_engine = inference_engine
        self.device_name = device_name
        if self.inference_engine is None:
            self._init_inference_engine()

    def _init_inference_engine(self):
        # Explicit experiment switch. Never silently relabel a legacy model as 512.
        experiment_path = os.environ.get("ARCHIGEN_FACADE_EXPERIMENT")
        if experiment_path:
            from src.bridge.facade_experimental import ExperimentalFacadeInference
            device = "cpu"
            if self.prefer_cuda and not experiment_path.lower().endswith(".onnx"):
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
            self.inference_engine = ExperimentalFacadeInference(experiment_path, device)
            self.enhance_textures = False
            self.device_name = f"Experimental 512 ({device})"
            return
        # 1. Try native PyTorch CUDA acceleration (<15ms per facade)
        if self.prefer_cuda:
            try:
                import torch
                if torch.cuda.is_available() and self.pytorch_model_path.exists():
                    from src.models.architecture import UNetGenerator
                    gen = UNetGenerator(in_channels=3, out_channels=3, upsample_mode="nearest")
                    ckpt = torch.load(self.pytorch_model_path, map_location="cuda", weights_only=True)
                    if isinstance(ckpt, dict) and "generator" in ckpt:
                        gen.load_state_dict(ckpt["generator"])
                    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                        gen.load_state_dict(ckpt["model_state_dict"])
                    else:
                        gen.load_state_dict(ckpt)
                    gen.to("cuda").eval()
                    self.inference_engine = PyTorchFacadeInference(gen, device="cuda", enhance=self.enhance_textures)
                    self.device_name = f"PyTorch CUDA ({torch.cuda.get_device_name(0)})"
                    print(f"[FACADE BRIDGE] Accelerated with {self.device_name} (~10ms latency)")
                    return
            except Exception as e:
                print(f"[FACADE BRIDGE] PyTorch CUDA acceleration skipped ({e}), attempting ONNX...")

        # 2. Fallback to ONNX Runtime Engine
        try:
            try:
                from app.inference import ArchiGenONNXInference
            except ImportError:
                from backend.app.inference import ArchiGenONNXInference
            
            onnx_target = self.onnx_model_path
            if not onnx_target.exists():
                for alt_name in ["generator_perceptual_ema.onnx", "generator_perceptual.onnx", "generator.onnx"]:
                    alt_path = onnx_target.parent / alt_name
                    if alt_path.exists():
                        onnx_target = alt_path
                        break

            if onnx_target.exists():
                self.inference_engine = ArchiGenONNXInference(str(onnx_target))
                self.device_name = "ONNX Runtime (CPU)"
                print(f"[FACADE BRIDGE] Loaded ONNX facade generator from {onnx_target}")
            else:
                print(f"[FACADE BRIDGE] Warning: ONNX model not found at {self.onnx_model_path}, mock mode active.")
        except Exception as e:
            print(f"[FACADE BRIDGE] Inference engine init deferred: {e}")

    @staticmethod
    def facade_frame(wall: Dict[str, Any]) -> Dict[str, Any]:
        """Canonical physical exterior face, shared by sketch, UVs and frames."""
        supplied = wall.get("facade_frame")
        if isinstance(supplied, dict):
            try:
                origin, direction, outward = (list(map(float, supplied[key])) for key in ("origin", "direction", "outward"))
                length = float(supplied["length_m"])
                if (all(len(vector) == 2 for vector in (origin, direction, outward))
                        and all(math.isfinite(v) for v in origin + direction + outward + [length])
                        and length > .01 and abs(math.hypot(*direction) - 1) < .001
                        and math.dist(outward, [-direction[1], direction[0]]) < .001):
                    return {"origin": origin, "direction": direction, "outward": outward, "length_m": length}
            except (KeyError, TypeError, ValueError):
                pass
        pts = wall.get("polygon_meters", [])
        if len(pts) < 2:
            pts = [[0.0, 0.0], [4.0, 0.0]]
        edges = [(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
        a, b = max(edges, key=lambda e: math.dist(e[0], e[1]))
        length = max(math.dist(a, b), 0.01)
        dx, dz = (b[0] - a[0]) / length, (b[1] - a[1]) / length
        area = sum(p[0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * p[1] for i, p in enumerate(pts))
        sign = 1 if area >= 0 else -1
        outward = [sign * dz, -sign * dx]
        # U increases to the right when looking at the wall from outside.
        return {"origin": list(b if sign > 0 else a), "direction": [outward[1], -outward[0]], "outward": outward, "length_m": length}

    @staticmethod
    def calculate_wall_length_m(wall: Dict[str, Any]) -> float:
        return FacadeBridge.facade_frame(wall)["length_m"]

    @classmethod
    def select_front_wall(cls, walls: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        candidates = [wall for wall in walls if wall.get("is_exterior") and len(wall.get("polygon_meters", [])) >= 3]
        if not candidates:
            return None
        longest = max(candidates, key=cls.calculate_wall_length_m)
        minimum = max(1.8, cls.calculate_wall_length_m(longest) * .65)
        entrances = [wall for wall in candidates if cls.calculate_wall_length_m(wall) >= minimum
                     and any(op.get("type") == "door" for op in wall.get("openings", []))]
        return max(entrances, key=cls.calculate_wall_length_m) if entrances else longest

    def synthesize_facade_sketch(self, wall, num_floors=1, floor_height_m=None, is_front=False, target_size=256):
        image, bounds, _, _ = self._build_facade_sketch(wall, num_floors, floor_height_m, is_front, target_size)
        return image, bounds

    def _build_facade_sketch(self, wall, num_floors=1, floor_height_m=None, is_front=False, target_size=256):
        """Draw the semantic canvas and return the exact opening rectangles in meters.

        Geometry is derived from the rasterized rectangles (including pixel rounding),
        so a 3D frame projects onto precisely the semantic input given to the cGAN.
        """
        from src.data.facade_palette import EXPERIMENT_CMP_PALETTE
        palette = EXPERIMENT_CMP_PALETTE if getattr(getattr(self, "inference_engine", None), "neural_style", False) else CMP_PALETTE
        frame = self.facade_frame(wall)
        wall_len_m = frame["length_m"]
        floor_height_m = floor_height_m if floor_height_m is not None else wall.get("height_m", 2.8)
        total_height_m = num_floors * floor_height_m

        # Target dimensions inside 256x256 canvas with margins
        margin_px = max(1, round(12 * target_size / 256))
        avail_w = target_size - 2 * margin_px
        avail_h = target_size - 2 * margin_px

        # Uniform scale factor (pixels per meter)
        scale_px_per_m = min(avail_w / wall_len_m, avail_h / total_height_m)
        detail_px = max(1, round(2 * target_size / 256))
        draw_w = int(round(wall_len_m * scale_px_per_m))
        draw_h = int(round(total_height_m * scale_px_per_m))

        # Center wall inside 256x256 canvas
        offset_x = (target_size - draw_w) // 2
        offset_y = (target_size - draw_h) // 2

        # UV coordinates for 3D mapping (normalized coordinates of active facade)
        uv_bounds = {
            "u_min": round(offset_x / target_size, 4),
            "u_max": round((offset_x + draw_w) / target_size, 4),
            "v_min": round((target_size - (offset_y + draw_h)) / target_size, 4),
            "v_max": round((target_size - offset_y) / target_size, 4),
            "wall_length_m": wall_len_m,
            "wall_height_m": total_height_m,
            "aspect_ratio": round(wall_len_m / total_height_m, 3)
        }

        # 1. Initialize Canvas with Sky/Background
        canvas = Image.new("RGB", (target_size, target_size), palette["background"])
        draw = ImageDraw.Draw(canvas)

        def safe_rect(bx0, by0, bx1, by1, f_color):
            rx0, rx1 = int(round(min(bx0, bx1))), int(round(max(bx0, bx1)))
            ry0, ry1 = int(round(min(by0, by1))), int(round(max(by0, by1)))
            if rx1 > rx0 and ry1 > ry0:
                draw.rectangle([rx0, ry0, rx1, ry1], fill=f_color)

        # 2. Draw Main Wall (Facade)
        wall_x0 = offset_x
        wall_y0 = offset_y
        wall_x1 = offset_x + draw_w
        wall_y1 = offset_y + draw_h
        draw.rectangle([wall_x0, wall_y0, wall_x1, wall_y1], fill=palette["facade"])

        # 3. Architectural Trim: Cornice at top, Base at bottom, Corner Pilasters
        cornice_h = max(int(round(0.25 * scale_px_per_m)), 4)
        base_h = max(int(round(0.30 * scale_px_per_m)), 5)
        draw.rectangle([wall_x0, wall_y0, wall_x1, wall_y0 + cornice_h], fill=palette["cornice"])
        draw.rectangle([wall_x0, wall_y1 - base_h, wall_x1, wall_y1], fill=palette["base"])

        # Corner pilasters (architectural column framing on facade edges using CMP molding class)
        pillar_w = max(int(round(0.28 * scale_px_per_m)), 3)
        if draw_w > (pillar_w * 5):
            safe_rect(wall_x0, wall_y0 + cornice_h, wall_x0 + pillar_w, wall_y1 - base_h, palette["molding"])
            safe_rect(wall_x1 - pillar_w, wall_y0 + cornice_h, wall_x1, wall_y1 - base_h, palette["molding"])

        # Detected positions are projected onto the same finite wall face used
        # for UVs. Openings belonging to another side of a compound wall are excluded.
        attached = wall.get("openings", [])
        templates = []
        for idx, op in enumerate(attached):
            kind = op.get("type")
            if kind not in ("door", "window"):
                continue
            along = op.get("position_along_wall_m")
            if along is None and op.get("position_meters") is not None:
                px, pz = op["position_meters"]
                ox, oz = frame["origin"]
                dx, dz = frame["direction"]
                along = (px - ox) * dx + (pz - oz) * dz
                distance = abs((px - ox) * dz - (pz - oz) * dx)
                if distance > 0.85 or along < 0 or along > wall_len_m:
                    continue
            if along is None:
                continue
            templates.append({
                "id": op.get("id", f"{kind}_{idx}"), "type": kind,
                "position_along_wall_m": float(along),
                "width_m": float(op.get("width_m", op.get("width", 1.15 if kind == "door" else 1.10))),
                "height_m": float(op.get("height_m", op.get("height", DOOR_HEIGHT_M if kind == "door" else WINDOW_HEIGHT_M))),
                "sill_m": float(op.get("sill_m", 0.0 if kind == "door" else WINDOW_SILL_M)),
            })
        # Procedural preview only when no measured openings were supplied.
        if not attached and not wall.get("geometry_fixed"):
            if is_front:
                templates.append({"id": "facade_door", "type": "door", "position_along_wall_m": wall_len_m / 2, "width_m": 1.15, "height_m": DOOR_HEIGHT_M, "sill_m": 0.0})
            bays = max(int(round(wall_len_m / 2.8)), 1)
            for i in range(bays):
                along = (i + 1) * wall_len_m / (bays + 1)
                if is_front and abs(along - wall_len_m / 2) < 1.3:
                    continue
                templates.append({"id": f"facade_window_{i}", "type": "window", "position_along_wall_m": along, "width_m": 1.1, "height_m": WINDOW_HEIGHT_M, "sill_m": WINDOW_SILL_M})

        rendered_openings = []
        for fl in range(num_floors):
            if fl > 0:
                y = wall_y1 - fl * floor_height_m / total_height_m * draw_h
                safe_rect(wall_x0, y - detail_px, wall_x1, y, palette["molding"])
            for op in templates:
                if op["type"] == "door" and fl > 0:
                    continue
                cx = wall_x0 + op["position_along_wall_m"] / wall_len_m * draw_w
                half = op["width_m"] / wall_len_m * draw_w / 2
                x0 = max(wall_x0 + 1, int(round(cx - half)))
                x1 = min(wall_x1 - 1, int(round(cx + half)))
                sill = fl * floor_height_m + op["sill_m"]
                y1 = min(wall_y1, int(round(wall_y1 - sill / total_height_m * draw_h)))
                y0 = max(wall_y0 + 1, int(round(wall_y1 - (sill + op["height_m"]) / total_height_m * draw_h)))
                if x1 <= x0 or y1 <= y0:
                    continue
                trim = detail_px
                safe_rect(x0 - trim, y0 - trim, x1 + trim, y0, palette["deco"])
                safe_rect(x0 - trim, y0, x0, y1, palette["molding"])
                safe_rect(x1, y0, x1 + trim, y1, palette["molding"])
                if op["type"] == "window":
                    safe_rect(x0 - trim, y1, x1 + trim, y1 + detail_px, palette["sill"])
                safe_rect(x0, y0, x1, y1, palette[op["type"]])
                rendered_openings.append({
                    **op, "floor_index": fl,
                    "position_along_wall_m": ((x0 + x1) / 2 - wall_x0) / draw_w * wall_len_m,
                    "width_m": (x1 - x0) / draw_w * wall_len_m,
                    "height_m": (y1 - y0) / draw_h * total_height_m,
                    "sill_m": (wall_y1 - y1) / draw_h * total_height_m,
                    "uv_rect": [x0 / target_size, 1 - y1 / target_size, x1 / target_size, 1 - y0 / target_size],
                })

        return canvas, uv_bounds, frame, rendered_openings

    def _finish_texture(self, image, sketch, bounds, style, material_finish=False):
        engine = self.inference_engine
        info = engine.describe(style) if hasattr(engine, "describe") else {
            "model": "legacy_256", "resolution": 256, "style_conditioned": False,
        }
        if getattr(engine, "neural_style", False):
            if material_finish and style in FACADE_STYLES and not info.get("style_conditioned"):
                image = apply_material_finish(image, sketch, bounds, style)
                info = {**info, "finish": style}
        elif style:
            image = apply_facade_style(image, style)
        return image, info

    def render_wall_facade(
        self,
        wall: Dict[str, Any],
        num_floors: int = 1,
        floor_height_m: Optional[float] = None,
        is_front: bool = False,
        style: Optional[str] = "Modern Minimalist",
        output_path: Optional[str] = None,
        material_finish: bool = False,
    ) -> Dict[str, Any]:
        """
        Renders photorealistic facade texture for a specific exterior wall.
        Runs:
        1. Semantic sketch synthesis (aspect-ratio preserved).
        2. Pix2Pix ONNX neural generation.
        3. Architectural style post-processing & enhancement.
        4. Returns texture image, sketch image, and UV coordinates for Three.js.
        """
        sketch_img, uv_bounds, frame, openings = self._build_facade_sketch(
            wall=wall,
            num_floors=num_floors,
            floor_height_m=floor_height_m,
            is_front=is_front,
            target_size=getattr(self.inference_engine, "resolution", 256)
        )

        # Convert sketch to PNG bytes
        buf = io.BytesIO()
        sketch_img.save(buf, format="PNG")
        sketch_bytes = buf.getvalue()

        # Generate photorealistic texture
        if self.inference_engine is not None:
            if getattr(self.inference_engine, "neural_style", False):
                photo_bytes = self.inference_engine.generate(sketch_bytes, style=style)
            else:
                photo_bytes = self.inference_engine.generate(sketch_bytes)
            photo_img = Image.open(io.BytesIO(photo_bytes))
            if self.enhance_textures and not isinstance(self.inference_engine, (PyTorchFacadeInference, ArchiGenFacadeInference)):
                photo_img = enhance_facade_details(photo_img)
        else:
            # Fallback mock texture if ONNX engine not loaded
            photo_img = sketch_img

        photo_img, model_info = self._finish_texture(photo_img, sketch_img, uv_bounds, style, material_finish)

        buf_enh = io.BytesIO()
        photo_img.save(buf_enh, format="PNG")
        photo_bytes = buf_enh.getvalue()

        if output_path:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            photo_img.save(out_file)

        return {
            "wall_id": wall.get("id", "unknown_wall"),
            "is_front": is_front,
            "style": style,
            "model_info": model_info,
            "sketch_image": sketch_img,
            "sketch_bytes": sketch_bytes,
            "photo_image": photo_img,
            "photo_bytes": photo_bytes,
            "uv_bounds": uv_bounds,
            "facade_frame": frame,
            "openings": openings,
            "wall_length_m": uv_bounds["wall_length_m"],
            "wall_height_m": uv_bounds["wall_height_m"]
        }

    def process_layout_facades(
        self,
        layout: Dict[str, Any],
        front_wall_id: Optional[str] = None,
        num_floors: int = 1,
        front_only: bool = True,
        style: str = "Modern Minimalist",
        save_to_disk: bool = False,
        output_dir: str = "models/samples/facades"
    ) -> Dict[str, Any]:
        """
        Processes exterior walls in a layout:
        1. Selects or validates front wall (`front_wall_id`).
        2. If `front_only=True` (default for real-time web & defense):
           Renders photorealistic neural facade strictly for the front wall (~24ms).
           Other exterior walls receive procedural material mappings and UV coordinates.
        3. If `front_only=False`:
           Uses GPU tensor batching `[N, 3, 256, 256]` to generate all exterior wall textures
           in a single forward pass without sequential loop overhead.
        4. Annotates layout metadata with generated facade texture specs.
        """
        out_path = Path(output_dir) if save_to_disk else None
        if out_path:
            out_path.mkdir(parents=True, exist_ok=True)

        walls = layout.get("walls", [])
        exterior_walls = [w for w in walls if w.get("is_exterior", False)]

        if not any(w.get("id") == front_wall_id for w in exterior_walls):
            front = self.select_front_wall(exterior_walls)
            front_wall_id = front.get("id") if front else None

        results = {}

        if front_only and len(exterior_walls) > 0:
            # ON-DEMAND FRONT FACADE MODE (Eliminates P95 latency tail: 710ms -> 34ms)
            for w in exterior_walls:
                w_id = w.get("id")
                is_front = (w_id == front_wall_id)
                w["is_front"] = is_front

                if is_front:
                    tex_filepath = str(out_path / f"{w_id}_facade.png") if out_path else None
                    res = self.render_wall_facade(
                        wall={**w, "geometry_fixed": True},
                        num_floors=num_floors,
                        is_front=True,
                        style=style,
                        output_path=tex_filepath
                    )
                    b64_str = base64.b64encode(res["photo_bytes"]).decode("ascii")
                    sketch_b64 = base64.b64encode(res["sketch_bytes"]).decode("ascii")
                    w["facade_texture"] = {
                        "texture_file": f"{w_id}_facade.png",
                        "texture_path": tex_filepath,
                        "texture_base64": f"data:image/png;base64,{b64_str}",
                        "sketch_base64": f"data:image/png;base64,{sketch_b64}",
                        "style": style,
                        "model_info": res.get("model_info"),
                        "wall_id": w_id,
                        "num_floors": num_floors,
                        "uv_bounds": res["uv_bounds"],
                        "facade_frame": res["facade_frame"],
                        "openings": res["openings"],
                        "wall_height_m": res["wall_height_m"],
                        "is_front": True,
                        "mode": "neural_generated"
                    }
                    results[w_id] = res
                else:
                    # Procedural / on-demand placeholder
                    _, uv_bounds = self.synthesize_facade_sketch(w, num_floors=num_floors, is_front=False)
                    w["facade_texture"] = {
                        "uv_bounds": uv_bounds,
                        "is_front": False,
                        "mode": "procedural_exterior"
                    }
        elif len(exterior_walls) > 0:
            # BATCHED MULTI-WALL MODE (All exterior walls in 1 GPU forward pass)
            sketches = []
            wall_order = []
            for w in exterior_walls:
                w_id = w.get("id")
                is_front = (w_id == front_wall_id)
                w["is_front"] = is_front
                sketch_img, uv_bounds, frame, openings = self._build_facade_sketch({**w, "geometry_fixed": True}, num_floors=num_floors, is_front=is_front, target_size=getattr(self.inference_engine, "resolution", 256))
                sketches.append(sketch_img)
                wall_order.append((w, sketch_img, uv_bounds, frame, openings))

            if hasattr(self.inference_engine, "generate_batch"):
                generated_imgs = self.inference_engine.generate_batch(sketches, style=style) if getattr(self.inference_engine, "neural_style", False) else self.inference_engine.generate_batch(sketches)
            else:
                generated_imgs = []
                for sketch in sketches:
                    if self.inference_engine is None:
                        generated_imgs.append(sketch)
                    else:
                        buffer = io.BytesIO()
                        sketch.save(buffer, format="PNG")
                        generated_imgs.append(Image.open(io.BytesIO(self.inference_engine.generate(buffer.getvalue()))).convert("RGB"))
                if self.enhance_textures:
                    generated_imgs = [enhance_facade_details(im) for im in generated_imgs]

            for idx, (w, s_img, uv_bounds, frame, openings) in enumerate(wall_order):
                w_id = w.get("id")
                g_img, model_info = self._finish_texture(generated_imgs[idx], s_img, uv_bounds, style)
                buf = io.BytesIO()
                g_img.save(buf, format="PNG")
                photo_bytes = buf.getvalue()

                tex_filepath = None
                if out_path:
                    tex_filepath = str(out_path / f"{w_id}_facade.png")
                    g_img.save(tex_filepath)

                b64_str = base64.b64encode(photo_bytes).decode("ascii")
                res = {
                    "wall_id": w_id,
                    "model_info": model_info,
                    "is_front": (w_id == front_wall_id),
                    "sketch_image": s_img,
                    "photo_image": g_img,
                    "photo_bytes": photo_bytes,
                    "uv_bounds": uv_bounds,
                    "facade_frame": frame,
                    "openings": openings,
                    "wall_length_m": uv_bounds["wall_length_m"],
                    "wall_height_m": uv_bounds["wall_height_m"]
                }
                w["facade_texture"] = {
                    "texture_file": f"{w_id}_facade.png",
                    "texture_path": tex_filepath,
                    "texture_base64": f"data:image/png;base64,{b64_str}",
                    "model_info": res.get("model_info"),
                    "wall_id": w_id,
                    "num_floors": num_floors,
                    "uv_bounds": uv_bounds,
                    "facade_frame": frame,
                    "openings": openings,
                    "wall_height_m": uv_bounds["wall_height_m"],
                    "is_front": (w_id == front_wall_id),
                    "mode": "neural_batch"
                }
                results[w_id] = res

        layout["metadata"]["front_wall_id"] = front_wall_id
        layout["metadata"]["num_floors"] = num_floors
        layout["metadata"]["facade_inference_device"] = self.device_name
        layout["metadata"]["facade_mode"] = "front_only" if front_only else "batched_all"

        return {
            "front_wall_id": front_wall_id,
            "exterior_walls_count": len(exterior_walls),
            "inference_device": self.device_name,
            "facade_results": results,
            "updated_layout": layout
        }
