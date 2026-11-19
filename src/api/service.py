"""
Inference Service Layer & Pipeline Manager
Manages ONNX Runtime sessions, PyTorch fallbacks, pre/post-processing,
colorization, and 3D mesh generation.
"""

import io
import time
import base64
import numpy as np
from PIL import Image
import torch
import onnxruntime as ort
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from src.training.checkpoints import load_weights
from src.config import (
    FLOORPLAN_CFG,
    FACADE_CFG,
    STRUCTURE_CLASSES,
    STRUCTURE_COLORS,
    ROOM_CLASSES,
    ROOM_COLORS,
    GEOMETRY_CFG
)
from src.data.svg_preprocessor import (
    isotropic_letterbox,
    generate_synthetic_cubicasa_sample,
    generate_synthetic_facade_sample
)
from src.geometry.vectorizer_3d import vectorize_floorplan_pipeline
from src.geometry.layout import layout_from_probabilities
from src.geometry.layout_vectorizer import ROOM_NAMES, score_room_polygon
from src.models.unet_plus_plus import UNetPlusPlusMultiTask
from src.models.pix2pix_cgan import FacadeGenerator
from src.mlops.drift_detector import (
    extract_floorplan_features,
    extract_facade_features,
    log_inference_telemetry
)


class ModelService:
    """Singleton inference manager for Floorplan & Facade models"""
    _instance = None

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ort_floorplan = None
        self.ort_facade = None
        self.pt_floorplan = None
        self.pt_facade = None
        self.start_time = time.time()
        self._initialize_sessions()

    @classmethod
    def get_instance(cls) -> "ModelService":
        if cls._instance is None:
            cls._instance = ModelService()
        return cls._instance

    def _initialize_sessions(self):
        """Initializes ONNX sessions or falls back to PyTorch"""
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
        
        # 1. Floorplan Model
        if FLOORPLAN_CFG.onnx_path.exists():
            try:
                opts = ort.SessionOptions()
                self.ort_floorplan = ort.InferenceSession(str(FLOORPLAN_CFG.onnx_path), opts, providers=providers)
                print(f"[ModelService] Loaded Floorplan ONNX: {FLOORPLAN_CFG.onnx_path}")
            except Exception as e:
                print(f"[ModelService] Failed to load Floorplan ONNX: {e}. Falling back to PyTorch.")
        
        if self.ort_floorplan is None:
            self.pt_floorplan = UNetPlusPlusMultiTask(
                in_channels=FLOORPLAN_CFG.in_channels,
                num_structure_classes=FLOORPLAN_CFG.num_structure_classes,
                num_room_classes=FLOORPLAN_CFG.num_room_classes
            )
            if FLOORPLAN_CFG.checkpoint_path.exists():
                try:
                    load_weights(self.pt_floorplan, FLOORPLAN_CFG.checkpoint_path)
                    print("[ModelService] Loaded Floorplan PyTorch checkpoint.")
                except Exception as e:
                    print(f"[ModelService] Floorplan checkpoint corrupted ({e}). Using randomly initialized weights.")
            self.pt_floorplan.eval()
            print("[ModelService] Initialized Floorplan PyTorch model.")

        # 2. Facade Model
        if FACADE_CFG.onnx_path.exists():
            try:
                opts = ort.SessionOptions()
                self.ort_facade = ort.InferenceSession(str(FACADE_CFG.onnx_path), opts, providers=providers)
                print(f"[ModelService] Loaded Facade ONNX: {FACADE_CFG.onnx_path}")
            except Exception as e:
                print(f"[ModelService] Failed to load Facade ONNX: {e}. Falling back to PyTorch.")

        if self.ort_facade is None:
            self.pt_facade = FacadeGenerator(
                in_channels=FACADE_CFG.in_channels,
                out_channels=FACADE_CFG.out_channels,
                ngf=FACADE_CFG.ngf
            )
            if FACADE_CFG.checkpoint_path.exists():
                try:
                    load_weights(self.pt_facade, FACADE_CFG.checkpoint_path)
                    print("[ModelService] Loaded Facade PyTorch checkpoint.")
                except Exception as e:
                    print(f"[ModelService] Facade checkpoint corrupted ({e}). Using randomly initialized weights.")
            self.pt_facade.eval()
            print("[ModelService] Initialized Facade PyTorch model.")

    def colorize_mask(self, mask: np.ndarray, color_dict: Dict[int, Tuple[int, int, int]]) -> np.ndarray:
        """Applies RGB color map to single-channel categorical mask"""
        h, w = mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for class_id, color in color_dict.items():
            rgb[mask == class_id] = color
        return rgb

    def numpy_to_base64_png(self, img_array: np.ndarray) -> str:
        """Encodes RGB numpy array to base64 PNG data URL string"""
        pil_img = Image.fromarray(img_array.astype(np.uint8))
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    def predict_floorplan(self, input_image: Image.Image) -> Dict[str, Any]:
        """
        Runs segmentation and 3D vectorization on input floorplan image.
        """
        t0 = time.perf_counter()
        raw_rgb = np.array(input_image.convert("RGB"))
        orig_h, orig_w, _ = raw_rgb.shape

        # Letterbox to 512x512
        boxed_img, meta = isotropic_letterbox(raw_rgb, target_size=(512, 512), is_mask=False, fill_value=255)

        # Normalize to [0, 1] tensor shape [1, 3, 512, 512]
        norm_tensor = (boxed_img.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]

        # Inference
        if self.ort_floorplan is not None:
            ort_outs = self.ort_floorplan.run(None, {"image": norm_tensor})
            struct_logits, room_logits = ort_outs[0], ort_outs[1]
        else:
            with torch.no_grad():
                t_in = torch.from_numpy(norm_tensor)
                s_out, r_out = self.pt_floorplan(t_in)
                struct_logits, room_logits = s_out.numpy(), r_out.numpy()

        # Extract argmax predictions [512, 512]
        struct_mask = np.argmax(struct_logits[0], axis=0).astype(np.uint8)
        room_mask = np.argmax(room_logits[0], axis=0).astype(np.uint8)

        # 3D Vectorization & Room areas calculation
        vector_data = vectorize_floorplan_pipeline(
            structure_mask=struct_mask,
            room_mask=room_mask,
            scale_m_per_px=GEOMETRY_CFG.scale_meters_per_pixel
        )

        # Colorize masks for web rendering
        struct_rgb = self.colorize_mask(struct_mask, STRUCTURE_COLORS)
        room_rgb = self.colorize_mask(room_mask, ROOM_COLORS)

        t1 = time.perf_counter()
        proc_time = round((t1 - t0) * 1000.0, 2)

        # MLOps Level 3: Telemetry logging for continuous drift monitoring
        try:
            feat = extract_floorplan_features(
                image_np=boxed_img,
                struct_mask=struct_mask,
                room_mask=room_mask,
                total_area_m2=vector_data["total_habitable_area_m2"]
            )
            log_inference_telemetry(domain="floorplan", features=feat)
        except Exception as e:
            pass
        return {
            "image_width": orig_w,
            "image_height": orig_h,
            "total_habitable_area_m2": vector_data["total_habitable_area_m2"],
            "num_doors_detected": vector_data["num_doors_detected"],
            "num_windows_detected": vector_data["num_windows_detected"],
            "rooms": vector_data["rooms"],
            "meshes": vector_data["meshes"],
            "structure_mask_base64": self.numpy_to_base64_png(struct_rgb),
            "room_mask_base64": self.numpy_to_base64_png(room_rgb),
            "processing_time_ms": proc_time,
        }

    def _variant_model(self, variant: str):
        """PyTorch model of an A/B challenger checkpoint (models/checkpoints/floorplan_unetpp_<variant>.pt), loaded once."""
        cache = self.__dict__.setdefault("_variant_models", {})
        if variant not in cache:
            from src.api.ab_testing import ABRouter
            model = UNetPlusPlusMultiTask(in_channels=FLOORPLAN_CFG.in_channels,
                                          num_structure_classes=FLOORPLAN_CFG.num_structure_classes,
                                          num_room_classes=FLOORPLAN_CFG.num_room_classes)
            load_weights(model, ABRouter.checkpoint(variant), require_real_data=True)
            cache[variant] = model.eval()
        return cache[variant]

    def _floorplan_probabilities(self, boxed_img: np.ndarray, variant: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Softmax probabilities (C, H, W) for the structure and room heads on a 512x512 letterboxed RGB image.

        variant=None (or the champion) uses the production session (ONNX if exported); another name runs
        that challenger checkpoint (A/B testing)."""
        from src.api.ab_testing import ROUTER
        norm_tensor = (boxed_img.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]
        if variant not in (None, ROUTER.champion):
            with torch.no_grad():
                struct_logits, room_logits = self._variant_model(variant)(torch.from_numpy(norm_tensor))
        elif self.ort_floorplan is not None:
            struct_logits, room_logits = self.ort_floorplan.run(None, {"image": norm_tensor})[:2]
            struct_logits, room_logits = torch.from_numpy(struct_logits), torch.from_numpy(room_logits)
        else:
            with torch.no_grad():
                struct_logits, room_logits = self.pt_floorplan(torch.from_numpy(norm_tensor))
        prob_s = torch.softmax(struct_logits, dim=1)[0].numpy()
        prob_r = torch.softmax(room_logits, dim=1)[0].numpy()
        return prob_s, prob_r

    def analyze_floorplan_layout(
        self,
        input_image: Image.Image,
        scale_override: Optional[float] = None,
        reference_rooms: Optional[list] = None,
        variant: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Segments a floor plan and vectorizes it into the wall / room / opening layout
        that the web studio extrudes into a 3D house. variant selects the A/B model (None = champion).
        """
        t0 = time.perf_counter()
        raw_rgb = np.array(input_image.convert("RGB"))
        boxed_img, _ = isotropic_letterbox(raw_rgb, target_size=(512, 512), is_mask=False, fill_value=255)
        prob_s, prob_r = self._floorplan_probabilities(boxed_img, variant)
        layout = layout_from_probabilities(prob_s, prob_r, boxed_img, scale_override=scale_override)

        reference_confidences = None
        if reference_rooms is not None:
            class_ids = {name: idx for idx, name in ROOM_NAMES.items() if idx}
            reference_confidences = {}
            for region in reference_rooms:
                class_id = class_ids.get(region.get("room_type"))
                polygon = region.get("polygon_px") or []
                reference_confidences[region.get("id")] = (
                    score_room_polygon(prob_r, class_id, polygon) if class_id is not None and len(polygon) >= 3 else None
                )

        rooms = layout.get("rooms", [])
        return {
            "total_area_m2": round(sum(r.get("surface_m2", 0.0) for r in rooms), 2),
            "num_rooms": len(rooms),
            "rooms": [{**r, "polygon_points": r["polygon_meters"]} for r in rooms],
            "walls": layout.get("walls", []),
            "doors": layout.get("doors", []),
            "windows": layout.get("windows", []),
            "envelope": layout.get("envelope", []),
            "metadata": layout.get("metadata", {}),
            "reference_room_confidences": reference_confidences,
            "processing_time_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    def _run_facade_generator(self, batch: np.ndarray) -> np.ndarray:
        """(N, 3, 256, 256) label maps in [-1, 1] -> generated facades in [-1, 1]."""
        batch = batch.astype(np.float32)
        if self.ort_facade is not None:
            return np.concatenate([self.ort_facade.run(None, {"semantic_mask": b[None]})[0] for b in batch])
        with torch.no_grad():
            return self.pt_facade(torch.from_numpy(batch)).numpy()

    # ---- High-quality facades: ControlNet fine-tuned in Run 3 (models/controlnet_facade) ----
    CONTROLNET_DIR = Path(__file__).resolve().parents[2] / "models" / "controlnet_facade"
    CONTROLNET_STEPS_CPU, CONTROLNET_STEPS_GPU = 20, 25

    def controlnet_status(self) -> Dict[str, Any]:
        if not (self.CONTROLNET_DIR / "config.json").exists():
            return {"available": False, "reason": "No fine-tuned ControlNet yet: unzip the Run 3 results so that "
                                                  "models/controlnet_facade/ exists, then restart the API."}
        try:
            import diffusers  # noqa: F401
        except ImportError:
            return {"available": False, "reason": "pip install -r requirements-diffusion.txt"}
        steps = self.CONTROLNET_STEPS_GPU if torch.cuda.is_available() else self.CONTROLNET_STEPS_CPU
        return {"available": True, "device": self.device, "steps": steps, "resolution": 512,
                "expected_seconds_per_tile": 4 if torch.cuda.is_available() else 380,   # measured: 383 s on the dev laptop CPU
                "note": "The first use downloads Stable Diffusion 1.5 (~4 GB) once."}

    def controlnet_generate_batch(self, tiles: list) -> list:
        """CMP label-map tiles (256 px PIL) -> 512 px facades with the fine-tuned ControlNet."""
        from src.evaluation.evaluate_controlnet import load_pipeline
        from src.training.train_controlnet import NEGATIVE, PROMPT
        if getattr(self, "_controlnet_pipe", None) is None:
            self._controlnet_pipe = load_pipeline(str(self.CONTROLNET_DIR), device=self.device)
        pipe = self._controlnet_pipe
        steps = self.CONTROLNET_STEPS_GPU if self.device == "cuda" else self.CONTROLNET_STEPS_CPU
        conds = [t.convert("RGB").resize((512, 512), Image.Resampling.NEAREST) for t in tiles]
        gens = [torch.Generator(device=self.device).manual_seed(42 + i) for i in range(len(conds))]
        with torch.no_grad():
            return pipe([PROMPT] * len(conds), image=conds, negative_prompt=[NEGATIVE] * len(conds),
                        num_inference_steps=steps, guidance_scale=5.0, generator=gens).images

    def facade_bridge(self):
        """Wall -> CMP label map -> THIS repository's Pix2Pix (bridge ported from the original ArchiGen)."""
        if getattr(self, "_facade_bridge", None) is None:
            from src.bridge.facade_bridge import ArchiGenFacadeInference, FacadeBridge
            runtime = "ONNX Runtime" if self.ort_facade is not None else "PyTorch"
            engine = ArchiGenFacadeInference(self._run_facade_generator, name=f"ArchiGen Pix2Pix ({runtime})")
            self._facade_bridge = FacadeBridge(inference_engine=engine, device_name=f"{runtime} ({self.device})")
        return self._facade_bridge

    def predict_facade(self, mask_image: Image.Image) -> Dict[str, Any]:
        """
        Runs Pix2Pix generation on input facade semantic mask (256x256).
        """
        t0 = time.perf_counter()
        mask_rgb = np.array(mask_image.convert("RGB").resize((256, 256), Image.Resampling.NEAREST))

        # Normalize to [-1, 1] shape [1, 3, 256, 256]
        norm_mask = ((mask_rgb.astype(np.float32) / 127.5) - 1.0).transpose(2, 0, 1)[np.newaxis, ...]

        # Inference
        if self.ort_facade is not None:
            ort_out = self.ort_facade.run(None, {"semantic_mask": norm_mask})[0]
            gen_tensor = ort_out[0]
        else:
            with torch.no_grad():
                t_in = torch.from_numpy(norm_mask)
                gen_tensor = self.pt_facade(t_in).numpy()[0]

        # Denormalize output from [-1, 1] to [0, 255] RGB
        gen_rgb = np.clip(((gen_tensor.transpose(1, 2, 0) + 1.0) * 127.5), 0, 255).astype(np.uint8)

        t1 = time.perf_counter()
        proc_time = round((t1 - t0) * 1000.0, 2)

        # MLOps Level 3: Telemetry logging for continuous drift monitoring
        try:
            feat = extract_facade_features(
                mask_np=mask_rgb,
                generated_np=gen_rgb
            )
            log_inference_telemetry(domain="facade", features=feat)
        except Exception as e:
            pass
        return {
            "width": 256,
            "height": 256,
            "input_mask_base64": self.numpy_to_base64_png(mask_rgb),
            "generated_facade_base64": self.numpy_to_base64_png(gen_rgb),
            "processing_time_ms": proc_time,
        }

    def get_synthetic_floorplan(self) -> Dict[str, Any]:
        """Generates demonstration floorplan sample and runs pipeline"""
        img, _, _ = generate_synthetic_cubicasa_sample(512, 512)
        pil_img = Image.fromarray(img)
        res = self.predict_floorplan(pil_img)
        res["input_image_base64"] = self.numpy_to_base64_png(img)
        return res

    def get_synthetic_facade(self) -> Dict[str, Any]:
        """Generates demonstration facade sample and runs pipeline"""
        mask, _ = generate_synthetic_facade_sample(256, 256)
        pil_mask = Image.fromarray(mask)
        return self.predict_facade(pil_mask)
