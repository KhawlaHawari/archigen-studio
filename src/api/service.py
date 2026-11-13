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
