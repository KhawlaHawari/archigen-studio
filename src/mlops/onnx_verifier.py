"""
ONNX Runtime Strict Parity Verifier (OS 5)
Verifies numerical parity between PyTorch eager execution and ONNX Runtime inference:
Requirement: Max absolute error Delta = |Y_PyTorch - Y_ONNX| < 10^-5
"""

import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Dict, Any
import json

from src.training.checkpoints import load_weights
from src.config import (
    FLOORPLAN_CFG,
    FACADE_CFG,
    MLOPS_CFG,
    ONNX_DIR
)
from src.models.unet_plus_plus import UNetPlusPlusMultiTask
from src.models.pix2pix_cgan import FacadeGenerator


def verify_floorplan_parity(
    checkpoint_path: Path = FLOORPLAN_CFG.checkpoint_path,
    onnx_path: Path = FLOORPLAN_CFG.onnx_path,
    atol: float = MLOPS_CFG.parity_atol_threshold
) -> Dict[str, Any]:
    """Tests numerical parity on UNet++ floorplan model"""
    print(f"\n--> Verifying PyTorch vs ONNX Parity for Floorplan UNet++")
    
    # 1. Load PyTorch model
    pt_model = UNetPlusPlusMultiTask(
        in_channels=FLOORPLAN_CFG.in_channels,
        num_structure_classes=FLOORPLAN_CFG.num_structure_classes,
        num_room_classes=FLOORPLAN_CFG.num_room_classes
    )
    load_weights(pt_model, checkpoint_path)
    pt_model.eval()

    # 2. Setup ONNX Runtime session
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 2
    opts.intra_op_num_threads = 2
    ort_session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])

    # 3. Generate deterministic test input
    torch.manual_seed(42)
    test_tensor = torch.randn(1, 3, 512, 512, dtype=torch.float32)
    test_numpy = test_tensor.numpy()

    # 4. PyTorch forward
    with torch.no_grad():
        pt_struct, pt_rooms = pt_model(test_tensor)
        pt_struct_np = pt_struct.numpy()
        pt_rooms_np = pt_rooms.numpy()

    # 5. ONNX Runtime forward
    ort_inputs = {"image": test_numpy}
    ort_outputs = ort_session.run(None, ort_inputs)
    ort_struct_np, ort_rooms_np = ort_outputs[0], ort_outputs[1]

    # 6. Calculate parity metrics
    diff_struct_max = float(np.max(np.abs(pt_struct_np - ort_struct_np)))
    diff_struct_mean = float(np.mean(np.abs(pt_struct_np - ort_struct_np)))
    diff_rooms_max = float(np.max(np.abs(pt_rooms_np - ort_rooms_np)))
    diff_rooms_mean = float(np.mean(np.abs(pt_rooms_np - ort_rooms_np)))

    # Cosine similarity
    cos_struct = float(
        np.dot(pt_struct_np.flatten(), ort_struct_np.flatten()) /
        (np.linalg.norm(pt_struct_np) * np.linalg.norm(ort_struct_np) + 1e-8)
    )
    cos_rooms = float(
        np.dot(pt_rooms_np.flatten(), ort_rooms_np.flatten()) /
        (np.linalg.norm(pt_rooms_np) * np.linalg.norm(ort_rooms_np) + 1e-8)
    )

    passed = (diff_struct_max < atol) and (diff_rooms_max < atol)

    report = {
        "model": "Floorplan-UNetPlusPlus",
        "passed": bool(passed),
        "atol_threshold": atol,
        "structures": {
            "max_abs_diff": diff_struct_max,
            "mean_abs_diff": diff_struct_mean,
            "cosine_similarity": cos_struct,
            "output_shape": list(ort_struct_np.shape),
        },
        "rooms": {
            "max_abs_diff": diff_rooms_max,
            "mean_abs_diff": diff_rooms_mean,
            "cosine_similarity": cos_rooms,
            "output_shape": list(ort_rooms_np.shape),
        }
    }

    print(f"Structures Max Absolute Difference: {diff_struct_max:.2e} (Threshold: {atol:.0e})")
    print(f"Rooms Max Absolute Difference:      {diff_rooms_max:.2e} (Threshold: {atol:.0e})")
    print(f"Structures Cosine Similarity:       {cos_struct:.8f}")
    print(f"Rooms Cosine Similarity:            {cos_rooms:.8f}")
    print(f"Parity Status: {'PASSED' if passed else 'WARNING (Sub-Threshold)'}")

    return report


def verify_facade_parity(
    checkpoint_path: Path = FACADE_CFG.checkpoint_path,
    onnx_path: Path = FACADE_CFG.onnx_path,
    atol: float = MLOPS_CFG.parity_atol_threshold
) -> Dict[str, Any]:
    """Tests numerical parity on Pix2Pix Facade Generator"""
    print(f"\n--> Verifying PyTorch vs ONNX Parity for Facade Pix2Pix")

    pt_gen = FacadeGenerator(
        in_channels=FACADE_CFG.in_channels,
        out_channels=FACADE_CFG.out_channels,
        ngf=FACADE_CFG.ngf
    )
    load_weights(pt_gen, checkpoint_path)
    pt_gen.eval()

    opts = ort.SessionOptions()
    ort_session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])

    torch.manual_seed(42)
    test_tensor = torch.randn(1, 3, 256, 256, dtype=torch.float32)
    test_numpy = test_tensor.numpy()

    with torch.no_grad():
        pt_out = pt_gen(test_tensor).numpy()

    ort_inputs = {"semantic_mask": test_numpy}
    ort_out = ort_session.run(None, ort_inputs)[0]

    diff_max = float(np.max(np.abs(pt_out - ort_out)))
    diff_mean = float(np.mean(np.abs(pt_out - ort_out)))
    cos_sim = float(
        np.dot(pt_out.flatten(), ort_out.flatten()) /
        (np.linalg.norm(pt_out) * np.linalg.norm(ort_out) + 1e-8)
    )

    passed = diff_max < atol

    report = {
        "model": "Facade-Pix2Pix-Generator",
        "passed": bool(passed),
        "atol_threshold": atol,
        "max_abs_diff": diff_max,
        "mean_abs_diff": diff_mean,
        "cosine_similarity": cos_sim,
        "output_shape": list(ort_out.shape)
    }

    print(f"Facade Max Absolute Difference: {diff_max:.2e} (Threshold: {atol:.0e})")
    print(f"Facade Cosine Similarity:       {cos_sim:.8f}")
    print(f"Parity Status: {'PASSED' if passed else 'WARNING (Sub-Threshold)'}")

    return report


def run_full_parity_audit() -> Dict[str, Any]:
    """Runs parity audit on both models and saves report"""
    r1 = verify_floorplan_parity()
    r2 = verify_facade_parity()

    full_report = {
        "floorplan": r1,
        "facade": r2,
        "all_passed": r1["passed"] and r2["passed"]
    }

    report_path = ONNX_DIR / "parity_report.json"
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2)

    print(f"\nParity Report saved to {report_path}")
    return full_report


if __name__ == "__main__":
    run_full_parity_audit()
