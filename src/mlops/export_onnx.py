"""
ONNX Model Exporter (OS 5)
Exports PyTorch UNet++ Multi-Task and Pix2Pix Facade Generator to ONNX format.
Configures dynamic batch axes, checks model validity with onnx.checker,
and persists PyTorch checkpoints alongside ONNX models.
"""

import os
import torch
import onnx
from pathlib import Path
from typing import Dict, Any, Optional

from src.config import (
    FLOORPLAN_CFG,
    FACADE_CFG,
    CHECKPOINTS_DIR,
    ONNX_DIR,
    MLOPS_CFG
)
from src.models.unet_plus_plus import UNetPlusPlusMultiTask
from src.models.pix2pix_cgan import FacadeGenerator
from src.training.checkpoints import CheckpointError, load_weights


def _allow_untrained_default() -> bool:
    # CI has no dataset/checkpoints: it sets ARCHIGEN_CI_UNTRAINED=1 to test the plumbing only.
    return os.environ.get("ARCHIGEN_CI_UNTRAINED") == "1"


def _load_or_init(model: torch.nn.Module, checkpoint_path: Path, allow_untrained: Optional[bool]) -> None:
    """Load trained weights. An existing checkpoint is NEVER overwritten.

    Without a checkpoint, export only proceeds when allow_untrained=True (unit tests / CI),
    and then writes an explicitly-labelled untrained checkpoint.
    """
    if allow_untrained is None:
        allow_untrained = _allow_untrained_default()
    if checkpoint_path.exists():
        meta = load_weights(model, checkpoint_path, require_real_data=not allow_untrained)
        print(f"Loaded weights from {checkpoint_path} (data_source={meta.get('data_source')})")
        return
    if not allow_untrained:
        raise CheckpointError(
            f"No trained checkpoint at {checkpoint_path}. Train first; refusing to export random weights."
        )
    torch.save({"model_state_dict": model.state_dict(), "data_source": "untrained"},
               str(checkpoint_path), _use_new_zipfile_serialization=False)
    print(f"[TEST MODE] Exporting UNTRAINED weights to {checkpoint_path}")


def export_floorplan_unetpp_to_onnx(
    checkpoint_path: Path = FLOORPLAN_CFG.checkpoint_path,
    onnx_path: Path = FLOORPLAN_CFG.onnx_path,
    opset: int = MLOPS_CFG.default_onnx_opset,
    allow_untrained: Optional[bool] = None,
) -> Path:
    """
    Exports Multi-Task UNet++ to ONNX.
    Inputs: [B, 3, 512, 512] -> Outputs: structures [B, 5, 512, 512], rooms [B, 12, 512, 512]
    """
    print(f"--> Exporting Floorplan UNet++ to ONNX: {onnx_path}")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Instantiate model
    model = UNetPlusPlusMultiTask(
        in_channels=FLOORPLAN_CFG.in_channels,
        num_structure_classes=FLOORPLAN_CFG.num_structure_classes,
        num_room_classes=FLOORPLAN_CFG.num_room_classes
    )

    _load_or_init(model, checkpoint_path, allow_untrained)

    model.eval()

    # Dummy input
    dummy_input = torch.randn(1, 3, 512, 512, dtype=torch.float32)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["structures", "rooms"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "structures": {0: "batch_size"},
            "rooms": {0: "batch_size"},
        }
    )

    # Validate ONNX graph structure
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f"Floorplan ONNX export verified: {onnx_path} (Size: {onnx_path.stat().st_size / (1024*1024):.2f} MB)")
    return onnx_path


def export_facade_pix2pix_to_onnx(
    checkpoint_path: Path = FACADE_CFG.checkpoint_path,
    onnx_path: Path = FACADE_CFG.onnx_path,
    opset: int = MLOPS_CFG.default_onnx_opset,
    allow_untrained: Optional[bool] = None,
) -> Path:
    """
    Exports Pix2Pix Facade Generator to ONNX.
    Inputs: [B, 3, 256, 256] -> Output: [B, 3, 256, 256] photorealistic facade
    """
    print(f"--> Exporting Facade Pix2Pix Generator to ONNX: {onnx_path}")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    generator = FacadeGenerator(
        in_channels=FACADE_CFG.in_channels,
        out_channels=FACADE_CFG.out_channels,
        ngf=FACADE_CFG.ngf
    )

    _load_or_init(generator, checkpoint_path, allow_untrained)

    generator.eval()

    dummy_input = torch.randn(1, 3, 256, 256, dtype=torch.float32)

    torch.onnx.export(
        generator,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["semantic_mask"],
        output_names=["generated_facade"],
        dynamic_axes={
            "semantic_mask": {0: "batch_size"},
            "generated_facade": {0: "batch_size"},
        }
    )

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f"Facade Pix2Pix ONNX export verified: {onnx_path} (Size: {onnx_path.stat().st_size / (1024*1024):.2f} MB)")
    return onnx_path


def export_all_models() -> Dict[str, Path]:
    """Runs export for both models"""
    p1 = export_floorplan_unetpp_to_onnx()
    p2 = export_facade_pix2pix_to_onnx()
    return {"floorplan": p1, "facade": p2}


if __name__ == "__main__":
    export_all_models()
