"""
Unit tests for ONNX Export & Strict Numerical Parity
"""

import pytest
from pathlib import Path
from src.mlops.export_onnx import export_floorplan_unetpp_to_onnx, export_facade_pix2pix_to_onnx
from src.mlops.onnx_verifier import verify_floorplan_parity, verify_facade_parity
from src.config import FLOORPLAN_CFG, FACADE_CFG


def test_floorplan_onnx_export_and_parity(tmp_path):
    """Exports floorplan UNet++ and tests max absolute error < 1e-4"""
    onnx_file = tmp_path / "test_floorplan.onnx"
    ckpt_file = tmp_path / "test_floorplan.pt"

    export_floorplan_unetpp_to_onnx(checkpoint_path=ckpt_file, onnx_path=onnx_file, allow_untrained=True)
    assert onnx_file.exists()
    assert onnx_file.stat().st_size > 1000

    report = verify_floorplan_parity(checkpoint_path=ckpt_file, onnx_path=onnx_file, atol=1e-4)
    assert report["structures"]["max_abs_diff"] < 1e-4
    assert report["rooms"]["max_abs_diff"] < 1e-4
    assert report["structures"]["cosine_similarity"] > 0.9999
    assert report["rooms"]["cosine_similarity"] > 0.9999


def test_facade_onnx_export_and_parity(tmp_path):
    """Exports facade Pix2Pix and tests max absolute error < 1e-4"""
    onnx_file = tmp_path / "test_facade.onnx"
    ckpt_file = tmp_path / "test_facade.pt"

    export_facade_pix2pix_to_onnx(checkpoint_path=ckpt_file, onnx_path=onnx_file, allow_untrained=True)
    assert onnx_file.exists()
    assert onnx_file.stat().st_size > 1000

    report = verify_facade_parity(checkpoint_path=ckpt_file, onnx_path=onnx_file, atol=1e-4)
    assert report["max_abs_diff"] < 1e-4
    assert report["cosine_similarity"] > 0.9999
