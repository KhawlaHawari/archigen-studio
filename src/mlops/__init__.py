"""
MLOps utilities module.

Exports are loaded lazily, so lightweight tools (retraining policy, Evidently report, feedback
loop) can run without importing PyTorch / ONNX Runtime (e.g. in a small CI job or a Python 3.11
environment that only has Evidently).
"""
from importlib import import_module

_EXPORTS = {
    "export_floorplan_unetpp_to_onnx": "src.mlops.export_onnx",
    "export_facade_pix2pix_to_onnx": "src.mlops.export_onnx",
    "export_all_models": "src.mlops.export_onnx",
    "verify_floorplan_parity": "src.mlops.onnx_verifier",
    "verify_facade_parity": "src.mlops.onnx_verifier",
    "run_full_parity_audit": "src.mlops.onnx_verifier",
    "run_latency_benchmark": "src.mlops.benchmark_latency",
    "log_experiment_to_mlflow": "src.mlops.mlflow_tracker",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module 'src.mlops' has no attribute {name!r}")
