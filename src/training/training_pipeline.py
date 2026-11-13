"""
ArchiGen 3D - End-to-End Training Pipeline Orchestrator
Runs training for both models (floorplan + facade), exports to ONNX,
verifies parity, and registers to MLflow Model Registry.
"""

from typing import Dict, Any, Optional
from pathlib import Path

from src.training.train_floorplan import train_floorplan, TrainResult as FloorplanResult
from src.training.train_facade import train_facade, TrainResult as FacadeResult
from src.mlops.model_registry import register_and_promote_models


def run_training_pipeline(
    floorplan_epochs: int = 25,
    floorplan_batch_size: int = 2,
    floorplan_lr: float = 1e-4,
    facade_epochs: int = 50,
    facade_batch_size: int = 2,
    facade_lr_g: float = 2e-4,
    facade_lr_d: float = 2e-4,
    seed: int = 42,
    auto_promote: bool = True,
) -> Dict[str, Any]:
    """Execute the full model training pipeline.

    Returns a summary dict with both training results and registry info.
    """
    print("=" * 60)
    print("ARCHIGEN 3D - TRAINING PIPELINE")
    print("=" * 60)

    # 1. Train Floorplan UNet++
    print("\n[1/4] Training Floorplan UNet++ Multi-Task ...")
    fp_result: FloorplanResult = train_floorplan(
        epochs=floorplan_epochs,
        batch_size=floorplan_batch_size,
        lr=floorplan_lr,
        seed=seed,
    )

    # 2. Train Facade Pix2Pix cGAN
    print("\n[2/4] Training Facade Pix2Pix cGAN ...")
    fc_result: FacadeResult = train_facade(
        epochs=facade_epochs,
        batch_size=facade_batch_size,
        lr_g=facade_lr_g,
        lr_d=facade_lr_d,
        seed=seed,
    )

    # 3. Export to ONNX + parity verification
    print("\n[3/4] Exporting models to ONNX & verifying parity ...")
    from src.mlops.export_onnx import export_all_models
    from src.mlops.onnx_verifier import run_full_parity_audit

    onnx_paths = export_all_models()
    parity_report = run_full_parity_audit()

    # 4. MLflow Model Registry promotion
    print("\n[4/4] Registering & promoting models to MLflow Registry ...")
    registry_info = None
    if auto_promote and parity_report.get("all_passed", False):
        registry_info = register_and_promote_models()
    elif auto_promote:
        print("[WARN] Skipping registry promotion: ONNX parity check failed.")

    summary = {
        "floorplan": {
            "run_id": fp_result.run_id,
            "experiment_name": fp_result.experiment_name,
            "best_metric": fp_result.best_metric,
            "best_metric_name": fp_result.best_metric_name,
            "checkpoint_path": str(fp_result.checkpoint_path),
            "epochs_completed": fp_result.epochs_completed,
            "mlflow_uri": fp_result.mlflow_uri,
        },
        "facade": {
            "run_id": fc_result.run_id,
            "experiment_name": fc_result.experiment_name,
            "best_metric": fc_result.best_metric,
            "best_metric_name": fc_result.best_metric_name,
            "checkpoint_path": str(fc_result.checkpoint_path),
            "epochs_completed": fc_result.epochs_completed,
            "mlflow_uri": fc_result.mlflow_uri,
        },
        "onnx": {k: str(v) for k, v in onnx_paths.items()},
        "parity": parity_report,
        "model_registry": registry_info,
        "all_passed": parity_report.get("all_passed", False),
    }

    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Floorplan best {fp_result.best_metric_name}: {fp_result.best_metric:.4f}")
    print(f"Facade best {fc_result.best_metric_name}: {fc_result.best_metric:.4f}")
    print(f"ONNX parity all_passed: {parity_report.get('all_passed')}")
    print(f"Model registry promoted: {registry_info is not None}")
    return summary


if __name__ == "__main__":
    run_training_pipeline()
