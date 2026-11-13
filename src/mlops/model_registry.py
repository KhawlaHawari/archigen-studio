"""
MLflow Model Registry Promotion (Staging → Production)
Registers trained floorplan & facade models, transitions them through
Staging → Production, and archives previous Production versions.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import (
    BASE_DIR,
    FLOORPLAN_CFG,
    FACADE_CFG,
    MLOPS_CFG,
    ONNX_DIR,
)

# Registered model names in the MLflow Model Registry
MODEL_NAME_FLOORPLAN = "ArchiGen3D-Floorplan-UNetPlusPlus"
MODEL_NAME_FACADE = "ArchiGen3D-Facade-Pix2Pix"


def _get_mlflow():
    try:
        import mlflow  # type: ignore
        from mlflow.tracking import MlflowClient  # type: ignore
        return mlflow, MlflowClient
    except Exception as exc:
        print(f"[WARN] MLflow not available ({exc}). Registry promotion skipped.")
        return None, None


def _transition_to_production(
    client: Any, model_name: str, version: int, archive_existing: bool = True
) -> None:
    """Transition a model version to Production, archiving prior Production versions."""
    # Archive existing Production versions
    if archive_existing:
        for mv in client.search_model_versions(f"name='{model_name}'"):
            if mv.current_stage == "Production":
                client.transition_model_version_stage(
                    name=model_name, version=mv.version, stage="Archived"
                )
                print(f"[Registry] Archived previous Production version {mv.version} of {model_name}")

    # Promote candidate to Production
    client.transition_model_version_stage(
        name=model_name, version=version, stage="Production"
    )
    print(f"[Registry] Promoted {model_name} v{version} → Production")


def register_and_promote_models(
    parity_report_path: Path = ONNX_DIR / "parity_report.json",
) -> Optional[Dict[str, Any]]:
    """Register both ONNX models to MLflow Model Registry and promote to Production.

    Only promotes if ONNX parity check passed (Δ < 1e-5).
    """
    mlflow, MlflowClient = _get_mlflow()
    if mlflow is None or MlflowClient is None:
        return None

    try:
        import mlflow.onnx  # type: ignore
        import onnx  # type: ignore
    except Exception as exc:
        print(f"[WARN] mlflow.onnx flavor not available ({exc}). Registry promotion skipped.")
        return None

    # Verify parity before promotion
    try:
        with open(parity_report_path, "r") as f:
            parity = json.load(f)
    except FileNotFoundError:
        print(f"[WARN] Parity report not found at {parity_report_path}. Skipping promotion.")
        return None

    if not parity.get("all_passed", False):
        print("[WARN] ONNX parity check failed. Skipping Model Registry promotion.")
        return None

    mlflow.set_tracking_uri(MLOPS_CFG.tracking_uri)
    client = MlflowClient(tracking_uri=MLOPS_CFG.tracking_uri)

    results: Dict[str, Any] = {"floorplan": None, "facade": None}

    # ----- Floorplan model -----
    fp_onnx = FLOORPLAN_CFG.onnx_path
    if fp_onnx.exists():
        with mlflow.start_run(run_name=f"{MODEL_NAME_FLOORPLAN}-register"):
            mlflow.log_param("architecture", "UNetPlusPlus-ResNet34")
            mlflow.log_param("task", "multi_task_segmentation")
            onnx_model = onnx.load(str(fp_onnx))
            model_info = mlflow.onnx.log_model(
                onnx_model=onnx_model,
                artifact_path="onnx",
                registered_model_name=MODEL_NAME_FLOORPLAN,
            )
            mv_version = model_info.registered_model_version
            _transition_to_production(client, MODEL_NAME_FLOORPLAN, mv_version)
            results["floorplan"] = {
                "model_name": MODEL_NAME_FLOORPLAN,
                "version": mv_version,
                "stage": "Production",
                "model_uri": f"models:/{MODEL_NAME_FLOORPLAN}/Production",
            }

    # ----- Facade model -----
    fc_onnx = FACADE_CFG.onnx_path
    if fc_onnx.exists():
        with mlflow.start_run(run_name=f"{MODEL_NAME_FACADE}-register"):
            mlflow.log_param("architecture", "Pix2Pix-cGAN-Perceptual-SN")
            mlflow.log_param("task", "image_to_image_translation")
            onnx_model = onnx.load(str(fc_onnx))
            model_info = mlflow.onnx.log_model(
                onnx_model=onnx_model,
                artifact_path="onnx",
                registered_model_name=MODEL_NAME_FACADE,
            )
            mv_version = model_info.registered_model_version
            _transition_to_production(client, MODEL_NAME_FACADE, mv_version)
            results["facade"] = {
                "model_name": MODEL_NAME_FACADE,
                "version": mv_version,
                "stage": "Production",
                "model_uri": f"models:/{MODEL_NAME_FACADE}/Production",
            }

    # Persist registry summary locally
    summary_path = BASE_DIR / "models" / "mlflow_registry_summary.json"
    payload = {
        "registry_promoted": True,
        "parity_passed": True,
        "models": results,
    }
    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[Registry] Summary persisted: {summary_path}")

    return payload


def get_production_model_uri(model_name: str) -> Optional[str]:
    """Resolve the Production model URI from the MLflow Model Registry."""
    _, MlflowClient = _get_mlflow()
    if MlflowClient is None:
        return None
    try:
        mlflow.set_tracking_uri(MLOPS_CFG.tracking_uri)
        client = MlflowClient(tracking_uri=MLOPS_CFG.tracking_uri)
        latest = client.get_latest_versions(model_name, stages=["Production"])
        if latest:
            return latest[0].source
    except Exception as exc:
        print(f"[WARN] Could not resolve Production URI for {model_name}: {exc}")
    return None


if __name__ == "__main__":
    register_and_promote_models()
