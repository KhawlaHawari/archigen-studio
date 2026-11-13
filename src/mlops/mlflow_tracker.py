"""
MLflow experiment summary - MEASURED metrics only.

Metrics are read from files written by real runs:
  * models/metrics/<model>_val.json   (training: best validation epoch)
  * models/metrics/<model>_test.json  (src.evaluation.evaluate on the test split)

Nothing is hardcoded. A model without a metrics file is reported as
status="not_measured" instead of with a number.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import BASE_DIR, FACADE_CFG, FLOORPLAN_CFG, MLOPS_CFG

METRICS_DIR = BASE_DIR / "models" / "metrics"

# Model name -> description; checkpoint stems produced by the training scripts.
TRACKED_MODELS = {
    "facade_unet_l1": "U-Net, L1 loss only (reference baseline) - CMP Facades",
    "facade_pix2pix": "Pix2Pix cGAN (U-Net + PatchGAN) - CMP Facades",
    "floorplan_unetpp": "UNet++ multi-task (structures + rooms) - CubiCasa5K",
}


def _read(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def get_measured_metrics() -> Dict[str, Any]:
    """Collect measured val/test metrics per model. Never returns invented numbers."""
    out: Dict[str, Any] = {}
    for name, desc in TRACKED_MODELS.items():
        val = _read(METRICS_DIR / f"{name}_val.json")
        test = _read(METRICS_DIR / f"{name}_best_test.json")
        entry: Dict[str, Any] = {"description": desc}
        if val is None and test is None:
            entry["status"] = "not_measured"
        else:
            entry["status"] = "measured"
            if val is not None:
                entry["validation"] = val.get("best", {})
                entry["hparams"] = val.get("hparams", {})
                entry["epochs_completed"] = val.get("epochs_completed")
                if val.get("max_train_batches"):
                    entry["warning"] = "partial epochs (smoke run) - not a reportable result"
            if test is not None:
                entry["test"] = {k: test.get(k) for k in
                                 ("n_images", "l1", "psnr_db", "ssim", "lpips", "merged_wall_iou",
                                  "structure", "rooms", "env") if k in test}
        out[name] = entry

    # Team reference results from an earlier campaign (checkpoints not in this repo).
    ref = _read(BASE_DIR / "models" / "reference" / "team_reference_results.json")
    if ref is not None:
        fc = ref.get("facade_cmp_test_106", {})
        best = fc.get("Perceptual (EMA + Schedule)", {})
        fp = ref.get("floorplan_cubicasa_test_400", {})
        fpm = {k: v.get("baseline") for k, v in fp.get("metrics", {}).items() if isinstance(v, dict)}
        out["team_reference"] = {
            "description": "Team reference (earlier campaign, weights not in this repo)",
            "status": "reference",
            "test": {
                "facade_l1": best.get("l1_loss"), "facade_ssim": best.get("ssim"),
                "facade_psnr_db": best.get("psnr_db"), "facade_lpips": best.get("lpips"),
                "facade_kid_x1000": best.get("kid_x1000"),
                "floorplan_merged_wall_iou_pct": fpm.get("Merged Wall IoU"),
                "floorplan_door_iou_pct": fpm.get("Door IoU"),
                "floorplan_window_iou_pct": fpm.get("Window IoU"),
                "floorplan_room_pixel_acc_pct": fpm.get("Room Pixel Acc"),
                "floorplan_room_miou_10c_pct": fpm.get("Room mIoU (10c, inc. Office)"),
            },
            "checkpoints": {"facade": best.get("checkpoint"), "floorplan": fp.get("checkpoint")},
        }
    return out


def log_experiment_to_mlflow(
    metrics_override: Optional[Dict[str, Any]] = None,
    run_name: str = "ArchiGen3D-Measured-Summary",
) -> Dict[str, Any]:
    """Log the measured summary to MLflow (if available) and to models/mlflow_run_summary.json."""
    tracking_uri = MLOPS_CFG.tracking_uri
    experiment_name = MLOPS_CFG.experiment_name
    metrics_data = metrics_override or get_measured_metrics()
    summary_path = BASE_DIR / "models" / "mlflow_run_summary.json"

    mlflow_active = False
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name):
            for model, entry in metrics_data.items():
                mlflow.log_param(f"{model}_status", entry.get("status"))
                for split in ("validation", "test"):
                    for k, v in (entry.get(split) or {}).items():
                        if isinstance(v, dict) and "mean" in v:
                            v = v["mean"]
                        if isinstance(v, (int, float)):
                            mlflow.log_metric(f"{model}_{split}_{k}", float(v))
            for p in (FLOORPLAN_CFG.onnx_path, FACADE_CFG.onnx_path):
                if p.exists():
                    mlflow.log_artifact(str(p), artifact_path="onnx")
        mlflow_active = True
    except Exception as e:
        print(f"MLflow tracking offline: {e}")

    payload = {
        "experiment_name": experiment_name,
        "run_name": run_name,
        "mlflow_active": mlflow_active,
        "tracking_uri": tracking_uri,
        "source": "measured metric files in models/metrics/ (no hardcoded values)",
        "metrics_and_parameters": metrics_data,
    }
    summary_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Local tracking summary persisted: {summary_path}")
    return payload


if __name__ == "__main__":
    log_experiment_to_mlflow()
