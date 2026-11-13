"""
ArchiGen 3D - DVC Remote Storage & Artifact Manager
TEK-UP UNIVERSITÉ | Deep Learning Avancé & MLOps 2026/2027
Encadrant : Haythem Ghazouani
Équipe : Khalil Ayeb · Amine Ben Fadhel · Khawla Hawari

Handles programmatic interaction with DVC remote storage (MinIO, S3, Local).
Provides verification of tracked artifacts (weights, ONNX models, processed datasets).
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import configparser

from src.config import BASE_DIR, MODELS_DIR, DATA_DIR, CHECKPOINTS_DIR, ONNX_DIR


def get_dvc_config_path() -> Path:
    return BASE_DIR / ".dvc" / "config"


def get_dvc_remotes() -> Dict[str, Dict[str, str]]:
    """Parses .dvc/config to discover configured remotes."""
    cfg_file = get_dvc_config_path()
    if not cfg_file.exists():
        return {}

    config = configparser.ConfigParser()
    config.read(cfg_file)

    remotes = {}
    default_remote = config.get("core", "remote", fallback=None)

    for section in config.sections():
        if section.startswith('remote "') or section.startswith("'remote \""):
            # Clean section name
            clean_name = section.replace("remote", "").replace('"', '').replace("'", "").strip()
            remotes[clean_name] = {
                "url": config.get(section, "url", fallback=""),
                "endpointurl": config.get(section, "endpointurl", fallback=""),
                "is_default": (clean_name == default_remote)
            }
    return remotes


def run_dvc_command(args: List[str], timeout_s: int = 120) -> Dict[str, Any]:
    """Executes a DVC CLI command via python -m dvc and returns output and exit code."""
    cmd = [sys.executable, "-m", "dvc"] + args
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s
        )
        return {
            "success": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip()
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"Command timed out after {timeout_s}s"
        }
    except Exception as e:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": str(e)
        }


def get_tracked_artifacts_status() -> Dict[str, Any]:
    """Inspects tracked data and model files and their sizes/hashes."""
    tracked = {
        "models": {
            "floorplan_pytorch": {
                "path": "models/checkpoints/floorplan_unetpp_best.pt",
                "exists": (CHECKPOINTS_DIR / "floorplan_unetpp_best.pt").exists(),
                "size_mb": round((CHECKPOINTS_DIR / "floorplan_unetpp_best.pt").stat().st_size / (1024 * 1024), 2)
                if (CHECKPOINTS_DIR / "floorplan_unetpp_best.pt").exists() else 0.0
            },
            "facade_pytorch": {
                "path": "models/checkpoints/facade_pix2pix_best.pt",
                "exists": (CHECKPOINTS_DIR / "facade_pix2pix_best.pt").exists(),
                "size_mb": round((CHECKPOINTS_DIR / "facade_pix2pix_best.pt").stat().st_size / (1024 * 1024), 2)
                if (CHECKPOINTS_DIR / "facade_pix2pix_best.pt").exists() else 0.0
            },
            "floorplan_onnx": {
                "path": "models/onnx/floorplan_unetpp.onnx",
                "exists": (ONNX_DIR / "floorplan_unetpp.onnx").exists(),
                "size_mb": round((ONNX_DIR / "floorplan_unetpp.onnx").stat().st_size / (1024 * 1024), 2)
                if (ONNX_DIR / "floorplan_unetpp.onnx").exists() else 0.0
            },
            "facade_onnx": {
                "path": "models/onnx/facade_pix2pix.onnx",
                "exists": (ONNX_DIR / "facade_pix2pix.onnx").exists(),
                "size_mb": round((ONNX_DIR / "facade_pix2pix.onnx").stat().st_size / (1024 * 1024), 2)
                if (ONNX_DIR / "facade_pix2pix.onnx").exists() else 0.0
            }
        },
        "data": {
            "cubicasa_cache": {
                "path": "data/processed/cubicasa_cache",
                "exists": (DATA_DIR / "processed" / "cubicasa_cache").exists(),
                "num_files": len(list((DATA_DIR / "processed" / "cubicasa_cache").glob("*.npz")))
                if (DATA_DIR / "processed" / "cubicasa_cache").exists() else 0
            },
            "cmp_cache": {
                "path": "data/processed/cmp_cache",
                "exists": (DATA_DIR / "processed" / "cmp_cache").exists(),
                "num_files": len(list((DATA_DIR / "processed" / "cmp_cache").glob("*.npz")))
                if (DATA_DIR / "processed" / "cmp_cache").exists() else 0
            }
        }
    }
    return tracked


def configure_remote(name: str, url: str, endpointurl: Optional[str] = None, default: bool = True) -> bool:
    """Configures a DVC remote."""
    cmd = ["remote", "add", "-f"]
    if default:
        cmd.append("-d")
    cmd.extend([name, url])

    res = run_dvc_command(cmd)
    if not res["success"]:
        return False

    if endpointurl:
        mod_res = run_dvc_command(["remote", "modify", name, "endpointurl", endpointurl])
        if not mod_res["success"]:
            return False

    return True


def get_full_dvc_status() -> Dict[str, Any]:
    """Returns complete DVC status including remotes, pipeline status, and artifact audit."""
    remotes = get_dvc_remotes()
    artifacts = get_tracked_artifacts_status()
    dvc_status = run_dvc_command(["status"], timeout_s=30)

    return {
        "dvc_initialized": (BASE_DIR / ".dvc").exists(),
        "remotes": remotes,
        "default_remote": next((k for k, v in remotes.items() if v.get("is_default")), None),
        "artifacts": artifacts,
        "dvc_status_output": dvc_status.get("stdout") or dvc_status.get("stderr"),
    }


if __name__ == "__main__":
    import json
    status = get_full_dvc_status()
    print("==================================================")
    print(" 🏛️ ArchiGen 3D - DVC Remote Management Summary")
    print("==================================================")
    print(json.dumps(status, indent=2))
