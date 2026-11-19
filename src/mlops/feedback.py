"""
Feedback loop ("boucle de feedback"): user corrections from the studio become new training data.

1. In the studio the user fixes a plan (moves walls, adds doors, relabels rooms) and clicks
   "Send correction". POST /feedback/layout stores, under data/feedback/<id>/:
       plan.<ext>     the original uploaded image (bytes, unchanged)
       layout.json    the corrected layout (meters, same schema as /predict/floorplan/layout)
       meta.json      time, model variant that produced the original prediction, scale, sha256
2. `python -m src.data.feedback_dataset` rasterises every correction into the CubiCasa training
   format (512 px image + structure mask + room mask) in data/processed/feedback_cache/train/.
3. Training with `--include-feedback` adds those examples to the training split; the retraining
   policy (src/mlops/retraining_policy.py) counts new corrections as a retraining trigger.

data/feedback/ holds user data: it is git-ignored and versioned with DVC.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

FEEDBACK_DIR = Path(__file__).resolve().parents[2] / "data" / "feedback"
MAX_IMAGE_BYTES = 20 * 1024 * 1024


def save_feedback(image_bytes: bytes, filename: str, layout: Dict[str, Any], model_variant: Optional[str],
                  scale_m_per_px: Optional[float]) -> Dict[str, Any]:
    if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("The plan image is missing or larger than 20 MB.")
    if not isinstance(layout, dict) or not layout.get("walls") or not isinstance(layout.get("rooms"), list):
        raise ValueError("The corrected layout must contain walls and rooms.")
    digest = hashlib.sha256(image_bytes).hexdigest()
    fid = time.strftime("%Y%m%d-%H%M%S") + "_" + digest[:10]
    folder = FEEDBACK_DIR / fid
    folder.mkdir(parents=True, exist_ok=False)
    ext = Path(filename or "plan.png").suffix.lower()
    ext = ext if ext in (".png", ".jpg", ".jpeg", ".webp") else ".png"
    (folder / f"plan{ext}").write_bytes(image_bytes)
    (folder / "layout.json").write_text(json.dumps(layout), encoding="utf-8")
    meta = {"id": fid, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "image_sha256": digest,
            "image_file": f"plan{ext}", "model_variant": model_variant,
            "scale_meters_per_pixel": scale_m_per_px or layout.get("metadata", {}).get("scale_meters_per_pixel"),
            "num_walls": len(layout.get("walls", [])), "num_rooms": len(layout.get("rooms", [])),
            "num_doors": len(layout.get("doors", [])), "num_windows": len(layout.get("windows", []))}
    (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def feedback_stats(since: Optional[float] = None) -> Dict[str, Any]:
    """Number of stored corrections (in total and, if `since` is a UNIX time, created after it)."""
    items = sorted(p for p in FEEDBACK_DIR.glob("*/meta.json")) if FEEDBACK_DIR.exists() else []
    newer = [p for p in items if since is None or p.stat().st_mtime > since]
    return {"total": len(items), "new_since_last_training": len(newer), "folder": str(FEEDBACK_DIR)}
