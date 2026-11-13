"""
ArchiGen 3D - single place to load trained weights.

Accepts both checkpoint formats used in the project:
  * trainer format: {"model_state_dict": ..., "data_source": "real:...", "metrics": ..., ...}
  * bare state_dict (older exports)

It NEVER silently falls back to random weights: a missing or incompatible
checkpoint raises, so no score or export can be produced from an untrained model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


class CheckpointError(RuntimeError):
    pass


def read_checkpoint(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise CheckpointError(f"Checkpoint not found: {path}. Train the model first.")
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt
    if isinstance(ckpt, dict):
        return {"model_state_dict": ckpt, "data_source": "unknown (bare state_dict)"}
    raise CheckpointError(f"Unrecognised checkpoint format in {path}")


def load_weights(model: torch.nn.Module, path: Path, require_real_data: bool = False) -> Dict[str, Any]:
    """Load weights into `model` (strict). Returns the checkpoint metadata (without tensors).

    require_real_data=True refuses checkpoints that were not produced by a training run
    on real data (e.g. the old synthetic-data checkpoints).
    """
    ckpt = read_checkpoint(path)
    source = str(ckpt.get("data_source", "unknown"))
    if require_real_data and not source.startswith("real:"):
        raise CheckpointError(
            f"{path} was not trained on real data (data_source={source!r}). "
            "Retrain with src.training.train_facade / train_floorplan."
        )
    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except RuntimeError as exc:
        raise CheckpointError(f"Checkpoint {path} does not match the model architecture: {exc}") from exc
    return {k: v for k, v in ckpt.items()
            if k not in ("model_state_dict", "optimizers", "schedulers", "discriminator_state_dict")}
