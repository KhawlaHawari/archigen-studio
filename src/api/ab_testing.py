"""
A/B testing / canary release of floor-plan models (MLOps maturity level 2).

* champion   : the production model (models/checkpoints/floorplan_unetpp_best.pt, served from ONNX);
* challenger : another checkpoint (models/checkpoints/floorplan_unetpp_<name>.pt) that receives a
               small share of the traffic (canary fraction, default 10 %).

Assignment is sticky: the plan image's SHA-256 decides the bucket, so a given plan always gets the
same model (no flicker in the studio, clean comparison). A client can force a variant with the
`X-Model-Variant` header (used by tests and demos).

Every request is recorded per variant in Prometheus (src/api/observability.py). GET /ab/status
compares the variants on the business metric "usable plan rate" with a two-proportion z-test and
says whether the challenger should be promoted, kept in canary, or rolled back.

Configuration (environment variables):
    ARCHIGEN_AB_CHAMPION=run3   ARCHIGEN_AB_CHALLENGER=run2   ARCHIGEN_AB_CANARY_FRACTION=0.10
    ARCHIGEN_AB_CHALLENGER=none  -> A/B testing off (100 % champion)
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional

from prometheus_client import REGISTRY

CHECKPOINTS = Path(__file__).resolve().parents[2] / "models" / "checkpoints"
MIN_SAMPLES_PER_VARIANT = 30


class ABRouter:
    def __init__(self):
        self.champion = os.environ.get("ARCHIGEN_AB_CHAMPION", "run3")
        challenger = os.environ.get("ARCHIGEN_AB_CHALLENGER", "run2")
        self.challenger = None if challenger.lower() in ("", "none", "off") else challenger
        self.fraction = float(os.environ.get("ARCHIGEN_AB_CANARY_FRACTION", "0.10"))
        if self.challenger and not self.checkpoint(self.challenger).exists():
            self.challenger = None   # nothing to compare against

    @staticmethod
    def checkpoint(variant: str) -> Path:
        return CHECKPOINTS / f"floorplan_unetpp_{variant}.pt"

    def variants(self):
        return [self.champion] + ([self.challenger] if self.challenger else [])

    def assign(self, image_bytes: bytes, requested: Optional[str] = None) -> str:
        if requested and requested in self.variants():
            return requested
        if not self.challenger or self.fraction <= 0:
            return self.champion
        bucket = int(hashlib.sha256(image_bytes).hexdigest()[:8], 16) / 0xFFFFFFFF
        return self.challenger if bucket < self.fraction else self.champion

    def status(self) -> Dict[str, Any]:
        def value(name, variant):
            return REGISTRY.get_sample_value(name, {"variant": variant}) or 0.0

        stats = {}
        for v in self.variants():
            n = value("archigen_plans_analysed_total", v)
            usable = value("archigen_plans_usable_total", v)
            shell = value("archigen_plans_shell_closed_total", v)
            lat_sum = REGISTRY.get_sample_value("archigen_model_inference_seconds_sum", {"task": "floorplan", "variant": v}) or 0.0
            stats[v] = {"role": "champion" if v == self.champion else "challenger", "requests": int(n),
                        "usable_rate": round(usable / n, 4) if n else None,
                        "closed_shell_rate": round(shell / n, 4) if n else None,
                        "mean_latency_s": round(lat_sum / n, 3) if n else None}
        result = {"champion": self.champion, "challenger": self.challenger, "canary_fraction": self.fraction,
                  "variants": stats}
        if self.challenger:
            result["decision"] = self._decide(stats[self.champion], stats[self.challenger])
        return result

    @staticmethod
    def _decide(champ: Dict[str, Any], chall: Dict[str, Any]) -> Dict[str, Any]:
        n1, n2 = champ["requests"], chall["requests"]
        if min(n1, n2) < MIN_SAMPLES_PER_VARIANT:
            return {"action": "keep_canary", "reason": f"need >= {MIN_SAMPLES_PER_VARIANT} plans per variant (have {n1} / {n2})"}
        p1, p2 = champ["usable_rate"], chall["usable_rate"]
        p = (p1 * n1 + p2 * n2) / (n1 + n2)
        se = math.sqrt(max(p * (1 - p) * (1 / n1 + 1 / n2), 1e-12))
        z = (p2 - p1) / se
        p_value = math.erfc(abs(z) / math.sqrt(2))     # two-sided
        if p_value < 0.05 and z > 0:
            action = "promote_challenger"
        elif p_value < 0.05 and z < 0:
            action = "rollback_challenger"
        else:
            action = "keep_canary"
        return {"action": action, "metric": "usable_rate", "difference": round(p2 - p1, 4),
                "z": round(z, 3), "p_value": round(p_value, 4), "test": "two-proportion z-test (two-sided)"}


ROUTER = ABRouter()
