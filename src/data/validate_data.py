"""
Data validation gate ("Qualité et validation des données"): explicit expectations that FAIL the
pipeline (exit code 1) when the training data is wrong. src.data.audit_data only reports statistics;
this module decides pass / fail and runs before training in DVC (stage validate_data) and in CI.

Expectations
  CubiCasa cache (data/processed/cubicasa_cache, + feedback_cache for the feedback loop)
    E1 split sizes = official 4200 / 400 / 400
    E2 every file has image (512,512,3) uint8, structure_mask and room_mask (512,512) uint8
    E3 structure labels within {0..4}; room labels within {0..11, 255}
    E4 no plan id appears in two splits (no leakage)
    E5 no image is blank (std > 3) and every plan has walls (>= 0.5 % wall pixels)
    E6 class coverage: every structure class and every room class appears in train
  CMP Facades (data/raw/facades)
    F1 split sizes = 400 / 100 / 106
    F2 every pair decodes to 512x256 RGB (photo | label map)
    F3 no identical label map across splits (no leakage)

    python -m src.data.validate_data                 # full check
    python -m src.data.validate_data --sample 300    # quick check (CI), random files per split
Output: data/processed/data_validation.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from src.config import DATA_DIR

CACHE = DATA_DIR / "processed" / "cubicasa_cache"
FEEDBACK_CACHE = DATA_DIR / "processed" / "feedback_cache"
FACADES = DATA_DIR / "raw" / "facades"
OUT = DATA_DIR / "processed" / "data_validation.json"
CUBI_SPLITS = {"train": 4200, "val": 400, "test": 400}
FACADE_SPLITS = {"train": 400, "val": 100, "test": 106}
STRUCT_OK = set(range(5))
ROOM_OK = set(range(12)) | {255}


class Report:
    def __init__(self):
        self.checks: List[Dict[str, Any]] = []

    def expect(self, cid: str, ok: bool, description: str, detail: Any = None):
        self.checks.append({"id": cid, "passed": bool(ok), "expectation": description, "detail": detail})

    @property
    def passed(self) -> bool:
        return all(c["passed"] for c in self.checks)


def _plan_id(path: Path) -> str:
    return path.stem


def validate_npz_files(files: List[Path], rep: Report, prefix: str, check_coverage: bool = False):
    bad_schema, bad_labels, blank, no_walls = [], [], [], []
    struct_seen, room_seen = set(), set()
    for f in files:
        try:
            with np.load(f) as d:
                img, s, r = d["image"], d["structure_mask"], d["room_mask"]
        except Exception as exc:
            bad_schema.append(f"{f.name}: {type(exc).__name__}")
            continue
        if img.shape != (512, 512, 3) or img.dtype != np.uint8 or s.shape != (512, 512) or r.shape != (512, 512):
            bad_schema.append(f"{f.name}: {img.shape} {img.dtype} {s.shape} {r.shape}")
            continue
        us, ur = set(np.unique(s).tolist()), set(np.unique(r).tolist())
        if not us <= STRUCT_OK or not ur <= ROOM_OK:
            bad_labels.append(f"{f.name}: structure {sorted(us - STRUCT_OK)} room {sorted(ur - ROOM_OK)}")
        struct_seen |= us
        room_seen |= ur
        if float(img.std()) <= 3.0:
            blank.append(f.name)
        if np.isin(s, (1, 2)).mean() < 0.005:
            no_walls.append(f.name)
    rep.expect(f"{prefix}-E2", not bad_schema, "arrays have the expected shapes and dtypes", bad_schema[:10])
    rep.expect(f"{prefix}-E3", not bad_labels, "label values are valid", bad_labels[:10])
    rep.expect(f"{prefix}-E5", not blank and not no_walls, "no blank image, every plan has walls",
               {"blank": blank[:10], "no_walls": no_walls[:10]})
    if check_coverage:
        missing = {"structure": sorted(STRUCT_OK - struct_seen), "room": sorted(set(range(12)) - room_seen)}
        rep.expect(f"{prefix}-E6", not missing["structure"] and not missing["room"],
                   "every structure and room class appears in the training split", missing)


def validate_cubicasa(rep: Report, sample: int | None, rng: random.Random):
    split_files = {s: sorted((CACHE / s).glob("*.npz")) for s in CUBI_SPLITS}
    sizes = {s: len(v) for s, v in split_files.items()}
    rep.expect("cubicasa-E1", sizes == CUBI_SPLITS, "official split sizes 4200 / 400 / 400", sizes)
    ids = {s: {_plan_id(f) for f in v} for s, v in split_files.items()}
    overlap = {f"{a}&{b}": sorted(ids[a] & ids[b])[:5] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
               if ids[a] & ids[b]}
    rep.expect("cubicasa-E4", not overlap, "no plan appears in two splits", overlap)
    for split, files in split_files.items():
        chosen = files if not sample or len(files) <= sample else rng.sample(files, sample)
        validate_npz_files(chosen, rep, f"cubicasa-{split}", check_coverage=(split == "train"))
    fb = sorted((FEEDBACK_CACHE / "train").glob("*.npz"))
    if fb:
        validate_npz_files(fb, rep, "feedback")


def validate_facades(rep: Report):
    files = {s: sorted((FACADES / s).glob("*.jpg")) for s in FACADE_SPLITS}
    sizes = {s: len(v) for s, v in files.items()}
    rep.expect("facades-F1", sizes == FACADE_SPLITS, "CMP split sizes 400 / 100 / 106", sizes)
    bad, label_hash = [], {}
    for split, paths in files.items():
        for p in paths:
            try:
                im = Image.open(p).convert("RGB")
            except Exception as exc:
                bad.append(f"{split}/{p.name}: {type(exc).__name__}")
                continue
            if im.size != (512, 256):
                bad.append(f"{split}/{p.name}: {im.size}")
                continue
            label = np.asarray(im)[:, 256:]
            label_hash.setdefault(hashlib.sha256(label.tobytes()).hexdigest(), set()).add(split)
    rep.expect("facades-F2", not bad, "every pair decodes to 512x256 RGB", bad[:10])
    leaks = sum(1 for splits in label_hash.values() if len(splits) > 1)
    rep.expect("facades-F3", leaks == 0, "no identical label map in two splits", {"shared_label_maps": leaks})


def run(sample: int | None = None, seed: int = 42) -> Dict[str, Any]:
    rep, rng, t0 = Report(), random.Random(seed), time.time()
    if CACHE.exists():
        validate_cubicasa(rep, sample, rng)
    else:
        rep.expect("cubicasa-present", False, "CubiCasa cache exists (python -m src.data.rasterize_cubicasa)", str(CACHE))
    if FACADES.exists():
        validate_facades(rep)
    else:
        rep.expect("facades-present", False, "CMP Facades exist (dvc pull data/raw/facades.dvc)", str(FACADES))
    result = {"passed": rep.passed, "n_checks": len(rep.checks), "n_failed": sum(not c["passed"] for c in rep.checks),
              "sample_per_split": sample, "elapsed_s": round(time.time() - t0, 1), "checks": rep.checks}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Data validation gate (fails on broken data)")
    ap.add_argument("--sample", type=int, default=None, help="check N random files per split (quick mode)")
    a = ap.parse_args()
    r = run(a.sample)
    for c in r["checks"]:
        print(("PASS " if c["passed"] else "FAIL ") + f"{c['id']:<20} {c['expectation']}" + ("" if c["passed"] else f"  -> {c['detail']}"))
    print(f"\n{r['n_checks'] - r['n_failed']}/{r['n_checks']} expectations passed in {r['elapsed_s']} s")
    sys.exit(0 if r["passed"] else 1)
