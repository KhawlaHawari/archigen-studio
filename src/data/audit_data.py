"""
ArchiGen 3D - REAL data audit (replaces the old synthetic npz_caching stage).

CMP Facades (data/raw/facades):
  counts per split, PIL verify + full decode, size/mode, exact duplicate files
  (SHA-256 of bytes), duplicates after decoding (SHA-256 of pixels, per half,
  across all splits), photo mean-RGB statistics per split.

CubiCasa5K (raw + data/processed/cubicasa_cache):
  official split sizes, missing files, overlap between splits, exact duplicate
  F1_scaled.png files across the dataset, and EXACT pixel frequencies per class
  and per split (lossless masks), incl. share of ignored room pixels.

Output: data/processed/data_audit_summary.json

    python -m src.data.audit_data
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, ROOM_CLASSES, STRUCTURE_CLASSES
from src.data.facade_dataset import FACADES_DIR, list_split
from src.data.rasterize_cubicasa import DEFAULT_OUT as CUBI_CACHE, DEFAULT_SOURCE as CUBI_RAW
from src.data.rasterize_cubicasa import IGNORE_INDEX, find_dataset_root, read_split_file

SPLITS = ("train", "val", "test")
OUT = PROCESSED_DATA_DIR / "data_audit_summary.json"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _dup_groups(hash_to_items: Dict[str, List[str]]) -> List[List[str]]:
    return [v for v in hash_to_items.values() if len(v) > 1]


def audit_facades(root: Path = FACADES_DIR) -> Dict[str, Any]:
    file_h, photo_h, map_h = defaultdict(list), defaultdict(list), defaultdict(list)
    report: Dict[str, Any] = {"splits": {}, "corrupted": [], "bad_size_or_mode": []}
    for split in SPLITS:
        files = list_split(split, root)
        means = []
        for f in files:
            tag = f"{split}/{f.name}"
            raw = f.read_bytes()
            file_h[_sha(raw)].append(tag)
            try:
                with Image.open(f) as im:
                    im.verify()
                with Image.open(f) as im:
                    mode, size = im.mode, im.size
                    arr = np.asarray(im.convert("RGB"))
            except Exception as exc:
                report["corrupted"].append({"file": tag, "error": str(exc)})
                continue
            if mode != "RGB" or size != (512, 256):
                report["bad_size_or_mode"].append({"file": tag, "mode": mode, "size": size})
            photo, sem = arr[:, :256], arr[:, 256:]
            photo_h[_sha(photo.tobytes())].append(tag)
            map_h[_sha(sem.tobytes())].append(tag)
            means.append(float(photo.mean()))
        m = np.array(means)
        report["splits"][split] = {
            "pairs": len(files),
            "photo_mean_rgb": {"mean": round(m.mean(), 2), "min": round(m.min(), 2), "max": round(m.max(), 2)},
        }
    total = sum(s["pairs"] for s in report["splits"].values())
    report.update({
        "total_pairs": total,
        "split_share_pct": {s: round(100 * v["pairs"] / total, 2) for s, v in report["splits"].items()},
        "duplicate_files_sha256": _dup_groups(file_h),
        "duplicate_photos_pixels": _dup_groups(photo_h),
        "duplicate_maps_pixels": _dup_groups(map_h),
        "note": "Semantic maps are JPEG-compressed: exact per-class pixel counts are not possible from this archive.",
    })
    return report


def audit_cubicasa(raw: Path = CUBI_RAW, cache: Path = CUBI_CACHE) -> Dict[str, Any]:
    root = find_dataset_root(raw)
    report: Dict[str, Any] = {"root": str(root), "splits": {}}
    split_sets = {s: read_split_file(root / f"{s}.txt") for s in SPLITS}
    report["split_overlap"] = {
        f"{a}&{b}": len(set(split_sets[a]) & set(split_sets[b]))
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    img_h = defaultdict(list)
    missing = []
    for split, rels in split_sets.items():
        for rel in rels:
            d = root / rel
            for fn in ("F1_scaled.png", "model.svg"):
                if not (d / fn).exists():
                    missing.append(f"{rel}/{fn}")
            if (d / "F1_scaled.png").exists():
                img_h[_sha((d / "F1_scaled.png").read_bytes())].append(f"{split}:{rel}")
    dups = _dup_groups(img_h)
    report["missing_files"] = missing
    report["duplicate_images_sha256"] = {
        "groups": len(dups),
        "cross_split_groups": sum(1 for g in dups if len({x.split(":")[0] for x in g}) > 1),
        "examples": dups[:10],
    }

    manifest_path = Path(cache) / "manifest.json"
    if manifest_path.exists():
        man = json.loads(manifest_path.read_text())
        report["rasterization"] = {"counts": man.get("counts"), "failures": len(man.get("failures", [])),
                                   "unmapped_room_types": man.get("unmapped_room_types")}

    for split, rels in split_sets.items():
        files = sorted((Path(cache) / split).glob("*.npz"))
        s_cnt = np.zeros(len(STRUCTURE_CLASSES), np.int64)
        r_cnt = np.zeros(256, np.int64)
        s_presence = np.zeros(len(STRUCTURE_CLASSES), np.int64)
        r_presence = np.zeros(len(ROOM_CLASSES), np.int64)
        for f in files:
            with np.load(f) as d:
                s, r = d["structure_mask"], d["room_mask"]
            sc = np.bincount(s.ravel(), minlength=len(STRUCTURE_CLASSES))[: len(STRUCTURE_CLASSES)]
            rc = np.bincount(r.ravel(), minlength=256)
            s_cnt += sc
            r_cnt += rc
            s_presence += sc > 0
            r_presence += rc[: len(ROOM_CLASSES)] > 0
        tot = max(1, int(s_cnt.sum()))
        report["splits"][split] = {
            "houses_listed": len(rels),
            "houses_rasterized": len(files),
            "structure_pixel_pct": {c: round(100 * s_cnt[i] / tot, 3) for i, c in enumerate(STRUCTURE_CLASSES)},
            "structure_presence_pct": {c: round(100 * s_presence[i] / max(1, len(files)), 1)
                                       for i, c in enumerate(STRUCTURE_CLASSES)},
            "room_pixel_pct": {c: round(100 * r_cnt[i] / tot, 3) for i, c in enumerate(ROOM_CLASSES)},
            "room_ignored_pixel_pct": round(100 * r_cnt[IGNORE_INDEX] / tot, 3),
            "room_presence_pct": {c: round(100 * r_presence[i] / max(1, len(files)), 1)
                                  for i, c in enumerate(ROOM_CLASSES)},
        }
    return report


def run_audit(out: Path = OUT) -> Dict[str, Any]:
    t0 = time.time()
    report: Dict[str, Any] = {"date": time.strftime("%Y-%m-%d %H:%M:%S")}
    report["cmp_facades"] = audit_facades()
    try:
        report["cubicasa5k"] = audit_cubicasa()
    except FileNotFoundError as exc:
        report["cubicasa5k"] = {"status": "not available", "reason": str(exc)}
    report["seconds"] = round(time.time() - t0, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"Audit written to {out} ({report['seconds']} s)")
    return report


if __name__ == "__main__":
    r = run_audit()
    f = r["cmp_facades"]
    print(f"Facades: {f['total_pairs']} pairs {f['split_share_pct']} | corrupted={len(f['corrupted'])} "
          f"| dup files={len(f['duplicate_files_sha256'])} | dup photos={len(f['duplicate_photos_pixels'])} "
          f"| dup maps={len(f['duplicate_maps_pixels'])}")
    c = r["cubicasa5k"]
    if "splits" in c:
        counts = {s: v["houses_rasterized"] for s, v in c["splits"].items()}
        print(f"CubiCasa: {counts} | "
              f"overlap={c['split_overlap']} | missing={len(c['missing_files'])} | dups={c['duplicate_images_sha256']}")
