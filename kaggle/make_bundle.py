"""
Build zip(s) to upload as Kaggle Datasets.

    python kaggle/make_bundle.py                  # everything (~730 MB)
    python kaggle/make_bundle.py --part facades   # code + CMP Facades + external plans (~65 MB)
    python kaggle/make_bundle.py --part cubicasa  # rasterized CubiCasa5K only (~700 MB) -> floorplan model

Outputs dist/archigen_<part>.zip. Each part can be uploaded as its own Kaggle Dataset;
the notebook (kaggle/archigen_kaggle.ipynb) finds whatever parts are attached.
Layout inside the zip:
    archigen/src/, archigen/params.yaml
    archigen/data/raw/facades/
    archigen/data/raw/{dev_external_plans,real_external_plans,external_test_plans}/
    archigen/data/processed/cubicasa_cache/
"""

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CODE = [
    ("src", lambda p: "__pycache__" not in p.parts and p.suffix != ".pyc"),
    ("params.yaml", None),
]
FACADES = [("data/raw/facades", lambda p: p.suffix.lower() == ".jpg")]
# Out-of-distribution floor plans (no labels, ~32 MB) for src.evaluation.evaluate_external.
_IMG = lambda p: p.suffix.lower() in (".jpg", ".jpeg", ".png", ".json")
EXTERNAL = [("data/raw/dev_external_plans", _IMG), ("data/raw/real_external_plans", _IMG),
            ("data/raw/external_test_plans", _IMG)]
CUBICASA = [("data/processed/cubicasa_cache", lambda p: p.suffix in (".npz", ".json"))]
PARTS = {"all": CODE + FACADES + EXTERNAL + CUBICASA, "facades": CODE + FACADES + EXTERNAL, "cubicasa": CUBICASA}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=PARTS, default="all")
    part = ap.parse_args().part
    if part in ("all", "cubicasa"):
        cache = ROOT / "data/processed/cubicasa_cache"
        for split in ("train", "val", "test"):
            if not any((cache / split).glob("*.npz")):
                raise SystemExit(f"Missing {cache / split}. Run: python -m src.data.rasterize_cubicasa")
    out = ROOT / "dist" / ("archigen_kaggle.zip" if part == "all" else f"archigen_{part}.zip")
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    # .npz/.jpg are already compressed: store them (fast); deflate only text files.
    with zipfile.ZipFile(out, "w") as z:
        for rel, keep in PARTS[part]:
            base = ROOT / rel
            files = [base] if base.is_file() else sorted(p for p in base.rglob("*") if p.is_file())
            for f in files:
                if keep and not keep(f.relative_to(ROOT)):
                    continue
                comp = zipfile.ZIP_STORED if f.suffix.lower() in (".npz", ".jpg", ".png") else zipfile.ZIP_DEFLATED
                z.write(f, Path("archigen") / f.relative_to(ROOT), compress_type=comp)
                n += 1
    print(f"Wrote {out} ({n} files, {out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
