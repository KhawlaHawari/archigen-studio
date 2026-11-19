"""
Install the models and metrics of a Kaggle run (archigen_results_runN.zip) into the project.

YOU choose which models to install (read the metrics first!). The script then does the rest:
  1. backs up the current metrics to models/metrics/<backup-tag>/ and copies the new ones;
  2. for every chosen model: renames the current file to *_<backup-tag> and installs the new one
     under the name the API loads (see src/config.py);
  3. re-exports the ONNX files and checks PyTorch/ONNX parity (the API loads ONNX first!);
  4. reminds you to restart the backend.

Examples (PowerShell, from the repo root):
    python scripts/install_run.py "$env:USERPROFILE\\Downloads\\archigen_results_run3.zip" --new-tag run3 --backup-tag run2 --floorplan --controlnet --dry-run
    python scripts/install_run.py "$env:USERPROFILE\\Downloads\\archigen_results_run3.zip" --new-tag run3 --backup-tag run2 --floorplan --controlnet

Rollback: rename models/checkpoints/floorplan_unetpp_<backup-tag>.pt back to floorplan_unetpp_best.pt,
then run `python -m src.mlops.export_onnx` and restart the backend.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CK, METRICS, MODELS = ROOT / "models" / "checkpoints", ROOT / "models" / "metrics", ROOT / "models"
# zip member -> file name the API loads
COMPONENTS = {
    "floorplan": ("checkpoints/floorplan_unetpp_best.pt", "floorplan_unetpp_best.pt"),
    "pix2pix": ("checkpoints/facade_pix2pix_best.pt", "facade_pix2pix_best.pt"),
    "parser": ("checkpoints/facade_parser_best.pt", "facade_parser_best.pt"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zip", type=Path, help="archigen_results_runN.zip downloaded from Kaggle")
    ap.add_argument("--new-tag", required=True, help="name of the run you install, e.g. run3")
    ap.add_argument("--backup-tag", required=True, help="name given to the files being replaced, e.g. run2")
    for name in ("floorplan", "pix2pix", "parser", "controlnet"):
        ap.add_argument(f"--{name}", action="store_true", help=f"install the {name} model of this run")
    ap.add_argument("--dry-run", action="store_true", help="only print what would be done")
    a = ap.parse_args()

    z = zipfile.ZipFile(a.zip)
    names = set(z.namelist())
    act = (lambda msg: print("  [dry-run]", msg)) if a.dry_run else (lambda msg: print("  ", msg))

    def extract(member: str, dest: Path) -> None:
        act(f"extract {member} -> {dest.relative_to(ROOT)}")
        if not a.dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as s, open(dest, "wb") as d:
                shutil.copyfileobj(s, d, length=16 << 20)

    def move(src: Path, dst: Path) -> None:
        act(f"rename {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
        if not a.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
            shutil.move(str(src), dst)

    print(f"1) Metrics: back up current -> models/metrics/{a.backup_tag}/, copy the {a.new_tag} metrics")
    for f in sorted(list(METRICS.glob("*.json")) + list(METRICS.glob("*.png"))):
        move(f, METRICS / a.backup_tag / f.name)
    for m in sorted(n for n in names if n.startswith("metrics/") and not n.endswith("/")):
        name = Path(m).name
        if not a.pix2pix:   # the run's Pix2Pix is NOT deployed: keep its metrics under its run name
            name = name.replace("facade_pix2pix_best", f"facade_pix2pix_{a.new_tag}").replace(
                "facade_pix2pix_val", f"facade_pix2pix_{a.new_tag}_val")
        extract(m, METRICS / name)

    print("2) Models")
    changed_onnx = False
    for comp, (member, target) in COMPONENTS.items():
        if member not in names:
            continue
        dest = CK / target
        if getattr(a, comp):
            if dest.exists():
                move(dest, CK / target.replace("_best.pt", f"_{a.backup_tag}.pt"))
            extract(member, dest)
            changed_onnx |= comp in ("floorplan", "pix2pix")
        else:   # keep a copy of the run's model without deploying it
            extract(member, CK / target.replace("_best.pt", f"_{a.new_tag}.pt"))
    if a.controlnet:
        members = [n for n in names if n.startswith("controlnet_facade/") and not n.endswith("/")]
        if not members:
            sys.exit("This zip has no controlnet_facade/ folder.")
        if (MODELS / "controlnet_facade").exists():
            move(MODELS / "controlnet_facade", MODELS / f"controlnet_facade_{a.backup_tag}")
        for m in members:
            extract(m, MODELS / m)

    print("3) ONNX (the API loads ONNX before .pt files)")
    if changed_onnx:
        for cmd in (["-m", "src.mlops.export_onnx"], ["-m", "src.mlops.onnx_verifier"]):
            act("python " + " ".join(cmd))
            if not a.dry_run:
                for f in (ROOT / "models" / "onnx").glob("*.onnx*"):
                    if cmd[1] == "src.mlops.export_onnx":
                        f.unlink()
                subprocess.run([sys.executable, *cmd], cwd=ROOT, check=True)
    else:
        print("   nothing to re-export (no floor-plan / Pix2Pix change)")

    print("4) Restart the backend so it loads the new models:  uvicorn src.api.main:app --port 8000")
    print("   Then check http://127.0.0.1:8000/facade/capabilities and update the README tables.")


if __name__ == "__main__":
    main()
