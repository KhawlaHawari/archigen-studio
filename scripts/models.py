"""
Model weights distribution with pinned SHA-256 checksums (weights are never stored in Git).

The manifest (models/manifest.json, committed) lists every weight file: path in the project, size,
SHA-256, download URL (GitHub Release asset of this repository, or the upstream URL for third-party
weights) and the profiles that need it.

Profiles
  runtime : what the API needs for plan -> 3D and fast facades (floor-plan Run 3, Pix2Pix Run 2, super-resolution)
  studio  : runtime + the ControlNet for high-quality facades
  ab      : runtime + the A/B challenger (floor-plan Run 2)
  eval    : runtime + the facade parser used by the FCN-score
  all     : everything

Maintainer (once per new model version):
    python scripts/models.py manifest --release-tag models-run3     # hash local files -> models/manifest.json
    python scripts/models.py assets --stage dist/release-models-run3   # copies the files with their asset names
    then create the GitHub Release "models-run3" and upload everything in that folder
Teammate / CI:
    python scripts/models.py pull --profile runtime                 # download + verify
    python scripts/models.py check --profile runtime                # verify only (exit 1 on mismatch)
    python -m src.mlops.export_onnx                                  # rebuild the ONNX files from the .pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models" / "manifest.json"
REPO = "KhawlaHawari/archigen-studio"
# path in project -> (release asset name, profiles); upstream URL instead of an asset for third-party weights
FILES = {
    "models/checkpoints/floorplan_unetpp_best.pt": ("floorplan_unetpp_run3.pt", ["runtime", "studio", "ab", "eval"]),
    "models/checkpoints/facade_pix2pix_best.pt": ("facade_pix2pix_run2.pt", ["runtime", "studio", "ab", "eval"]),
    "models/superres/realesr-general-x4v3.pth": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth", ["runtime", "studio", "ab", "eval"]),
    "models/controlnet_facade/config.json": ("controlnet_facade_config.json", ["studio"]),
    "models/controlnet_facade/diffusion_pytorch_model.safetensors": ("controlnet_facade_run3.safetensors", ["studio"]),
    "models/checkpoints/floorplan_unetpp_run2.pt": ("floorplan_unetpp_run2.pt", ["ab"]),
    "models/checkpoints/facade_parser_best.pt": ("facade_parser_run3.pt", ["eval"]),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict:
    if not MANIFEST.exists():
        sys.exit(f"No manifest at {MANIFEST}. The maintainer creates it with: python scripts/models.py manifest --release-tag <tag>")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def selected(manifest: dict, profile: str):
    return [f for f in manifest["files"] if profile == "all" or profile in f["profiles"]]


def cmd_manifest(tag: str) -> None:
    base = f"https://github.com/{REPO}/releases/download/{tag}/"
    files = []
    for rel, (asset, profiles) in FILES.items():
        p = ROOT / rel
        if not p.exists():
            print(f"  skip (not on disk): {rel}")
            continue
        url = asset if asset.startswith("http") else base + asset
        files.append({"path": rel, "size": p.stat().st_size, "sha256": sha256(p), "url": url,
                      "release_asset": None if asset.startswith("http") else asset, "profiles": profiles})
        print(f"  {rel}  {p.stat().st_size / 1e6:8.1f} MB  {files[-1]['sha256'][:16]}...")
    MANIFEST.write_text(json.dumps({"release_tag": tag, "repository": REPO, "files": files}, indent=2), encoding="utf-8")
    print(f"Wrote {MANIFEST} ({len(files)} files). Next: create the GitHub Release '{tag}' and upload the assets "
          f"(python scripts/models.py assets).")


def cmd_assets(stage: str | None = None) -> None:
    m = load_manifest()
    print(f"Upload these files to the GitHub Release '{m['release_tag']}' with EXACTLY these asset names:")
    out = Path(stage) if stage else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    for f in m["files"]:
        if f["release_asset"]:
            print(f"  {f['path']:<62} -> {f['release_asset']}")
            if out:
                import shutil
                shutil.copyfile(ROOT / f["path"], out / f["release_asset"])
    if out:
        print(f"Copied with the right names into {out}: drag them all into the release page, or run\n"
              f"  gh release create {m['release_tag']} {out}/* --title \"Model weights {m['release_tag']}\" --notes \"See models/manifest.json\"")


def download(url: str, dest: Path, expected_sha: str, expected_size: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False, suffix=".part") as tmp:
        tmp_path = Path(tmp.name)
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                for chunk in iter(lambda: r.read(1 << 20), b""):
                    tmp.write(chunk)
        except BaseException:
            tmp.close()
            tmp_path.unlink(missing_ok=True)
            raise
    if tmp_path.stat().st_size != expected_size or sha256(tmp_path) != expected_sha:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {url}")
    tmp_path.replace(dest)   # atomic: a half-downloaded file never replaces a good one


def cmd_pull(profile: str, skip_if_unpublished: bool) -> int:
    m = load_manifest()
    for f in selected(m, profile):
        dest = ROOT / f["path"]
        if dest.exists() and dest.stat().st_size == f["size"] and sha256(dest) == f["sha256"]:
            print(f"  ok        {f['path']}")
            continue
        print(f"  download  {f['path']}  ({f['size'] / 1e6:.0f} MB)")
        try:
            download(f["url"], dest, f["sha256"], f["size"])
        except urllib.error.HTTPError as e:
            if skip_if_unpublished and e.code == 404:
                print(f"  release asset not published yet ({f['url']}); skipping (--skip-if-unpublished)")
                return 0
            raise
    print("All files verified. Rebuild the ONNX files with: python -m src.mlops.export_onnx")
    return 0


def cmd_check(profile: str) -> int:
    m, bad = load_manifest(), 0
    for f in selected(m, profile):
        p = ROOT / f["path"]
        if not p.exists():
            print(f"  MISSING   {f['path']}")
            bad += 1
        elif p.stat().st_size != f["size"] or sha256(p) != f["sha256"]:
            print(f"  MISMATCH  {f['path']}")
            bad += 1
        else:
            print(f"  ok        {f['path']}")
    print("All files match the manifest." if not bad else f"{bad} file(s) missing or different from the manifest.")
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("manifest"); s.add_argument("--release-tag", required=True)
    s = sub.add_parser("assets"); s.add_argument("--stage", default=None, help="copy the files, renamed, into this folder")
    for name in ("pull", "check"):
        s = sub.add_parser(name)
        s.add_argument("--profile", default="runtime", choices=["runtime", "studio", "ab", "eval", "all"])
        if name == "pull":
            s.add_argument("--skip-if-unpublished", action="store_true")
    a = ap.parse_args()
    if a.cmd == "manifest":
        cmd_manifest(a.release_tag)
    elif a.cmd == "assets":
        cmd_assets(a.stage)
    elif a.cmd == "pull":
        sys.exit(cmd_pull(a.profile, a.skip_if_unpublished))
    else:
        sys.exit(cmd_check(a.profile))
