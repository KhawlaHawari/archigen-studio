"""The data validation gate must pass on good data and FAIL on broken data."""
import numpy as np

import src.data.validate_data as vd


def _plan(walls=True, value=1):
    img = np.random.default_rng(0).integers(0, 255, (512, 512, 3), dtype=np.uint8)
    s = np.zeros((512, 512), np.uint8)
    if walls:
        s[:20] = value
    r = np.zeros((512, 512), np.uint8)
    r[100:200, 100:200] = 2
    return img, s, r


def _write(path, img, s, r):
    np.savez_compressed(path, image=img, structure_mask=s, room_mask=r)


def test_good_files_pass(tmp_path):
    f = tmp_path / "ok.npz"
    _write(f, *_plan())
    rep = vd.Report()
    vd.validate_npz_files([f], rep, "t")
    assert rep.passed


def test_bad_labels_blank_images_and_missing_walls_fail(tmp_path):
    img, s, r = _plan(value=9)                     # structure label 9 does not exist
    _write(tmp_path / "bad_label.npz", img, s, r)
    _write(tmp_path / "blank.npz", np.zeros((512, 512, 3), np.uint8), *_plan()[1:])
    _write(tmp_path / "no_walls.npz", *_plan(walls=False))
    rep = vd.Report()
    vd.validate_npz_files(sorted(tmp_path.glob("*.npz")), rep, "t")
    failed = {c["id"] for c in rep.checks if not c["passed"]}
    assert failed == {"t-E3", "t-E5"}


def test_wrong_shape_fails(tmp_path):
    f = tmp_path / "small.npz"
    np.savez_compressed(f, image=np.zeros((256, 256, 3), np.uint8), structure_mask=np.zeros((256, 256), np.uint8),
                        room_mask=np.zeros((256, 256), np.uint8))
    rep = vd.Report()
    vd.validate_npz_files([f], rep, "t")
    assert not rep.passed and any(c["id"] == "t-E2" and not c["passed"] for c in rep.checks)


def test_split_leakage_is_detected(tmp_path, monkeypatch):
    for split in ("train", "val", "test"):
        (tmp_path / split).mkdir()
    _write(tmp_path / "train" / "plan_1.npz", *_plan())
    _write(tmp_path / "test" / "plan_1.npz", *_plan())      # same plan id in train and test
    monkeypatch.setattr(vd, "CACHE", tmp_path)
    monkeypatch.setattr(vd, "FEEDBACK_CACHE", tmp_path / "none")
    rep = vd.Report()
    vd.validate_cubicasa(rep, None, __import__("random").Random(0))
    leak = next(c for c in rep.checks if c["id"] == "cubicasa-E4")
    assert not leak["passed"] and "train&test" in leak["detail"]
