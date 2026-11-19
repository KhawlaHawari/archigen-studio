"""Bootstrap confidence intervals, fairness analysis and the weights manifest."""
import json
import sys
from pathlib import Path

import numpy as np

from src.evaluation.bootstrap import bootstrap, iou, seg_metrics
from src.evaluation.fairness import analyse, terciles

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import models as weights  # noqa: E402  (scripts/models.py)


def _cm(n, diag, off=0):
    return np.full((n, n), off, np.int64) + np.diag(np.full(n, diag, np.int64))


def test_iou_and_metric_definitions():
    cm = np.array([[8, 2], [2, 8]])
    assert np.allclose(iou(cm), [8 / 12, 8 / 12])
    m = seg_metrics(_cm(5, 100), _cm(12, 50))
    assert m["wall_iou"] == 1.0 and m["room_miou_11"] == 1.0 and m["room_pixel_acc"] == 1.0


def test_bootstrap_interval_contains_the_estimate_and_shrinks_with_n():
    rng = np.random.default_rng(0)
    small, large = rng.normal(0.7, 0.1, 30), rng.normal(0.7, 0.1, 600)
    ci_s = bootstrap(len(small), lambda i: {"m": small[i].mean()}, b=500)["m"]
    ci_l = bootstrap(len(large), lambda i: {"m": large[i].mean()}, b=500)["m"]
    assert ci_s["ci95_low"] <= ci_s["value"] <= ci_s["ci95_high"]
    assert (ci_l["ci95_high"] - ci_l["ci95_low"]) < (ci_s["ci95_high"] - ci_s["ci95_low"])


def test_fairness_flags_a_real_gap_and_not_noise():
    good_s, bad_s = _cm(5, 1000, 10), _cm(5, 300, 150)          # group B has much worse walls
    cm_s = np.stack([good_s] * 30 + [bad_s] * 30)
    cm_r = np.stack([_cm(12, 500, 5)] * 60)
    groups = np.array(["A"] * 30 + ["B"] * 30)
    res = analyse(cm_s, cm_r, groups, b=200)
    wall = res["disparity"]["wall_iou"]
    assert wall["worst_group"] == "B" and wall["flagged"] and wall["gap_points"] > 10
    assert not res["disparity"]["room_pixel_acc"]["flagged"]     # identical rooms -> no flag


def test_terciles_split_into_three_groups():
    t = terciles(np.arange(90, dtype=float))
    assert {str(x) for x in t} == {"small", "medium", "large"} and (t == "small").sum() == 30


def test_weights_check_detects_a_modified_file(tmp_path, monkeypatch):
    f = tmp_path / "w.pt"
    f.write_bytes(b"weights v1")
    manifest = {"release_tag": "t", "files": [{"path": "w.pt", "size": f.stat().st_size, "sha256": weights.sha256(f),
                                               "url": "https://example.invalid/w.pt", "release_asset": "w.pt", "profiles": ["runtime"]}]}
    monkeypatch.setattr(weights, "ROOT", tmp_path)
    monkeypatch.setattr(weights, "MANIFEST", tmp_path / "manifest.json")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert weights.cmd_check("runtime") == 0
    f.write_bytes(b"weights v2 (tampered)")
    assert weights.cmd_check("runtime") == 1
