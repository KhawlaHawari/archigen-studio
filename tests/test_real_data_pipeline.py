"""Tests for the real-data pipeline: loaders, ignore labels, metrics, checkpoint safety."""

import numpy as np
import pytest
import torch

from src.config import ROOM_CLASSES
from src.data.facade_dataset import FACADES_DIR, CMPFacadesDataset
from src.data.rasterize_cubicasa import IGNORE_INDEX, ROOM_TYPE_TO_ID
from src.evaluation.metrics import ConfusionMatrix, l1_per_image, psnr_per_image, ssim_per_image
from src.models.pix2pix_cgan import FacadeGenerator
from src.models.unet_plus_plus import FloorplanMultiTaskLoss
from src.training.checkpoints import CheckpointError, load_weights

needs_facades = pytest.mark.skipif(not (FACADES_DIR / "train").is_dir(), reason="CMP Facades not pulled")


@needs_facades
def test_facade_splits_and_tensor_range():
    sizes = {s: len(CMPFacadesDataset(s)) for s in ("train", "val", "test")}
    assert sizes == {"train": 400, "val": 100, "test": 106}
    m, p = CMPFacadesDataset("val")[0]
    assert m.shape == p.shape == (3, 256, 256)
    assert -1.0 <= m.min() and m.max() <= 1.0 and -1.0 <= p.min() and p.max() <= 1.0


@needs_facades
def test_val_and_test_are_not_augmented():
    ds = CMPFacadesDataset("test")
    a, b = ds[3], ds[3]
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_room_mapping_targets_valid_classes():
    for name, rid in ROOM_TYPE_TO_ID.items():
        assert rid == IGNORE_INDEX or 1 <= rid < len(ROOM_CLASSES), name
    assert ROOM_TYPE_TO_ID["Undefined"] == IGNORE_INDEX  # never a fake "living room"


def test_loss_ignores_unknown_room_pixels():
    torch.manual_seed(0)
    loss = FloorplanMultiTaskLoss()
    s_logits, r_logits = torch.randn(1, 5, 8, 8), torch.randn(1, 12, 8, 8)
    s_t = torch.randint(0, 5, (1, 8, 8))
    r_t = torch.randint(0, 12, (1, 8, 8))
    r_t_ign = r_t.clone()
    r_t_ign[:, 4:] = IGNORE_INDEX
    # Changing the logits only where the target is ignored must not change the loss.
    r_logits2 = r_logits.clone()
    r_logits2[:, :, 4:] += 100.0
    a = loss(s_logits, s_t, r_logits, r_t_ign)["loss_room"]
    b = loss(s_logits, s_t, r_logits2, r_t_ign)["loss_room"]
    assert torch.isfinite(a) and torch.allclose(a, b, atol=1e-5)


def test_confusion_matrix_skips_ignore():
    cm = ConfusionMatrix(3, ignore_index=255)
    target = torch.tensor([0, 1, 2, 255])
    pred = torch.tensor([0, 1, 1, 0])
    cm.update(pred, target)
    r = cm.compute()
    assert r["pixel_acc"] == pytest.approx(2 / 3)
    assert r["per_class_iou"]["2"] == 0.0


def test_image_metrics_identity():
    x = torch.rand(2, 3, 32, 32) * 2 - 1
    assert torch.allclose(l1_per_image(x, x), torch.zeros(2))
    assert (psnr_per_image(x, x) > 90).all()
    assert torch.allclose(ssim_per_image(x, x), torch.ones(2), atol=1e-4)


def test_untrained_or_synthetic_checkpoints_are_refused(tmp_path):
    g = FacadeGenerator()
    with pytest.raises(CheckpointError):
        load_weights(g, tmp_path / "missing.pt")
    bare = tmp_path / "bare.pt"
    torch.save(g.state_dict(), bare)
    load_weights(g, bare)  # loads fine without the real-data requirement ...
    with pytest.raises(CheckpointError):
        load_weights(g, bare, require_real_data=True)  # ... but is refused for evaluation
    real = tmp_path / "real.pt"
    torch.save({"model_state_dict": g.state_dict(), "data_source": "real:cmp_facades"}, real)
    assert load_weights(g, real, require_real_data=True)["data_source"] == "real:cmp_facades"
