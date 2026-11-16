"""Unit tests for the Run-2 ("v2") training recipe pieces. No network access or datasets required."""

import numpy as np
import pytest
import torch

from src.evaluation.evaluate import _predict_tta
from src.models.unet_plus_plus import FloorplanMultiTaskLoss, ResNet34Encoder, load_imagenet_resnet34
from src.training.train_facade import lambda_lr_factor


def test_lr_factor_constant_then_linear_decay():
    f = lambda_lr_factor(decay_start=50, total_epochs=100)
    assert f(0) == 1.0 and f(49) == 1.0
    assert 0.0 < f(50) < 1.0
    assert f(99) < f(75) < f(50)
    assert f(99) >= 0.0


def test_imagenet_mapping_copies_layer_weights():
    torchvision = pytest.importorskip("torchvision")
    src = torchvision.models.resnet34(weights=None).state_dict()   # random weights, no download
    enc = ResNet34Encoder()
    copied = load_imagenet_resnet34(enc, source_state=src)
    assert copied > 150
    dst = enc.state_dict()
    assert torch.equal(dst["layer3.2.conv1.weight"], src["layer3.2.conv1.weight"])
    assert torch.equal(dst["layer2.0.shortcut.0.weight"], src["layer2.0.downsample.0.weight"])


def test_room_weights_change_the_loss_and_keep_ignore():
    torch.manual_seed(0)
    s_logits, r_logits = torch.randn(1, 5, 8, 8), torch.randn(1, 12, 8, 8)
    s_t, r_t = torch.randint(0, 5, (1, 8, 8)), torch.randint(0, 12, (1, 8, 8))
    r_t[:, :2] = 255
    plain = FloorplanMultiTaskLoss()(s_logits, s_t, r_logits, r_t)["loss_room_ce"]
    weights = [0.1] + [1.0] * 10 + [5.0]
    weighted = FloorplanMultiTaskLoss(room_weights=weights)(s_logits, s_t, r_logits, r_t)["loss_room_ce"]
    assert torch.isfinite(weighted) and not torch.isclose(plain, weighted)


def test_tta_returns_probabilities_aligned_with_input():
    class Probe(torch.nn.Module):
        """Predicts class 1 exactly on the left half of the input; TTA must un-flip correctly."""
        def forward(self, x):
            left = (x[:, :1] > 0).float()
            logits = torch.cat([-10 * left, 10 * left], 1)
            return logits[:, :2], logits
    img = torch.zeros(1, 3, 4, 4)
    img[..., :2] = 1.0
    s, r = _predict_tta(Probe(), img)
    assert torch.allclose(s.sum(1), torch.ones(1, 4, 4), atol=1e-5)
    assert (s.argmax(1)[0, :, :2] == 1).all() and (s.argmax(1)[0, :, 2:] == 0).all()


def test_facade_jitter_keeps_size_and_pairing():
    from src.data.facade_dataset import FACADES_DIR, CMPFacadesDataset
    if not (FACADES_DIR / "train").is_dir():
        pytest.skip("CMP Facades not pulled")
    torch.manual_seed(1)
    m, p = CMPFacadesDataset("train", jitter=True)[0]
    assert m.shape == p.shape == (3, 256, 256)
    assert CMPFacadesDataset("val", jitter=True).jitter is False   # never on val/test
