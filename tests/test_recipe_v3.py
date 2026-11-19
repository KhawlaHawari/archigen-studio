"""Unit tests for the Run-3 ("v3") pieces and the layout evaluation. No datasets or network needed."""

import numpy as np
import torch

from src.data.style_augment import STYLES, stylize
from src.evaluation.evaluate_layout import centroids, match, one_hot, prf
from src.models.pix2pix_cgan import FacadeCompositeLoss, MultiScaleDiscriminator


def _toy_plan():
    img = np.full((64, 64, 3), 230, np.uint8)
    img[..., 1] = 240
    struct = np.zeros((64, 64), np.uint8)
    struct[8:12, 8:56] = 1        # exterior wall
    struct[8:56, 30:33] = 2       # interior wall
    struct[8:12, 40:48] = 3       # door in the top wall
    struct[20:28, 8:11] = 4       # window
    img[np.isin(struct, (1, 2))] = (20, 120, 20)   # CubiCasa-like green walls
    return img, struct


def test_stylize_keeps_shape_and_changes_only_the_image():
    img, struct = _toy_plan()
    before = struct.copy()
    for i, style in enumerate(STYLES):
        out = stylize(img, struct, np.random.default_rng(i), style)
        assert out.shape == img.shape and out.dtype == np.uint8
        assert np.array_equal(struct, before)          # labels are never modified
        assert not np.array_equal(out, img)


def test_stylize_is_deterministic_for_a_seed():
    img, struct = _toy_plan()
    a = stylize(img, struct, np.random.default_rng(7))
    b = stylize(img, struct, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_multiscale_discriminator_and_loss():
    d = MultiScaleDiscriminator(in_channels=6, ndf=8, num_scales=2)
    cond, img = torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64)
    preds, feats = d(cond, img)
    assert len(preds) == 2 and len(feats) == 8
    assert preds[1].shape[-1] < preds[0].shape[-1]      # second scale sees a half-size input
    loss = FacadeCompositeLoss()
    d_loss = loss.discriminator_loss(preds, [p.detach() * 0 for p in preds])
    assert torch.isfinite(d_loss["d_total"])
    g = loss.generator_loss(preds, feats, feats, img, img)
    assert torch.isfinite(g["g_total"]) and g["g_fm"].item() == 0.0


def test_detection_matching_counts():
    mask = np.zeros((50, 50), bool)
    mask[5:9, 5:9] = True
    mask[30:34, 30:34] = True
    gt = centroids(mask)
    pred = gt + np.array([[2.0, 0.0], [30.0, 0.0]])     # one close, one far away
    tp, dists = match(pred, gt, radius=12)
    assert tp == 1 and abs(dists[0] - 2.0) < 1e-6
    r = prf(tp, len(pred), len(gt))
    assert r["precision"] == 0.5 and r["recall"] == 0.5 and r["fp"] == 1 and r["fn"] == 1


def test_one_hot_maps_ignore_to_background():
    m = np.array([[0, 2], [255, 1]], np.uint8)
    oh = one_hot(m, 3)
    assert oh.shape == (3, 2, 2) and np.array_equal(oh.argmax(0), [[0, 2], [0, 1]])


def test_cmp_label_colours_decode_to_their_classes():
    from src.data.facade_palette import CMP_TRAINING_LUT
    from src.models.facade_parser import CMP_CLASSES, rgb_to_class
    img = CMP_TRAINING_LUT[1:13][None].astype(np.uint8)            # one pixel per class
    assert rgb_to_class(img).tolist() == [list(range(len(CMP_CLASSES)))]
    noisy = np.clip(img.astype(int) + 6, 0, 255).astype(np.uint8)   # JPEG-like shift keeps the class
    assert rgb_to_class(noisy).tolist() == [list(range(len(CMP_CLASSES)))]


def test_controlnet_inputs_are_512_and_in_range():
    from src.training.train_controlnet import to_512
    masks, photos = torch.rand(2, 3, 256, 256) * 2 - 1, torch.rand(2, 3, 256, 256) * 2 - 1
    cond, target = to_512(masks, photos)
    assert cond.shape == target.shape == (2, 3, 512, 512)
    assert 0 <= cond.min() and cond.max() <= 1 and -1 <= target.min() and target.max() <= 1


def test_lovasz_softmax_rewards_correct_predictions_and_ignores_255():
    from src.models.unet_plus_plus import FloorplanMultiTaskLoss, lovasz_softmax
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[:, :, 4:] = 2
    good = torch.full((1, 3, 8, 8), -5.0)
    good[:, 0, :, :4] = 5.0
    good[:, 2, :, 4:] = 5.0
    assert lovasz_softmax(good, target) < 0.05 < lovasz_softmax(-good, target)
    masked = target.clone()
    masked[:, :, :4] = 255
    assert torch.isfinite(lovasz_softmax(torch.randn(1, 3, 8, 8), masked))
    s_logits, r_logits = torch.randn(1, 5, 8, 8, requires_grad=True), torch.randn(1, 12, 8, 8, requires_grad=True)
    out = FloorplanMultiTaskLoss(room_lovasz_weight=0.5)(s_logits, torch.randint(0, 5, (1, 8, 8)),
                                                         r_logits, torch.randint(0, 12, (1, 8, 8)))
    out["total_loss"].backward()
    assert out["loss_room_lovasz"] > 0 and r_logits.grad is not None


def test_superres_network_is_4x_and_loads_public_weights_if_present():
    import pytest
    from PIL import Image
    from src.models import superres
    net = superres.SRVGGNetCompact()
    assert net(torch.rand(1, 3, 10, 12)).shape == (1, 3, 40, 48)
    if not superres.available():
        pytest.skip("Real-ESRGAN weights not downloaded")
    out = superres.upscale(Image.new("RGB", (30, 20), (120, 100, 90)))
    assert out.size == (120, 80)
