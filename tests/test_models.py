"""
Unit tests for PyTorch Deep Learning Model Architectures
"""

import torch
import pytest
from src.models.unet_plus_plus import UNetPlusPlusMultiTask, FloorplanMultiTaskLoss
from src.models.pix2pix_cgan import FacadeGenerator, PatchGANDiscriminator, FacadeCompositeLoss


def test_unetpp_forward_shapes():
    """Verifies that UNet++ dual heads output expected tensor dimensions"""
    model = UNetPlusPlusMultiTask(in_channels=3, num_structure_classes=5, num_room_classes=12)
    model.eval()

    # Fast test input [1, 3, 256, 256] or [1, 3, 512, 512]
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        struct_logits, room_logits = model(x)

    assert struct_logits.shape == (1, 5, 256, 256), f"Unexpected struct shape: {struct_logits.shape}"
    assert room_logits.shape == (1, 12, 256, 256), f"Unexpected room shape: {room_logits.shape}"


def test_floorplan_loss_backward():
    """Verifies that the composite multi-task loss computes gradients correctly"""
    model = UNetPlusPlusMultiTask(in_channels=3, num_structure_classes=5, num_room_classes=12)
    criterion = FloorplanMultiTaskLoss(struct_weights=[0.5, 4.0, 4.0, 25.0, 10.0])

    x = torch.randn(1, 3, 128, 128)
    struct_targets = torch.randint(0, 5, (1, 128, 128))
    room_targets = torch.randint(0, 12, (1, 128, 128))

    struct_logits, room_logits = model(x)
    loss_dict = criterion(struct_logits, struct_targets, room_logits, room_targets)

    assert "total_loss" in loss_dict
    assert loss_dict["total_loss"].item() > 0

    loss_dict["total_loss"].backward()
    # Check that gradients flow to encoder parameters
    first_param = next(model.encoder.parameters())
    assert first_param.grad is not None


def test_facade_generator_shape_and_bounds():
    """Verifies UNet-256 nearest-neighbor generator output shape and [-1, 1] range"""
    gen = FacadeGenerator(in_channels=3, out_channels=3, ngf=32)
    gen.eval()

    x = torch.randn(1, 3, 256, 256).clamp(-1.0, 1.0)
    with torch.no_grad():
        out = gen(x)

    assert out.shape == (1, 3, 256, 256)
    assert out.min().item() >= -1.0
    assert out.max().item() <= 1.0


def test_patchgan_spectral_norm_discriminator():
    """Verifies PatchGAN 70x70 with spectral normalization returns scalar patch grid and features"""
    disc = PatchGANDiscriminator(in_channels=6, ndf=32, use_spectral_norm=True)
    disc.eval()

    cond = torch.randn(1, 3, 256, 256)
    target = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        out, features = disc(cond, target)

    # PatchGAN produces 30x30 receptive field patches at 256x256
    assert out.dim() == 4
    assert out.shape[1] == 1
    assert len(features) == 4, f"Expected 4 intermediate feature maps for FM loss, got {len(features)}"
