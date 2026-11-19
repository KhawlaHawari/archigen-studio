"""
Facade parser (photo -> 12 CMP classes) used to measure LAYOUT FIDELITY of generated facades.

FCN-score protocol (Isola et al., pix2pix, 2017): a segmentation network trained on real photos
parses each generated facade; the parsed labels are compared with the label map the generator was
given. A realistic image that ignores the requested layout (e.g. missing pillars) scores low, so
this complements KID / LPIPS, which only measure realism.

Classes follow the CMP Facade Database: 0 background, 1 facade, 2 window, 3 door, 4 cornice,
5 sill, 6 balcony, 7 blind, 8 deco, 9 molding, 10 pillar, 11 shop.
"""
from __future__ import annotations

import numpy as np
import torch

from src.data.facade_palette import CMP_TRAINING_LUT

CMP_CLASSES = ["background", "facade", "window", "door", "cornice", "sill",
               "balcony", "blind", "deco", "molding", "pillar", "shop"]
_LUT = CMP_TRAINING_LUT[1:13].astype(np.int32)          # 12 colours, index = class id


def rgb_to_class(label_rgb: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 CMP label map (JPEG-compressed) -> (H, W) class ids by nearest palette colour."""
    flat = label_rgb.reshape(-1, 3).astype(np.int32)
    d = ((flat[:, None, :] - _LUT[None, :, :]) ** 2).sum(-1)
    return d.argmin(1).reshape(label_rgb.shape[:2]).astype(np.int64)


def labels_from_tensor(masks: torch.Tensor) -> torch.Tensor:
    """[-1, 1] (N, 3, H, W) label-map tensor -> (N, H, W) long class ids."""
    rgb = ((masks.detach().cpu().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).numpy()
    return torch.from_numpy(np.stack([rgb_to_class(m) for m in rgb]))


def build_parser(num_classes: int = len(CMP_CLASSES), pretrained_backbone: bool = True) -> torch.nn.Module:
    """DeepLabV3-ResNet50 with an ImageNet backbone (different architecture from the generators)."""
    from torchvision.models import ResNet50_Weights
    from torchvision.models.segmentation import deeplabv3_resnet50
    return deeplabv3_resnet50(weights=None, num_classes=num_classes, aux_loss=False,
                              weights_backbone=ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None)


_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def parser_input(images: torch.Tensor) -> torch.Tensor:
    """[-1, 1] RGB images -> ImageNet-normalised input."""
    x = (images.clamp(-1, 1) + 1) / 2
    return (x - _MEAN.to(x.device)) / _STD.to(x.device)


@torch.no_grad()
def parse(parser: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    return parser(parser_input(images))["out"].argmax(1)
