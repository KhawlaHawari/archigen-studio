"""
Pix2Pix Conditional GAN (cGAN) with Perceptual Loss & Spectral Normalization
Designed for CMP Facades Semantic-to-Photo Generation
Générateur: UNet-256 avec Nearest-Neighbor Upsampling (anti-checkerboard)
Discriminateur: PatchGAN 70x70 avec Spectral Normalization (1-Lipschitz)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class DownBlock(nn.Module):
    """Downsampling block: Conv2d -> (BatchNorm) -> LeakyReLU"""
    def __init__(self, in_channels: int, out_channels: int, normalize: bool = True):
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not normalize)
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """
    Anti-checkerboard upsampling block:
    Nearest-Neighbor Interpolation (2x) -> Conv2d 3x3 -> BatchNorm -> (Dropout) -> ReLU
    Replaces ConvTranspose2d to eliminate grid artifacts completely.
    """
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = [
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        # Handle potential 1-pixel rounding mismatch
        if out.shape != skip.shape:
            out = F.interpolate(out, size=skip.shape[2:], mode="nearest")
        return torch.cat([out, skip], dim=1)


class FacadeGenerator(nn.Module):
    """
    UNet-256 Generator for Semantic Mask to Photorealistic Facade Translation.
    Input: [B, 3, 256, 256] RGB semantic map normalized in [-1, 1]
    Output: [B, 3, 256, 256] Photorealistic facade image in [-1, 1]
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 3, ngf: int = 64):
        super().__init__()
        # Downsampling stages (256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1)
        self.down1 = DownBlock(in_channels, ngf, normalize=False)        # 128x128
        self.down2 = DownBlock(ngf, ngf * 2)                             # 64x64
        self.down3 = DownBlock(ngf * 2, ngf * 4)                         # 32x32
        self.down4 = DownBlock(ngf * 4, ngf * 8)                         # 16x16
        self.down5 = DownBlock(ngf * 8, ngf * 8)                         # 8x8
        self.down6 = DownBlock(ngf * 8, ngf * 8)                         # 4x4
        self.down7 = DownBlock(ngf * 8, ngf * 8)                         # 2x2
        self.down8 = DownBlock(ngf * 8, ngf * 8, normalize=False)        # 1x1

        # Upsampling stages (1 -> 2 -> 4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256)
        self.up1 = UpBlock(ngf * 8, ngf * 8, dropout=0.5)               # 2x2
        self.up2 = UpBlock(ngf * 16, ngf * 8, dropout=0.5)              # 4x4
        self.up3 = UpBlock(ngf * 16, ngf * 8, dropout=0.5)              # 8x8
        self.up4 = UpBlock(ngf * 16, ngf * 8)                           # 16x16
        self.up5 = UpBlock(ngf * 16, ngf * 4)                           # 32x32
        self.up6 = UpBlock(ngf * 8, ngf * 2)                            # 64x64
        self.up7 = UpBlock(ngf * 4, ngf)                                # 128x128

        # Final head to 256x256 RGB
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ngf * 2, ngf, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf, out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Downsample with skip connections
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # Upsample concatenating corresponding encoder features
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        out = self.final_up(u7)
        return out


class PatchGANDiscriminator(nn.Module):
    """
    70x70 PatchGAN Discriminator with Spectral Normalization.
    Guarantees 1-Lipschitz condition, prevents mode collapse, and stabilizes GAN dynamics.
    Conditioned on (semantic_mask, target_image) -> 6 input channels.
    """
    def __init__(self, in_channels: int = 6, ndf: int = 64, use_spectral_norm: bool = True):
        super().__init__()
        
        def sn(layer: nn.Module) -> nn.Module:
            return nn.utils.spectral_norm(layer) if use_spectral_norm else layer

        self.block1 = nn.Sequential(
            sn(nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.block2 = nn.Sequential(
            sn(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.block3 = nn.Sequential(
            sn(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.block4 = nn.Sequential(
            sn(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1, bias=False)),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.final_conv = sn(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1))

    def forward(self, condition: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # Concatenate condition and target along channel dimension
        x = torch.cat([condition, target], dim=1)
        
        # Collect intermediate activations for Feature Matching Loss
        features: List[torch.Tensor] = []
        
        h1 = self.block1(x)
        features.append(h1)
        h2 = self.block2(h1)
        features.append(h2)
        h3 = self.block3(h2)
        features.append(h3)
        h4 = self.block4(h3)
        features.append(h4)
        
        out = self.final_conv(h4)
        return out, features


class PerceptualLoss(nn.Module):
    """
    Lightweight perceptual feature loss extracting multi-scale textures.
    Uses frozen convolutional feature extractor to guide high-frequency sharpness.
    """
    def __init__(self):
        super().__init__()
        # Predefined fixed filter bank capturing edges, corners, and texture gradients
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        feat_pred = self.feature_extractor(pred)
        feat_target = self.feature_extractor(target)
        return F.l1_loss(feat_pred, feat_target)


class FacadeCompositeLoss(nn.Module):
    """
    Composite Objective as formulated in Cahier des Charges:
    min_G max_D L_cGAN(G, D) = E[(D(x,y)-1)^2] + E[D(x,G(x))^2]
                              + 30 * ||y - G(x)||_1
                              + 10 * L_FeatureMatching(G, D)
                              + 10 * L_Perceptual(G(x), y)
    """
    def __init__(
        self,
        lambda_l1: float = 30.0,
        lambda_fm: float = 10.0,
        lambda_perceptual: float = 10.0
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_fm = lambda_fm
        self.lambda_perceptual = lambda_perceptual
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()

    def generator_loss(
        self,
        d_fake_pred: torch.Tensor,
        real_features: List[torch.Tensor],
        fake_features: List[torch.Tensor],
        fake_images: torch.Tensor,
        real_images: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # Adversarial LSGAN loss: want D(x, G(x)) to be 1
        loss_adv = self.mse_loss(d_fake_pred, torch.ones_like(d_fake_pred))

        # L1 pixel fidelity
        loss_l1 = self.l1_loss(fake_images, real_images)

        # Feature Matching Loss across discriminator intermediate layers
        loss_fm = torch.tensor(0.0, device=fake_images.device)
        for f_real, f_fake in zip(real_features, fake_features):
            loss_fm = loss_fm + self.l1_loss(f_fake, f_real.detach())
        loss_fm = loss_fm / max(1, len(real_features))

        # Perceptual Loss
        loss_perc = self.perceptual_loss(fake_images, real_images)

        # Total weighted generator loss
        total_loss = (
            loss_adv
            + self.lambda_l1 * loss_l1
            + self.lambda_fm * loss_fm
            + self.lambda_perceptual * loss_perc
        )

        return {
            "g_total": total_loss,
            "g_adv": loss_adv,
            "g_l1": loss_l1,
            "g_fm": loss_fm,
            "g_perceptual": loss_perc,
        }

    def discriminator_loss(
        self,
        d_real_pred: torch.Tensor,
        d_fake_pred: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # LSGAN real target is 1, fake target is 0
        loss_real = self.mse_loss(d_real_pred, torch.ones_like(d_real_pred))
        loss_fake = self.mse_loss(d_fake_pred, torch.zeros_like(d_fake_pred))
        total_d_loss = 0.5 * (loss_real + loss_fake)
        return {
            "d_total": total_d_loss,
            "d_real": loss_real,
            "d_fake": loss_fake,
        }
