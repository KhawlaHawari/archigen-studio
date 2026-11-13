"""
Multi-Task UNet++ (Nested Dense Skip Pathways) with ResNet-34 Backbone
Designed for CubiCasa5K Floorplan Semantic Segmentation
Tête 1: Structures (5 classes: Bg, Mur Ext, Mur Int, Porte, Fenêtre)
Tête 2: Pièces (12 classes: Living, Bedroom, Kitchen, Bathroom, etc.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional


class ConvBlock(nn.Module):
    """Double convolution block with BatchNorm and ReLU"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class BasicResBlock(nn.Module):
    """Standard ResNet basic residual block"""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class ResNet34Encoder(nn.Module):
    """ResNet-34 5-level hierarchical feature extractor"""
    def __init__(self, in_channels: int = 3):
        super().__init__()
        # Level 0: 1/1 resolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Level 1: 1/2 resolution
        self.layer1 = nn.Sequential(
            BasicResBlock(64, 64, stride=2),
            BasicResBlock(64, 64),
            BasicResBlock(64, 64),
        )

        # Level 2: 1/4 resolution
        self.layer2 = nn.Sequential(
            BasicResBlock(64, 128, stride=2),
            BasicResBlock(128, 128),
            BasicResBlock(128, 128),
            BasicResBlock(128, 128),
        )

        # Level 3: 1/8 resolution
        self.layer3 = nn.Sequential(
            BasicResBlock(128, 256, stride=2),
            BasicResBlock(256, 256),
            BasicResBlock(256, 256),
            BasicResBlock(256, 256),
            BasicResBlock(256, 256),
            BasicResBlock(256, 256),
        )

        # Level 4: 1/16 resolution
        self.layer4 = nn.Sequential(
            BasicResBlock(256, 512, stride=2),
            BasicResBlock(512, 512),
            BasicResBlock(512, 512),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x0_0 = self.init_conv(x)      # [B, 64, H, W]
        x1_0 = self.layer1(x0_0)     # [B, 64, H/2, W/2]
        x2_0 = self.layer2(x1_0)     # [B, 128, H/4, W/4]
        x3_0 = self.layer3(x2_0)     # [B, 256, H/8, W/8]
        x4_0 = self.layer4(x3_0)     # [B, 512, H/16, W/16]
        return x0_0, x1_0, x2_0, x3_0, x4_0


class UNetPlusPlusMultiTask(nn.Module):
    """
    Nested UNet++ with multi-task dual heads for floorplan segmentation.
    - Head 1: Structural boundaries & openings (5 classes)
    - Head 2: Functional room zones (12 classes)
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_structure_classes: int = 5,
        num_room_classes: int = 12,
        base_channels: int = 64
    ):
        super().__init__()
        self.encoder = ResNet34Encoder(in_channels=in_channels)

        # Channel sizes at each level
        c0, c1, c2, c3, c4 = 64, 64, 128, 256, 512

        # Upsampling modules
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Level 1 nested nodes (j=1)
        self.conv0_1 = ConvBlock(c0 + c1, c0)
        self.conv1_1 = ConvBlock(c1 + c2, c1)
        self.conv2_1 = ConvBlock(c2 + c3, c2)
        self.conv3_1 = ConvBlock(c3 + c4, c3)

        # Level 2 nested nodes (j=2)
        self.conv0_2 = ConvBlock(c0 * 2 + c1, c0)
        self.conv1_2 = ConvBlock(c1 * 2 + c2, c1)
        self.conv2_2 = ConvBlock(c2 * 2 + c3, c2)

        # Level 3 nested nodes (j=3)
        self.conv0_3 = ConvBlock(c0 * 3 + c1, c0)
        self.conv1_3 = ConvBlock(c1 * 3 + c2, c1)

        # Level 4 nested node (j=4)
        self.conv0_4 = ConvBlock(c0 * 4 + c1, c0)

        # Multi-task heads
        self.structure_head = nn.Sequential(
            nn.Conv2d(c0, c0 // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c0 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c0 // 2, num_structure_classes, kernel_size=1)
        )

        self.room_head = nn.Sequential(
            nn.Conv2d(c0, c0 // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c0 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c0 // 2, num_room_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encoder forward pass
        x0_0, x1_0, x2_0, x3_0, x4_0 = self.encoder(x)

        # Column j=1
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))

        # Column j=2
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], dim=1))

        # Column j=3
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], dim=1))

        # Column j=4
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], dim=1))

        # Predict logits for both tasks
        struct_logits = self.structure_head(x0_4)
        room_logits = self.room_head(x0_4)

        return struct_logits, room_logits


class SoftDiceLoss(nn.Module):
    """Multiclass differentiable Soft Dice Loss. Pixels equal to ignore_index are excluded."""
    def __init__(self, smooth: float = 1e-5, ignore_index: int = 255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, C, H, W], targets: [B, H, W]
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        valid = (targets != self.ignore_index).unsqueeze(1).float()
        safe_targets = torch.where(targets == self.ignore_index, torch.zeros_like(targets), targets)
        targets_one_hot = F.one_hot(safe_targets.clamp(0, num_classes - 1), num_classes).permute(0, 3, 1, 2).float()
        probs = probs * valid
        targets_one_hot = targets_one_hot * valid

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dim=dims)
        cardinality = torch.sum(probs + targets_one_hot, dim=dims)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - torch.mean(dice_score)


class FloorplanMultiTaskLoss(nn.Module):
    """
    Composite Multi-Task Loss as defined in Cahier des Charges:
    L_floorplan = L_Struct(WCE + Dice) + 0.5 * L_Room(CE + Dice)
    With structure weights: Bg=0.5, ExtWall=4.0, IntWall=4.0, Door=25.0, Window=10.0
    """
    def __init__(self, struct_weights: Optional[List[float]] = None):
        super().__init__()
        if struct_weights is None:
            struct_weights = [0.5, 4.0, 4.0, 25.0, 10.0]
        
        self.register_buffer("struct_weights", torch.tensor(struct_weights, dtype=torch.float32))
        # 255 = unknown/unmapped label (e.g. CubiCasa "Space Undefined"): no gradient.
        self.struct_ce = nn.CrossEntropyLoss(weight=self.struct_weights, ignore_index=255)
        self.room_ce = nn.CrossEntropyLoss(ignore_index=255)
        self.dice = SoftDiceLoss()

    def forward(
        self,
        struct_logits: torch.Tensor,
        struct_targets: torch.Tensor,
        room_logits: torch.Tensor,
        room_targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        loss_struct_ce = self.struct_ce(struct_logits, struct_targets)
        loss_struct_dice = self.dice(struct_logits, struct_targets)
        loss_struct = loss_struct_ce + loss_struct_dice

        loss_room_ce = self.room_ce(room_logits, room_targets)
        loss_room_dice = self.dice(room_logits, room_targets)
        loss_room = loss_room_ce + loss_room_dice

        total_loss = loss_struct + 0.5 * loss_room

        return {
            "total_loss": total_loss,
            "loss_struct": loss_struct,
            "loss_room": loss_room,
            "loss_struct_ce": loss_struct_ce,
            "loss_struct_dice": loss_struct_dice,
            "loss_room_ce": loss_room_ce,
            "loss_room_dice": loss_room_dice,
        }
