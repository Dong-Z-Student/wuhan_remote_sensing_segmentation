from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.models.blocks import ConvBlock, DownBlock, UpBlock, OutConv


class SARUNet(nn.Module):
    """
    纯 SAR U-Net。
    输入：
        x: [B, in_channels, H, W]
    输出：
        logits: [B, num_classes, H, W]
    """

    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 8,
        base_channels: int = 32,
        bilinear: bool = True,
    ) -> None:
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        self.inc = ConvBlock(in_channels, c1)
        self.down1 = DownBlock(c1, c2)
        self.down2 = DownBlock(c2, c3)
        self.down3 = DownBlock(c3, c4)
        self.down4 = DownBlock(c4, c5)

        self.up1 = UpBlock(c5 + c4, c4, bilinear=bilinear)
        self.up2 = UpBlock(c4 + c3, c3, bilinear=bilinear)
        self.up3 = UpBlock(c3 + c2, c2, bilinear=bilinear)
        self.up4 = UpBlock(c2 + c1, c1, bilinear=bilinear)

        self.outc = OutConv(c1, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)       # [B, c1, H, W]
        x2 = self.down1(x1)    # [B, c2, H/2, W/2]
        x3 = self.down2(x2)    # [B, c3, H/4, W/4]
        x4 = self.down3(x3)    # [B, c4, H/8, W/8]
        x5 = self.down4(x4)    # [B, c5, H/16, W/16]

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits


def test_model() -> None:
    model = SARUNet(
        in_channels=2,
        num_classes=8,
        base_channels=32,
        bilinear=True,
    )

    x = torch.randn(2, 2, 256, 256)
    y = model(x)

    print("SARUNet 测试成功")
    print(f"input shape : {x.shape}")
    print(f"output shape: {y.shape}")


if __name__ == "__main__":
    test_model()