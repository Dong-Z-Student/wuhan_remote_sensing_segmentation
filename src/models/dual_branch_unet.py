from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.models.blocks import ConvBlock, DownBlock, UpBlock, OutConv


class EncoderBranch(nn.Module):
    """
    单个模态的 U-Net encoder 分支。
    输入:
        x: [B, in_channels, H, W]
    输出:
        x1, x2, x3, x4, x5
        分别对应 1/1, 1/2, 1/4, 1/8, 1/16 尺度特征。
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
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

    def forward(self, x: torch.Tensor):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        return x1, x2, x3, x4, x5


class DualBranchUNet(nn.Module):
    """
    多尺度 concat 融合的双分支 U-Net。
    SAR 分支:
        shape: [B, 2, H, W]
    Optical 分支:
        shape: [B, 6, H, W]
    融合方式:
        SAR encoder 和 Optical encoder 在每个尺度上分别提取特征；
        对同尺度特征做 concat；
        然后送入共享 decoder。
    输出:
        logits: [B, num_classes, H, W]
    """

    def __init__(
        self,
        sar_in_channels: int = 2,
        optical_in_channels: int = 6,
        num_classes: int = 8,
        base_channels: int = 32,
        bilinear: bool = True,
    ) -> None:
        super().__init__()

        self.sar_encoder = EncoderBranch(
            in_channels=sar_in_channels,
            base_channels=base_channels,
        )

        self.optical_encoder = EncoderBranch(
            in_channels=optical_in_channels,
            base_channels=base_channels,
        )

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        # 多尺度融合后，各层通道数翻倍
        f1 = c1 * 2
        f2 = c2 * 2
        f3 = c3 * 2
        f4 = c4 * 2
        f5 = c5 * 2

        # decoder:
        # up1 输入为 upsample(f5) concat f4，因此通道数为 f5 + f4
        self.up1 = UpBlock(f5 + f4, c4, bilinear=bilinear)
        self.up2 = UpBlock(c4 + f3, c3, bilinear=bilinear)
        self.up3 = UpBlock(c3 + f2, c2, bilinear=bilinear)
        self.up4 = UpBlock(c2 + f1, c1, bilinear=bilinear)

        self.outc = OutConv(c1, num_classes)

    def forward(
        self,
        sar: torch.Tensor,
        optical: torch.Tensor,
    ) -> torch.Tensor:
        s1, s2, s3, s4, s5 = self.sar_encoder(sar)
        o1, o2, o3, o4, o5 = self.optical_encoder(optical)

        # 多尺度 concat 融合
        f1 = torch.cat([s1, o1], dim=1)
        f2 = torch.cat([s2, o2], dim=1)
        f3 = torch.cat([s3, o3], dim=1)
        f4 = torch.cat([s4, o4], dim=1)
        f5 = torch.cat([s5, o5], dim=1)

        x = self.up1(f5, f4)
        x = self.up2(x, f3)
        x = self.up3(x, f2)
        x = self.up4(x, f1)

        logits = self.outc(x)
        return logits


def test_model() -> None:
    model = DualBranchUNet(
        sar_in_channels=2,
        optical_in_channels=6,
        num_classes=8,
        base_channels=32,
        bilinear=True,
    )

    sar = torch.randn(2, 2, 256, 256)
    optical = torch.randn(2, 6, 256, 256)

    y = model(sar, optical)

    print("DualBranchUNet 测试成功")
    print(f"sar input shape    : {sar.shape}")
    print(f"optical input shape: {optical.shape}")
    print(f"output shape       : {y.shape}")


if __name__ == "__main__":
    test_model()