# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Conv-BN-ReLU-Conv-BN-ReLU; nazwy modułów zgodne z checkpointem unet-point.pt."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """MaxPool + DoubleConv; nazwy modułów zgodne z checkpointem: downX.conv.block.*."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Bilinear upsample + concat skip + DoubleConv; bez parametrów w upsamplingu."""
    def __init__(self, in_channels_concat: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv(in_channels_concat, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetPoint(nn.Module):
    """
    Lekki U-Net do detekcji punktów.

    Architektura dopasowana do checkpointa unet-point.pt:
    - inc: 3 -> 16
    - down1: 16 -> 32
    - down2: 32 -> 64
    - down3: 64 -> 128
    - down4: 128 -> 256
    - up1: concat 128 + 256 = 384 -> 128
    - up2: concat 64 + 128 = 192 -> 64
    - hm_head: 64 -> 64 -> 1
    - off_head: 64 -> 64 -> 2

    Dla wejścia 1x3x256x512 wyjścia mają rozmiar:
    heatmap: 1x1x64x128
    offset:  1x2x64x128
    czyli down_ratio = 4.
    """
    def __init__(self):
        super().__init__()
        self.inc = DoubleConv(3, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.down4 = Down(128, 256)

        self.up1 = Up(128 + 256, 128)
        self.up2 = Up(64 + 128, 64)

        self.hm_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0, bias=True),
        )
        self.off_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1, padding=0, bias=True),
        )

    def forward(self, x: torch.Tensor):
        x1 = self.inc(x)       # 1/1, 16 ch
        x2 = self.down1(x1)    # 1/2, 32 ch
        x3 = self.down2(x2)    # 1/4, 64 ch
        x4 = self.down3(x3)    # 1/8, 128 ch
        x5 = self.down4(x4)    # 1/16, 256 ch

        x = self.up1(x5, x4)   # 1/8, 128 ch
        x = self.up2(x, x3)    # 1/4, 64 ch

        hm = self.hm_head(x)
        off = self.off_head(x)
        return {"heatmap": hm, "offset": off}


def make_unet_point():
    return UNetPoint()


make_unet_point_model = make_unet_point
