# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: Optional[int] = None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class BasicBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.relu(x + self.conv2(self.conv1(x)), inplace=True)


class FuseModule(nn.Module):
    def __init__(self, high_ch: int = 32, low_ch: int = 64):
        super().__init__()
        self.low_to_high = nn.Sequential(
            nn.Conv2d(low_ch, high_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(high_ch),
        )
        self.high_to_low = nn.Sequential(
            nn.Conv2d(high_ch, low_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(low_ch),
        )

    def forward(self, high, low):
        low_up = self.low_to_high(low)
        low_up = F.interpolate(low_up, size=high.shape[-2:], mode="bilinear", align_corners=False)
        high_down = self.high_to_low(high)
        if high_down.shape[-2:] != low.shape[-2:]:
            high_down = F.interpolate(high_down, size=low.shape[-2:], mode="bilinear", align_corners=False)
        high = F.relu(high + low_up, inplace=True)
        low = F.relu(low + high_down, inplace=True)
        return high, low


class HRStage(nn.Module):
    def __init__(self, high_ch: int = 32, low_ch: int = 64, blocks: int = 2):
        super().__init__()
        self.high_blocks = nn.Sequential(*[BasicBlock(high_ch) for _ in range(blocks)])
        self.low_blocks = nn.Sequential(*[BasicBlock(low_ch) for _ in range(blocks)])
        self.fuse = FuseModule(high_ch=high_ch, low_ch=low_ch)

    def forward(self, high, low):
        high = self.high_blocks(high)
        low = self.low_blocks(low)
        return self.fuse(high, low)


class HRNetLitePoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(3, 32, kernel_size=3, stride=2, padding=1),
            ConvBNAct(32, 32, kernel_size=3, stride=2, padding=1),
            BasicBlock(32),
            BasicBlock(32),
        )
        self.to_low = ConvBNAct(32, 64, kernel_size=3, stride=2, padding=1)
        self.stage1 = HRStage(32, 64, blocks=2)
        self.stage2 = HRStage(32, 64, blocks=2)
        self.stage3 = HRStage(32, 64, blocks=2)
        self.head_pre = nn.Sequential(
            ConvBNAct(96, 64, kernel_size=3, stride=1, padding=1),
            BasicBlock(64),
        )
        self.hm_head = nn.Sequential(
            ConvBNAct(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.off_head = nn.Sequential(
            ConvBNAct(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x):
        high = self.stem(x)
        low = self.to_low(high)
        high, low = self.stage1(high, low)
        high, low = self.stage2(high, low)
        high, low = self.stage3(high, low)
        low_up = F.interpolate(low, size=high.shape[-2:], mode="bilinear", align_corners=False)
        feat = torch.cat([high, low_up], dim=1)
        feat = self.head_pre(feat)
        return {"heatmap": self.hm_head(feat), "offset": self.off_head(feat)}


def make_hrnet_point():
    return HRNetLitePoint()


make_hrnet_lite_point = make_hrnet_point
