# -*- coding: utf-8 -*-
"""
Factory dla modeli oznakowania poziomego:
- linknet-linie.pth -> segmentation_models_pytorch.Linknet + mobilenet_v2
- unet-linie.pth    -> segmentation_models_pytorch.Unet + mobilenet_v2
"""

from __future__ import annotations

import segmentation_models_pytorch as smp


def make_ground_map_linknet_mobilenetv2():
    return smp.Linknet(
        encoder_name="mobilenet_v2",
        encoder_weights=None,
        in_channels=3,
        classes=3,
    )


def make_ground_map_unet_mobilenetv2():
    return smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights=None,
        in_channels=3,
        classes=3,
    )


# Aliasy awaryjne, żeby działały też starsze komendy
make_linknet_mobilenetv2 = make_ground_map_linknet_mobilenetv2
make_unet_mobilenetv2 = make_ground_map_unet_mobilenetv2
make_linknet = make_ground_map_linknet_mobilenetv2
make_unet = make_ground_map_unet_mobilenetv2
