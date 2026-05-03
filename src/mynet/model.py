from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


def _build_resnet34(pretrained: bool) -> nn.Module:
    try:
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        return models.resnet34(weights=weights)
    except AttributeError:
        return models.resnet34(pretrained=pretrained)


class CornerResNet34(nn.Module):
    """ResNet34 backbone with a dynamic stride-4 heatmap head.

    Input:  [B, 3, H, W]
    Output: [B, 8, H_out, W_out] logits, aligned to the layer1 feature map.
    """

    def __init__(self, out_channels: int = 8, pretrained: bool = True) -> None:
        super().__init__()
        resnet = _build_resnet34(pretrained=pretrained)

        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.layer1 = resnet.layer1  # stride 4, channels 64
        self.layer2 = resnet.layer2  # stride 8, channels 128
        self.layer3 = resnet.layer3  # stride 16, channels 256
        self.layer4 = resnet.layer4  # stride 32, channels 512

        self.up4 = self._up_block(512, 256)
        self.up3 = self._up_block(256 + 256, 128)
        self.up2 = self._up_block(128 + 128, 64)
        self.refine = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1),
        )

    @staticmethod
    def _up_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _resize_and_apply(x: torch.Tensor, target: torch.Tensor, block: nn.Sequential) -> torch.Tensor:
        x = F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=False)
        # Keep the old Sequential layout so fixed-size checkpoints still load.
        for layer in list(block.children())[1:]:
            x = layer(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(x)
        c1 = self.layer1(x0)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        x = self._resize_and_apply(c4, c3, self.up4)
        x = self._resize_and_apply(torch.cat([x, c3], dim=1), c2, self.up3)
        x = self._resize_and_apply(torch.cat([x, c2], dim=1), c1, self.up2)
        logits = self.refine(torch.cat([x, c1], dim=1))
        return logits
