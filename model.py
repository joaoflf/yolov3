from turtle import forward

import torch
from torch import nn


class CNNBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, has_bn: bool, **kwargs
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not has_bn, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.has_bn = has_bn

    def forward(self, x) -> torch.Tensor:
        if self.has_bn:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, repeats: int = 1) -> None:
        super().__init__
        self.layers = nn.ModuleList()
        for i in range(repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(
                        channels, channels // 2, has_bn=True, kernel_size=1, padding=1
                    ),
                    CNNBlock(
                        channels // 2, channels, has_bn=True, kernel_size=3, padding=1
                    ),
                )
            ]
        self.repeats = repeats

    def forward(self, x) -> torch.Tensor:
        for i in range(self.repeats):
            x = x + self.layers(x)
        return x
