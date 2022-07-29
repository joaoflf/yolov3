from re import I
from tkinter.ttk import Scale
from turtle import forward
from typing import Callable

import torch
from torch import nn


class CNNBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, has_bn: bool = True, **kwargs
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
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]

    def forward(self, x) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(x)
        return x


class ScalePrediction(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.exit_point_output = None
        self.exit_point = CNNBlock(
            in_channels // 2, in_channels, kernel_size=3, padding=1
        )
        self.exit_point.register_forward_hook(self.store_exit_point_output())

        self.layers = nn.Sequential(
            CNNBlock(in_channels, in_channels // 2, kernel_size=1),
            CNNBlock(in_channels // 2, in_channels, kernel_size=3, padding=1),
            CNNBlock(in_channels, in_channels // 2, kernel_size=1),
            self.exit_point,
            CNNBlock(in_channels, in_channels // 2, kernel_size=1),
            CNNBlock(in_channels // 2, in_channels, kernel_size=3, padding=1),
            CNNBlock(in_channels, 3 * (num_classes + 5), has_bn=False, kernel_size=1),
        )
        self.num_classes = num_classes

    def store_exit_point_output(self) -> Callable:
        def hook(module, input, output):
            self.exit_point_output = output

        return hook

    def forward(self, x) -> torch.Tensor:
        x = (
            self.layers(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )
        return x


class YoloV3(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.concat_outputs = {}

        self.first_concat_point = CNNBlock(256, 512, kernel_size=3, stride=2, padding=1)
        self.second_concat_point = CNNBlock(
            128, 256, kernel_size=3, stride=2, padding=1
        )

        self.first_concat_point.register_forward_hook(
            self.store_output_for_concat("first")
        )
        self.second_concat_point.register_forward_hook(
            self.store_output_for_concat("second")
        )

        self.darknet53 = nn.ModuleList()
        self.darknet53 += [
            CNNBlock(3, 32, kernel_size=3, padding=1),
            CNNBlock(32, 64, kernel_size=3, stride=2, padding=1),
            ResidualBlock(64),
            CNNBlock(64, 128, kernel_size=3, stride=2, padding=1),
            ResidualBlock(128, 2),
            self.second_concat_point,
            ResidualBlock(256, 8),
            self.first_concat_point,
            ResidualBlock(512, 8),
            CNNBlock(512, 1024, kernel_size=3, stride=2, padding=1),
            ResidualBlock(1024, 4),
        ]

        self.first_prediction = ScalePrediction(1024, num_classes)
        self.first_prediction.exit_point_output = (
            None  # 'have to init to access in forward'
        )
        self.first_upsample = nn.Sequential(
            CNNBlock(1024, 512, kernel_size=1, stride=1), nn.Upsample(scale_factor=2)
        )
        self.second_prediction = ScalePrediction(1024, num_classes)
        self.second_prediction.exit_point_output = None
        self.second_upsample = nn.Sequential(
            CNNBlock(1024, 256, kernel_size=1, stride=1), nn.Upsample(scale_factor=2)
        )
        self.third_prediction = ScalePrediction(512, num_classes)

    def store_output_for_concat(self, concat_point_id: str) -> Callable:
        def hook(module, input, output):
            self.concat_outputs[concat_point_id] = output

        return hook

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for layer in self.darknet53:
            x = layer(x)

        y0 = self.first_prediction(x)

        first_upsample = self.first_upsample(self.first_prediction.exit_point_output)
        y1 = self.second_prediction(
            torch.cat([first_upsample, self.concat_outputs["first"]], dim=1)
        )
        second_upsample = self.second_upsample(self.second_prediction.exit_point_output)
        y2 = self.third_prediction(
            torch.cat([second_upsample, self.concat_outputs["second"]], dim=1)
        )

        return y0, y1, y2
