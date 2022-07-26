import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels: list,
        repeats:int = 1
    ) -> None:
        super(ResidualBlock, self).__init__
        for i in range(repeats):

