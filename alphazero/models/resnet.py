"""ResNet-style AlphaZero backbone (the original Silver et al. layout)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .nn import ZeroNet


class ResBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.net(x) + x).relu_()


class ResNetNet(ZeroNet):
    def __init__(
        self,
        num_moves: int,
        device: torch.device,
        channels: int = 128,
        blocks: int = 5,
        value_fc: int = 256,
        memory_format: bool = True,
    ) -> None:
        super().__init__(num_moves, device)
        self.input = nn.Sequential(
            nn.Conv2d(self.obs_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.resblocks = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        # one 1x1 logit per cell: (B, ch, n, n) -> (B, 1, n, n) -> (B, n*n)
        self.policy_conv = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(channels, value_fc),
            nn.ReLU(inplace=True),
            nn.Linear(value_fc, 1),
        )
        # channels_last (NHWC) is faster for convs on NVIDIA GPUs.
        self._finalize(memory_format=memory_format)

    def forward(self, obs: torch.Tensor):
        x = self.input(obs)
        x = self.resblocks(x)
        pol = self.policy_conv(x).flatten(1)
        pooled = F.adaptive_avg_pool2d(self.value_conv(x), 1).flatten(1)
        val = torch.tanh(self.value_fc(pooled))
        return pol, val
