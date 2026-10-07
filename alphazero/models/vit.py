"""Vision-Transformer-style AlphaZero backbone.

The board is treated as a grid of tokens (one per cell by default, or a coarse
``patch`` x ``patch`` block when ``patch > 1``), embedded with a stride-equal-to-
patch conv, given a learned positional code, and passed through standard
pre-norm transformer blocks.  The policy head is a per-token linear (one logit
per cell); the value head is a global average pool -> MLP -> tanh.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .nn import ZeroNet


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViTNet(ZeroNet):
    def __init__(
        self,
        num_moves: int,
        device: torch.device,
        board_dim: int,
        dim: int = 128,
        heads: int = 4,
        depth: int = 4,
        patch: int = 1,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        value_fc: int = 256,
    ) -> None:
        super().__init__(num_moves, device)
        self.board_dim = board_dim
        self.patch = patch
        assert board_dim % patch == 0, "board_dim must be divisible by patch"
        self.grid = board_dim // patch
        self.num_tokens = self.grid * self.grid

        self.patch_embed = nn.Conv2d(self.obs_channels, dim, kernel_size=patch, stride=patch)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.Sequential(*[_Block(dim, heads, dim * mlp_ratio, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.policy_head = nn.Linear(dim, 1)
        self.value_head = nn.Sequential(
            nn.Linear(dim, value_fc), nn.ReLU(inplace=True), nn.Linear(value_fc, 1),
        )
        # scale init like standard ViT
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.trunc_normal_(p, std=0.02)
        self._finalize(memory_format=False)

    def forward(self, obs: torch.Tensor):
        B = obs.shape[0]
        x = self.patch_embed(obs)                       # (B, dim, grid, grid)
        x = x.flatten(2).transpose(1, 2)               # (B, tokens, dim)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        pol = self.policy_head(x).squeeze(-1)          # (B, tokens)
        if self.grid != self.board_dim:
            pol = pol.reshape(B, self.grid, self.grid)
            pol = F.interpolate(pol.unsqueeze(1), size=(self.board_dim, self.board_dim),
                                mode="nearest").squeeze(1)
        pol = pol.flatten(1)                            # (B, n*n)
        val = torch.tanh(self.value_head(x.mean(dim=1)))  # (B, 1)
        return pol, val
