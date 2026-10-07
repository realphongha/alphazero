"""Model factory: build a ResNet or ViT ZeroNet from a name + config dict."""
from __future__ import annotations

from typing import Dict

import torch

from ..games.base import Game
from .nn import ZeroNet
from .resnet import ResNetNet
from .vit import ViTNet


def build_model(
    game: Game,
    backbone: str = "resnet",
    device: torch.device = torch.device("cpu"),
    cfg: Dict | None = None,
) -> ZeroNet:
    """Create a ZeroNet sized to ``game``.

    ``backbone`` in {'resnet', 'vit'}.  ``cfg`` overrides architecture hyper
    parameters (channels/blocks for resnet; dim/heads/depth/patch for vit).
    """
    cfg = dict(cfg or {})
    num_moves = game.num_moves()
    if backbone == "resnet":
        return ResNetNet(
            num_moves=num_moves,
            device=device,
            channels=cfg.get("channels", 128),
            blocks=cfg.get("blocks", 5),
            value_fc=cfg.get("value_fc", 256),
            memory_format=cfg.get("memory_format", True),
        )
    if backbone == "vit":
        n = int(round(num_moves ** 0.5))
        assert n * n == num_moves, "ViT needs a square board"
        return ViTNet(
            num_moves=num_moves,
            device=device,
            board_dim=n,
            dim=cfg.get("dim", 128),
            heads=cfg.get("heads", 4),
            depth=cfg.get("depth", 4),
            patch=cfg.get("patch", 1),
            mlp_ratio=cfg.get("mlp_ratio", 4),
            dropout=cfg.get("dropout", 0.0),
            value_fc=cfg.get("value_fc", 256),
        )
    raise ValueError(f"unknown backbone '{backbone}' (use 'resnet' or 'vit')")
