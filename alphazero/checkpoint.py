"""Save / load a trained ZeroNet together with the metadata needed to rebuild it.

A checkpoint is a single ``.pt`` file holding ``{state_dict, meta}`` where
``meta`` records the game shape + backbone + device so :func:`load_model` can
reconstruct the exact architecture.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import torch

from .games.base import Game
from .games.mnk import MNKGame
from .models.factory import build_model
from .models.nn import ZeroNet, get_device


@dataclass
class ModelMeta:
    backbone: str = "resnet"
    n: int = 15
    m: int = 5
    num_moves: int = 225
    arch: dict = None

    def to_dict(self):
        d = asdict(self)
        d["arch"] = self.arch or {}
        return d


def save_model(model: ZeroNet, path: str, meta: ModelMeta) -> str:
    torch.save({"state_dict": model.state_dict(), "meta": meta.to_dict()}, path)
    return path


def load_model(path: str, device: Optional[torch.device] = None,
               preference: str = "auto") -> ZeroNet:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    meta = payload["meta"]
    game = MNKGame(n=meta["n"], m=meta["m"])
    dev = device or get_device(preference)
    model = build_model(game, backbone=meta["backbone"], device=dev, cfg=meta.get("arch") or {})
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
