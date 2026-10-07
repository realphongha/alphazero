"""A unified *Bot* interface: either a trained model (neural) or pure MCTS.

Used by the GUI (bot vs player / bot vs bot) and the text CLI.  A bot owns its
own MCTS + policy provider, so the two sides of a bot-vs-bot game can be
*completely different* (different model files, different sim budgets, or one
neural and one pure-MCTS).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .games.base import Game
from .inference import NeuralPolicyProvider, RandomPolicyProvider, OnnxPolicyProvider
from .mcts import AlphaZeroMCTS, MCTSConfig

# NOTE: torch-dependent helpers (load_model, get_device, resolve_amp_dtype) are
# imported lazily inside the 'model' branch below, so a torch-free ONNX runtime
# (e.g. an Apple M1 Mac with only onnxruntime) can build 'onnx'/'random' bots.


@dataclass
class BotConfig:
    name: str = "bot"
    kind: str = "random"            # 'model' (torch .pt) | 'onnx' (ONNX .onnx) | 'random'
    model_path: Optional[str] = None  # .pt for kind=model, .onnx for kind=onnx
    num_sims: int = 100
    batch_size: int = 16
    c_puct: float = 1.5
    num_playouts: int = 16
    device: str = "auto"
    amp: str = "none"               # none | bf16 | fp16
    seed: Optional[int] = None


class Bot:
    def __init__(self, cfg: BotConfig, game: Game) -> None:
        self.cfg = cfg
        self.game = game
        if cfg.kind == "model":
            from .checkpoint import load_model
            from .models.nn import get_device, resolve_amp_dtype
            if not cfg.model_path:
                raise ValueError(f"bot '{cfg.name}' kind=model needs model_path")
            device = get_device(cfg.device)
            self.device = device
            model = load_model(cfg.model_path, device)
            amp_dtype = resolve_amp_dtype(device, cfg.amp)
            self.provider = NeuralPolicyProvider(model, amp_dtype)
        elif cfg.kind == "onnx":
            if not cfg.model_path:
                raise ValueError(f"bot '{cfg.name}' kind=onnx needs model_path (.onnx)")
            self.device = "cpu"
            self.provider = OnnxPolicyProvider(cfg.model_path, game=game)
        elif cfg.kind == "random":
            self.device = "cpu"
            self.provider = RandomPolicyProvider(game, cfg.num_playouts)
        else:
            raise ValueError(f"unknown bot kind '{cfg.kind}'")
        self.mcts = AlphaZeroMCTS(
            game, self.provider,
            MCTSConfig(num_simulations=cfg.num_sims, c_puct=cfg.c_puct,
                       batch_size=cfg.batch_size, seed=cfg.seed),
        )

    def best_action(self, state: Game) -> int:
        return self.mcts.best_action(state)

    def describe(self) -> str:
        return f"{self.cfg.name} [{self.provider.describe()}, sims={self.cfg.num_sims}]"

    def __repr__(self) -> str:
        return self.describe()
