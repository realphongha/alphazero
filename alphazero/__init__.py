"""alphazero — a compact, optimised AlphaZero library.

Public surface:

    from alphazero import (
        MNKGame, MCTSConfig, AlphaZeroMCTS,
        build_model, make_provider, Trainer, TrainConfig,
        AlphaZeroAgent, run_selfplay, get_preset,
    )

Modules
--------
* :mod:`alphazero.games`     — game abstraction, bitboards, MNK + presets
* :mod:`alphazero.mcts`      — MCTS (abstract) + batched AlphaZero MCTS
* :mod:`alphazero.inference` — policy/value providers (neural or pure-MCTS)
* :mod:`alphazero.models`    — ResNet & ViT backbones + factory
* :mod:`alphazero.training`  — data, warmup-cosine scheduler, AMP trainer
* :mod:`alphazero.selfplay`  — supervised data generation
* :mod:`alphazero.alphazero` — the AlphaZero loop (abstract + agent)
* :mod:`alphazero.profiling` — per-phase latency breakdown
* :mod:`alphazero.gui`       — pygame GUI (player vs bot, bot vs bot)
* :mod:`alphazero.cli`       — ``python -m alphazero`` entry point
"""
from .games import Game, MNKGame, PRESETS, get_preset, make_game
from .inference import (
    PolicyProvider, NeuralPolicyProvider, RandomPolicyProvider, make_provider,
)
from .mcts import AlphaZeroMCTS, MCTSConfig
from .models import build_model
from .models.nn import ZeroNet, get_device
from .selfplay import SelfplayConfig, run_selfplay
from .training import Trainer, TrainConfig, Sample, WarmupCosineScheduler
from .alphazero import AlphaZero, AlphaZeroAgent
from .checkpoint import ModelMeta, save_model, load_model
from .profiling import prof

__version__ = "0.1.0"

__all__ = [
    "Game", "MNKGame", "PRESETS", "get_preset", "make_game",
    "PolicyProvider", "NeuralPolicyProvider", "RandomPolicyProvider", "make_provider",
    "AlphaZeroMCTS", "MCTSConfig",
    "build_model", "ZeroNet", "get_device",
    "SelfplayConfig", "run_selfplay",
    "Trainer", "TrainConfig", "Sample", "WarmupCosineScheduler",
    "AlphaZero", "AlphaZeroAgent",
    "ModelMeta", "save_model", "load_model",
    "prof",
    "__version__",
]
