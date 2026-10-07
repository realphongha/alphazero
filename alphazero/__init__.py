"""alphazero -- a compact, optimised AlphaZero library.

The public API is re-exported here.  Torch-heavy symbols (models, training,
checkpoint) are loaded *lazily* via ``__getattr__`` (PEP 562), so importing the
package -- or running the torch-free inference path (pure-MCTS / ONNX bots, MCTS,
games, GUI) -- does **not** require PyTorch.  This is what lets the game + ONNX
inference run on a machine (e.g. an Apple M1 Mac) with only ``onnxruntime``.

Public surface:

    from alphazero import (
        MNKGame, MCTSConfig, AlphaZeroMCTS,
        build_model, make_provider, Trainer, TrainConfig,
        AlphaZeroAgent, run_selfplay, get_preset,
    )

Modules
--------
* :mod:`alphazero.games`     -- game abstraction, bitboards, MNK + presets
* :mod:`alphazero.mcts`      -- MCTS (abstract) + batched AlphaZero MCTS
* :mod:`alphazero.inference` -- policy/value providers (neural / onnx / pure-MCTS)
* :mod:`alphazero.models`    -- ResNet & ViT backbones + factory   (torch)
* :mod:`alphazero.export`    -- export a .pt model to ONNX          (torch)
* :mod:`alphazero.training`  -- data, warmup-cosine scheduler, AMP trainer (torch)
* :mod:`alphazero.selfplay`  -- supervised data generation
* :mod:`alphazero.alphazero` -- the AlphaZero loop (abstract + agent)  (torch)
* :mod:`alphazero.profiling` -- per-phase latency breakdown
* :mod:`alphazero.gui`       -- pygame GUI (player vs bot, bot vs bot)
* :mod:`alphazero.cli`       -- ``python -m alphazero`` entry point
"""
from __future__ import annotations

import importlib

# -- torch-free core (safe to import without PyTorch) -------------------------
from .games import Game, MNKGame, PRESETS, get_preset, make_game
from .inference import (
    PolicyProvider, NeuralPolicyProvider, RandomPolicyProvider,
    OnnxPolicyProvider, make_provider,
)
from .mcts import AlphaZeroMCTS, MCTSConfig
from .selfplay import SelfplayConfig, run_selfplay
from .profiling import prof

__version__ = "0.1.0"

# -- torch-dependent symbols: lazy (imported on first access) -----------------
_LAZY = {
    "build_model": (".models", "build_model"),
    "ZeroNet": (".models.nn", "ZeroNet"),
    "get_device": (".models.nn", "get_device"),
    "Trainer": (".training", "Trainer"),
    "TrainConfig": (".training", "TrainConfig"),
    "Sample": (".training", "Sample"),
    "WarmupCosineScheduler": (".training", "WarmupCosineScheduler"),
    "AlphaZero": (".alphazero", "AlphaZero"),
    "AlphaZeroAgent": (".alphazero", "AlphaZeroAgent"),
    "ModelMeta": (".checkpoint", "ModelMeta"),
    "save_model": (".checkpoint", "save_model"),
    "load_model": (".checkpoint", "load_model"),
    "export_to_onnx": (".export", "export_to_onnx"),
}


def __getattr__(name):
    if name in _LAZY:
        module_name, attr = _LAZY[name]
        module = importlib.import_module(module_name, __name__)
        return getattr(module, attr)
    raise AttributeError(f"module 'alphazero' has no attribute {name!r}")


__all__ = [
    "Game", "MNKGame", "PRESETS", "get_preset", "make_game",
    "PolicyProvider", "NeuralPolicyProvider", "RandomPolicyProvider",
    "OnnxPolicyProvider", "make_provider",
    "AlphaZeroMCTS", "MCTSConfig",
    "build_model", "ZeroNet", "get_device",
    "SelfplayConfig", "run_selfplay",
    "Trainer", "TrainConfig", "Sample", "WarmupCosineScheduler",
    "AlphaZero", "AlphaZeroAgent",
    "ModelMeta", "save_model", "load_model",
    "export_to_onnx",
    "prof", "__version__",
]
