"""The AlphaZero loop, as an abstract base + a concrete agent.

``AlphaZero`` (ABC) defines the two pluggable halves — ``selfplay`` and
``train_on`` — and provides the fixed point iteration ``run`` (generate data ->
train -> repeat) as a template method.  ``AlphaZeroAgent`` wires a real
``PolicyProvider``, MCTS and trainer to implement the halves.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List, Optional

import numpy as np
import torch

from .games.base import Game
from .inference import PolicyProvider
from .mcts import AlphaZeroMCTS, MCTSConfig
from .profiling import prof
from .selfplay import SelfplayConfig, run_selfplay
from .training.data import Sample
from .training.train import TrainConfig, Trainer


class AlphaZero(ABC):
    """Abstract AlphaZero: fixed iteration, pluggable data-gen + training."""

    @abstractmethod
    def selfplay(self, num_games: int, **opts) -> List[Sample]:
        """Generate supervised data with the *current* policy."""

    @abstractmethod
    def train_on(self, samples: List[Sample], **opts) -> dict:
        """Update the policy from a batch of supervised data; return metrics."""

    def run(
        self,
        iterations: int,
        games_per_iter: int,
        epochs: int = 1,
        keep_history: bool = True,
        progress_cb: Optional[Callable[[int, dict, int], None]] = None,
    ) -> List[Sample]:
        history: List[Sample] = []
        for it in range(iterations):
            with prof.track("alphazero.selfplay"):
                samples = self.selfplay(games_per_iter)
            if keep_history:
                history += samples
            with prof.track("alphazero.train"):
                metrics = self.train_on(history, epochs=epochs)
            if progress_cb is not None:
                progress_cb(it, metrics, len(history))
        return history


class AlphaZeroAgent(AlphaZero):
    """Concrete AlphaZero over a neural provider (or pure-MCTS, no training)."""

    def __init__(
        self,
        game: Game,
        provider: PolicyProvider,
        mcts_cfg: Optional[MCTSConfig] = None,
        train_cfg: Optional[TrainConfig] = None,
        sp_cfg: Optional[SelfplayConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.game = game
        self.make_game = game.copy
        self.provider = provider
        self.mcts = AlphaZeroMCTS(game, provider, mcts_cfg or MCTSConfig())
        self.sp_cfg = sp_cfg or SelfplayConfig()
        self.device = device or (provider.model.device if hasattr(provider, "model") and provider.model is not None else torch.device("cpu"))
        self.model = getattr(provider, "model", None)
        self.trainer = Trainer(self.model, self.device, train_cfg or TrainConfig()) if self.model is not None else None
        self.history: List[Sample] = []

    def selfplay(self, num_games: int, **opts) -> List[Sample]:
        return run_selfplay(
            self.make_game, self.mcts, num_games,
            cfg=opts.get("sp_cfg", self.sp_cfg),
            seed=opts.get("seed"),
        )

    def train_on(self, samples: List[Sample], **opts) -> dict:
        if self.trainer is None:
            return {"note": "pure-MCTS provider has no model to train"}
        return self.trainer.fit(samples, epochs=opts.get("epochs", 1))
