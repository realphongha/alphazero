"""Abstract game interface shared by MCTS, AlphaZero, self-play and training.

A ``Game`` is an *immutable* state: ``apply`` returns a fresh instance.
This keeps MCTS node book-keeping simple (a node just holds the state it
reached) and makes states cheap to copy, hash and share.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Game(ABC):
    """Minimal contract a game must satisfy to be playable by AlphaZero."""

    # -- identity / shape -------------------------------------------------
    @abstractmethod
    def num_moves(self) -> int:
        """Number of legal move *slots* (the action space size)."""

    @abstractmethod
    def board_size(self) -> int:
        """Alias of :meth:`num_moves` kept for model ergonomics."""

    @abstractmethod
    def observation(self) -> np.ndarray:
        """Model input tensor, shape ``(C, H, W)`` float32 (batchable)."""

    # -- play -------------------------------------------------------------
    @abstractmethod
    def current_player(self) -> int:
        """Player to move (1 or 2)."""

    @abstractmethod
    def legal_moves(self) -> np.ndarray:
        """1-D int array of legal move indices."""

    @abstractmethod
    def is_move_legal(self, move: int) -> bool:
        """Whether ``move`` is currently legal."""

    @abstractmethod
    def apply(self, move: int) -> "Game":
        """Return a *new* state with ``move`` played."""

    @abstractmethod
    def copy(self) -> "Game":
        """Return an independent copy of this state."""

    # -- outcome ----------------------------------------------------------
    @abstractmethod
    def is_terminal(self) -> bool:
        """True if the game is over."""

    @abstractmethod
    def winner(self) -> int:
        """Global winner: ``1`` or ``2`` when decided, ``0`` otherwise."""

    def reward(self, player: int) -> float:
        """Outcome from ``player``'s perspective: +1 / -1 / 0 (draw)."""
        w = self.winner()
        if w == 0:
            return 0.0
        return 1.0 if w == player else -1.0

    # -- introspection ----------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(player={self.current_player()})"
