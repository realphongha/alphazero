"""Preset games + a small registry used by the CLI, GUI and tests."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from .base import Game
from .mnk import MNKGame


@dataclass(frozen=True)
class GamePreset:
    name: str
    n: int
    m: int
    description: str

    def make(self) -> Game:
        return MNKGame(n=self.n, m=self.m)


PRESETS: Dict[str, GamePreset] = {
    "tictactoe": GamePreset("tictactoe", 3, 3, "tic-tac-toe (3x3, connect-3)"),
    "gomoku9": GamePreset("gomoku9", 9, 5, "Gomoku 9x9 (connect-5)"),
    "gomoku15": GamePreset("gomoku15", 15, 5, "Gomoku 15x15 (connect-5)"),
}


def get_preset(name: str) -> GamePreset:
    try:
        return PRESETS[name]
    except KeyError:
        raise KeyError(f"unknown game '{name}'. Available: {sorted(PRESETS)}") from None


def make_game(name: str) -> Game:
    return get_preset(name).make()
