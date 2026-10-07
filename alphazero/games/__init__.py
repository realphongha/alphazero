"""Game abstractions: base contract, bitboard engine, MNK concrete + presets."""
from .base import Game
from .bitboard import BitboardConfig, build_config, has_win, legal_mask
from .mnk import MNKGame
from .presets import GamePreset, PRESETS, get_preset, make_game

__all__ = [
    "Game",
    "BitboardConfig",
    "build_config",
    "has_win",
    "legal_mask",
    "MNKGame",
    "GamePreset",
    "PRESETS",
    "get_preset",
    "make_game",
]
