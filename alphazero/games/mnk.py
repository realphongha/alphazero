"""Concrete MNK (connect-M on an N x N board) game backed by bitboards.

Covers the whole wishlist family in one class:

    * tic-tac-toe  : N=3,  M=3
    * Gomoku  9x9  : N=9,  M=5
    * Gomoku 15x15 : N=15, M=5

Overlines (more than M in a row) also win — this is the standard "freestyle"
Gomoku rule and keeps the bitboard win test a single fast expression.  Renju
forbidden moves are intentionally out of scope for a training/search toy.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Game
from .bitboard import BitboardConfig, build_config, has_win, legal_mask


def _bits_to_array(bits: int, size: int) -> np.ndarray:
    """Vectorised int->float array: ``out[i]`` is bit ``i`` of ``bits``."""
    nbytes = (size + 7) // 8
    b = bits.to_bytes(nbytes, "big")
    arr = np.unpackbits(np.frombuffer(b, dtype=np.uint8))
    return arr[::-1][:size].astype(np.float32)


class MNKGame(Game):
    __slots__ = ("cfg", "p1", "p2", "turn", "move_count", "last_move", "_legal")

    def __init__(self, n: int = 15, m: int = 5) -> None:
        self.cfg: BitboardConfig = build_config(n, m)
        self.p1: int = 0
        self.p2: int = 0
        self.turn: int = 1
        self.move_count: int = 0
        self.last_move: int = -1
        self._legal: Optional[int] = None

    # -- identity / shape -------------------------------------------------
    def num_moves(self) -> int:
        return self.cfg.size

    def board_size(self) -> int:
        return self.cfg.size

    @property
    def n(self) -> int:
        return self.cfg.n

    @property
    def m(self) -> int:
        return self.cfg.m

    # -- play -------------------------------------------------------------
    def current_player(self) -> int:
        return self.turn

    def _player_board(self, player: int) -> int:
        return self.p1 if player == 1 else self.p2

    def legal_moves(self) -> np.ndarray:
        if self._legal is None:
            self._legal = legal_mask(self.p1, self.p2, self.cfg)
        bits = self._legal
        out = []
        while bits:
            lsb = bits & -bits
            out.append(lsb.bit_length() - 1)
            bits ^= lsb
        return np.asarray(out, dtype=np.int64)

    def is_move_legal(self, move: int) -> bool:
        return 0 <= move < self.cfg.size and not ((self.p1 | self.p2) >> move) & 1

    def apply(self, move: int) -> "MNKGame":
        clone = self.copy()
        if not clone.is_move_legal(move):
            raise ValueError(f"illegal move {move}")
        bit = 1 << move
        if clone.turn == 1:
            clone.p1 |= bit
        else:
            clone.p2 |= bit
        clone.turn = 3 - clone.turn
        clone.move_count += 1
        clone.last_move = move
        clone._legal = None
        return clone

    def copy(self) -> "MNKGame":
        g = MNKGame.__new__(MNKGame)
        g.cfg = self.cfg
        g.p1 = self.p1
        g.p2 = self.p2
        g.turn = self.turn
        g.move_count = self.move_count
        g.last_move = self.last_move
        g._legal = self._legal
        return g

    # -- outcome ----------------------------------------------------------
    def _winner_of(self, player: int) -> bool:
        return has_win(self._player_board(player), self.cfg)

    def is_terminal(self) -> bool:
        if self._winner_of(1) or self._winner_of(2):
            return True
        return legal_mask(self.p1, self.p2, self.cfg) == 0  # board full

    def winner(self) -> int:
        if self._winner_of(1):
            return 1
        if self._winner_of(2):
            return 2
        return 0

    # -- observation ------------------------------------------------------
    def observation(self) -> np.ndarray:
        """(3, n, n) float32: [player1, player2, legal-mask]."""
        n = self.cfg.n
        size = n * n
        leg = legal_mask(self.p1, self.p2, self.cfg)
        obs = np.zeros((3, n, n), dtype=np.float32)
        obs[0] = _bits_to_array(self.p1, size).reshape(n, n)
        obs[1] = _bits_to_array(self.p2, size).reshape(n, n)
        obs[2] = _bits_to_array(leg, size).reshape(n, n)
        return obs

    # -- pretty printing (debug / CLI) ------------------------------------
    def render(self) -> str:
        n = self.cfg.n
        p1, p2 = self.p1, self.p2
        lines = []
        for r in range(n):
            row = []
            base = r * n
            for c in range(n):
                s = 1 << (base + c)
                row.append("X" if p1 & s else ("O" if p2 & s else "."))
            lines.append("".join(row))
        return "\n".join(lines)
