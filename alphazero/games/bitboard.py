"""Bitboard primitives for square connect-M games (tic-tac-toe, Gomoku...).

A board is two Python ``int``s (one per player) of ``n*n`` bits. Cell index is
``row * n + col`` (row 0 = top). Python ints are arbitrary precision, so a
15x15 (225-bit) board fits in a single register-sized value and every
set/shift/popcount operation runs in C with no allocation — which keeps MCTS
expand + terminal checks very cheap.

Win detection uses the classic "intersect successive shifts" trick:

    a run of M starts at cell ``s`` in direction ``d`` iff
        s, s+delta, ..., s+(M-1)*delta  are all stones

i.e. ``AND_k (stones >> k*delta)`` stays non-empty.  Shifting a stone that is
too close to an edge would wrap into the *wrong* row, so for each shift we
clear the "wrap" cells (``bad[k]``).  Off-board cells need no masking because
a right-shift simply drops them.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple

# Canonical directions as (label, per-step index delta).  All deltas are
# positive so a right-shift by ``k*delta`` always moves toward lower indices.
_DIRECTIONS = (("E", 1), ("S", 7), ("SE", 8), ("SW", 6))  # placeholders, real deltas set per board


@dataclass(frozen=True)
class BitboardConfig:
    n: int
    m: int
    size: int          # n*n cells
    full_mask: int     # (1 << size) - 1
    deltas: Tuple[int, ...]          # per-direction index delta
    bad: Tuple[Tuple[int, ...], ...]  # bad[k] per direction, k in 0..m-1

    def directions(self):
        return list(zip(self.deltas, self.bad))


def _straight_ok(direction: str, r: int, c: int, k: int, n: int) -> bool:
    """True if moving ``k`` straight steps from (r,c) stays on the board."""
    if direction == "E":
        return c + k < n
    if direction == "S":
        return r + k < n
    if direction == "SE":
        return r + k < n and c + k < n
    if direction == "SW":
        return r + k < n and c - k >= 0
    raise ValueError(direction)


def _delta_for(direction: str, n: int) -> int:
    return {"E": 1, "S": n, "SE": n + 1, "SW": n - 1}[direction]


@lru_cache(maxsize=None)
def build_config(n: int, m: int) -> BitboardConfig:
    """Precompute full masks + per-direction shift deltas and wrap masks.

    ``bad[d][k]`` is a bitmask of cells ``s`` for which the (k+1)-th stone of a
    run starting at ``s`` would *wrap across a row boundary* (i.e. is an
    in-range but geometrically-invalid continuation).  Such cells are cleared
    before each shift so a run can never "jump" rows.
    """
    size = n * n
    full_mask = (1 << size) - 1
    deltas: List[int] = []
    bad: List[Tuple[int, ...]] = []

    for direction in ("E", "S", "SE", "SW"):
        delta = _delta_for(direction, n)
        masks = [0] * m
        for k in range(1, m):
            wrap = 0
            for r in range(n):
                for c in range(n):
                    s = r * n + c
                    t = s + k * delta
                    if 0 <= t < size and not _straight_ok(direction, r, c, k, n):
                        wrap |= 1 << s
            masks[k] = wrap
        deltas.append(delta)
        bad.append(tuple(masks))

    return BitboardConfig(
        n=n, m=m, size=size, full_mask=full_mask,
        deltas=tuple(deltas), bad=tuple(bad),
    )


def has_win(stones: int, cfg: BitboardConfig) -> bool:
    """True if ``stones`` contains M-or-more-in-a-row anywhere on the board."""
    m = cfg.m
    for delta, masks in cfg.directions():
        acc = cfg.full_mask
        for k in range(m):
            acc &= (stones >> (k * delta)) & ~masks[k]
            if acc == 0:
                break
        if acc != 0:
            return True
    return False


def winning_mask(stones: int, cfg: BitboardConfig) -> int:
    """Bitmask of cells that are the *start* of a winning run (for debug/UI)."""
    m = cfg.m
    for delta, masks in cfg.directions():
        acc = cfg.full_mask
        for k in range(m):
            acc &= (stones >> (k * delta)) & ~masks[k]
            if acc == 0:
                break
        if acc != 0:
            return acc
    return 0


def legal_mask(p1: int, p2: int, cfg: BitboardConfig) -> int:
    return cfg.full_mask & ~(p1 | p2)


def popcount(x: int) -> int:
    return x.bit_count()


def iter_set_bits(x: int) -> List[int]:
    """Indices of set bits, ascending."""
    out = []
    i = x
    while i:
        lsb = i & -i
        out.append(lsb.bit_length() - 1)
        i ^= lsb
    return out


def cell_rc(index: int, n: int) -> Tuple[int, int]:
    return divmod(index, n)


def rc_cell(r: int, c: int, n: int) -> int:
    return r * n + c


# A naive reference, used only by the test-suite to property-test the fast
# bitboard above.  Deliberately simple and obviously correct.
def has_win_naive(board: List[List[int]], m: int) -> bool:
    """``board[r][c]`` is 0/1/2.  Returns True if any player has M-in-a-row."""
    n = len(board)
    dirs = ((0, 1), (1, 0), (1, 1), (1, -1))
    for r in range(n):
        for c in range(n):
            v = board[r][c]
            if not v:
                continue
            for dr, dc in dirs:
                ok = True
                for k in range(1, m):
                    rr, cc = r + dr * k, c + dc * k
                    if rr < 0 or rr >= n or cc < 0 or cc >= n or board[rr][cc] != v:
                        ok = False
                        break
                if ok:
                    return True
    return False
