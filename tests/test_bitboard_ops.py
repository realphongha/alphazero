"""Detailed unit tests for the bitboard engine (ops + win detection).

Includes systematic white-box checks of the precomputed wrap masks and the
successive-shift win test, plus a naive reference cross-check.
"""
import numpy as np

from alphazero.games.bitboard import (
    build_config,
    has_win,
    has_win_naive,
    legal_mask,
    iter_set_bits,
    popcount,
    winning_mask,
    cell_rc,
    rc_cell,
)


# ---------------------------------------------------------------------------
# helpers
def _cfg(n, m):
    return build_config(n, m)


def _place_line(n, m, r, c, dr, dc, k):
    """Place k stones from (r,c) along (dr,dc); return (stones, valid_in_board)."""
    stones = 0
    for j in range(k):
        rr, cc = r + dr * j, c + dc * j
        if 0 <= rr < n and 0 <= cc < n:
            stones |= 1 << (rr * n + cc)
    valid = all(0 <= r + dr * j < n and 0 <= c + dc * j < n for j in range(k))
    return stones, valid


DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


# ---------------------------------------------------------------------------
# primitive ops
def test_build_config_basics():
    cfg = _cfg(15, 5)
    assert cfg.size == 225
    assert cfg.full_mask == (1 << 225) - 1
    assert cfg.n == 15 and cfg.m == 5
    # deltas
    assert cfg.deltas == (1, 15, 16, 14)  # E, S, SE, SW
    # bad[0] is always 0 (no shift)
    for _, bad in cfg.directions():
        assert bad[0] == 0
        assert len(bad) == 5


def test_cell_roundtrip():
    for n in (3, 9, 15):
        for idx in range(n * n):
            r, c = cell_rc(idx, n)
            assert rc_cell(r, c, n) == idx


def test_popcount_and_iter_set_bits():
    x = (1 << 0) | (1 << 3) | (1 << 224)
    assert popcount(x) == 3
    assert iter_set_bits(x) == [0, 3, 224]
    assert popcount(0) == 0
    assert iter_set_bits(0) == []


def test_legal_mask():
    cfg = _cfg(3, 3)
    assert legal_mask(0, 0, cfg) == (1 << 9) - 1
    assert legal_mask(1, 2, cfg) == ((1 << 9) - 1) & ~3  # clear bits 0 and 1
    full = (1 << 9) - 1
    assert legal_mask(full // 1, full, cfg) == 0  # both players fill all bits -> none


# ---------------------------------------------------------------------------
# win detection: exact boundary (M-1 vs M vs M+1)
def test_exact_win_boundary():
    n, m = 5, 5
    cfg = _cfg(n, m)
    # 4 in a row (M-1) is NOT a win
    stones = _place_line(n, m, 2, 0, 0, 1, m - 1)[0]
    assert not has_win(stones, cfg)
    # 5 in a row (M) IS a win
    stones = _place_line(n, m, 2, 0, 0, 1, m)[0]
    assert has_win(stones, cfg)
    # 6 in a row (M+1, overline) IS a win (freestyle)
    stones = _place_line(n, m, 2, 0, 0, 1, m + 1)[0]
    assert has_win(stones, cfg)


def test_no_false_win_from_two_short_lines():
    # Two 3-in-a-rows that share nothing, M=5 -> no win
    n, m = 7, 5
    cfg = _cfg(n, m)
    s1, _ = _place_line(n, m, 0, 0, 0, 1, 3)
    s2, _ = _place_line(n, m, 6, 3, 0, 1, 3)
    assert not has_win(s1 | s2, cfg)


# ---------------------------------------------------------------------------
# systematic: every start cell x every direction, for M and M+1
def test_win_from_every_cell_and_direction():
    for n, m in [(5, 3), (5, 5), (7, 5)]:
        cfg = _cfg(n, m)
        for dr, dc in DIRS:
            for r in range(n):
                for c in range(n):
                    stones, valid = _place_line(n, m, r, c, dr, dc, m)
                    if valid:
                        assert has_win(stones, cfg), (n, m, r, c, dr, dc)
                    # break it: only M-1 stones -> no win (if it was fully in board)
                    if valid:
                        short, _ = _place_line(n, m, r, c, dr, dc, m - 1)
                        # a shorter line might still contain an M-run elsewhere? No,
                        # it's exactly M-1 contiguous; but check it is not a win on its own
                        assert not has_win(short, cfg), (n, m, r, c, dr, dc)


# ---------------------------------------------------------------------------
# wrap masks: white-box check against a naive reference
def _expected_bad(n, m):
    """Reference bad masks: bad[d][k] = cells s with s+k*delta in range but not a
    straight k-step line."""
    size = n * n
    deltas = [1, n, n + 1, n - 1]

    def straight_ok(d, r, c, k):
        if d == 0:   # E
            return c + k < n
        if d == 1:   # S
            return r + k < n
        if d == 2:   # SE
            return r + k < n and c + k < n
        return r + k < n and c - k >= 0  # SW
    out = []
    for d, delta in enumerate(deltas):
        masks = [0] * m
        for k in range(1, m):
            w = 0
            for r in range(n):
                for c in range(n):
                    s = r * n + c
                    t = s + k * delta
                    if 0 <= t < size and not straight_ok(d, r, c, k):
                        w |= 1 << s
            masks[k] = w
        out.append(tuple(masks))
    return tuple(out)


def test_wrap_masks_match_reference():
    for n, m in [(3, 3), (5, 5), (7, 5), (9, 5)]:
        cfg = _cfg(n, m)
        assert cfg.bad == _expected_bad(n, m), (n, m)


# ---------------------------------------------------------------------------
# winning_mask reports a start cell of the run
def test_winning_mask():
    cfg = _cfg(5, 3)
    r, c, dr, dc = 0, 0, 0, 1
    stones, _ = _place_line(5, 3, r, c, dr, dc, 3)
    wm = winning_mask(stones, cfg)
    assert wm != 0
    # the run start cell should be in the mask
    assert wm & (1 << (r * 5 + c))
    assert has_win_naive([[1 if stones & (1 << (rr * 5 + cc)) else 0
                           for cc in range(5)] for rr in range(5)], 3)


# ---------------------------------------------------------------------------
# cross-check has_win vs naive on dense random boards (larger than the base test)
def test_has_win_vs_naive_random_dense():
    rng = np.random.default_rng(7)
    for n, m in [(3, 3), (5, 5), (9, 5)]:
        cfg = _cfg(n, m)
        for _ in range(400):
            # denser boards (more likely to contain a win) -> less zeros
            board = (rng.random((n, n)) < 0.4).astype(int) * 2
            board[rng.random((n, n)) < 0.3] = 1  # sprinkle 1s
            p1 = p2 = 0
            for rr in range(n):
                for cc in range(n):
                    if board[rr][cc] == 1:
                        p1 |= 1 << (rr * n + cc)
                    elif board[rr][cc] == 2:
                        p2 |= 1 << (rr * n + cc)
            assert (has_win(p1, cfg) or has_win(p2, cfg)) == has_win_naive(board.tolist(), m)


# ---------------------------------------------------------------------------
# board full / no winner (the canonical full tic-tac-toe draw)
def test_board_full_no_win():
    cfg = _cfg(3, 3)
    # X O X / X O O / O X X  -> full, no 3-in-a-row for either player
    layout = {0: 1, 1: 2, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 1, 8: 1}
    p1 = sum(1 << i for i, w in layout.items() if w == 1)
    p2 = sum(1 << i for i, w in layout.items() if w == 2)
    assert not has_win(p1, cfg) and not has_win(p2, cfg)
    assert legal_mask(p1, p2, cfg) == 0  # board full -> no legal moves
