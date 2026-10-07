import numpy as np

from alphazero.games.bitboard import build_config, has_win, has_win_naive


def _board_to_pboards(board, n):
    p1 = p2 = 0
    for r in range(n):
        for c in range(n):
            if board[r][c] == 1:
                p1 |= 1 << (r * n + c)
            elif board[r][c] == 2:
                p2 |= 1 << (r * n + c)
    return p1, p2


def test_win_detection_matches_naive():
    rng = np.random.default_rng(0)
    for n, m in [(3, 3), (4, 4), (5, 5), (7, 5), (9, 5)]:
        cfg = build_config(n, m)
        for _ in range(200):
            board = rng.integers(0, 3, size=(n, n)).tolist()
            p1, p2 = _board_to_pboards(board, n)
            got = has_win(p1, cfg) or has_win(p2, cfg)
            assert got == has_win_naive(board, m), (n, m, board)


def test_exact_win_lines_detected():
    cfg = build_config(5, 3)
    # horizontal
    p = sum(1 << (0 * 5 + c) for c in range(3))
    assert has_win(p, cfg)
    # vertical
    p = sum(1 << (r * 5 + 0) for r in range(3))
    assert has_win(p, cfg)
    # diagonal
    p = sum(1 << ((i) * 5 + i) for i in range(3))
    assert has_win(p, cfg)
    # anti-diagonal
    p = sum(1 << ((i) * 5 + (2 - i)) for i in range(3))
    assert has_win(p, cfg)


def test_no_false_wrap_positive():
    cfg = build_config(5, 5)
    # 4 in a row but with a gap that "wraps" visually should NOT be a win.
    # place 4 stones at (0,0),(1,1),(2,2) then (4,4) - not connected
    p = (1 << 0) | (1 << 1 * 5 + 1) | (1 << 2 * 5 + 2) | (1 << 4 * 5 + 4)
    assert not has_win(p, cfg)


def test_config_is_cached():
    a = build_config(15, 5)
    b = build_config(15, 5)
    assert a is b
    assert a.full_mask == (1 << 225) - 1
