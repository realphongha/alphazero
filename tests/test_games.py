import numpy as np

from alphazero.games import make_game, PRESETS


def test_presets_exist():
    assert set(PRESETS) == {"tictactoe", "gomoku9", "gomoku15"}


def test_shapes_and_moves():
    for name, (n, m) in {"tictactoe": (3, 3), "gomoku9": (9, 5), "gomoku15": (15, 5)}.items():
        g = make_game(name)
        assert g.num_moves() == n * n
        assert g.n == n and g.m == m
        assert len(g.legal_moves()) == n * n
        obs = g.observation()
        assert obs.shape == (3, n, n)
        assert obs.dtype == np.float32
        assert obs.max() <= 1.0 and obs.min() >= 0.0


def test_apply_and_termination():
    g = make_game("tictactoe")
    # X wins top row: X=0,1,2 with O in between
    seq = [0, 3, 1, 4, 2]
    for mv in seq:
        g = g.apply(mv)
    assert g.winner() == 1
    assert g.is_terminal()
    # illegal move raises
    try:
        g.apply(2)
        raise AssertionError("expected illegal move to raise")
    except ValueError:
        pass


def test_apply_is_immutable():
    g = make_game("gomoku9")
    g2 = g.apply(0)
    assert g.p1 == 0            # original unchanged
    assert g2.p1 != 0
    assert g.turn != g2.turn


def test_draw_when_board_full():
    # fill a 3x3 board with no winner is impossible in tictactoe rules,
    # but we can verify terminal detection via a full non-winning pattern on 3x3.
    g = make_game("tictactoe")
    # X O X / X O O / O X X  -> no 3-in-a-row, board full
    layout = {
        0: 1, 1: 2, 2: 1,
        3: 1, 4: 2, 5: 2,
        6: 2, 7: 1, 8: 1,
    }
    order = list(layout.keys())
    state = g.copy()
    # interleave to respect turns: X plays 0,3,7 ; O plays 1,4,5,6,8? counts must alternate.
    # Instead just set bits directly to test terminal logic on a full board.
    p1 = 0
    p2 = 0
    for i, who in layout.items():
        if who == 1:
            p1 |= 1 << i
        else:
            p2 |= 1 << i
    from alphazero.games.mnk import MNKGame
    full = MNKGame(n=3, m=3)
    full.p1 = p1
    full.p2 = p2
    assert full.winner() == 0
    assert full.is_terminal()  # board full, no winner


def test_render_smoke():
    g = make_game("tictactoe")
    g = g.apply(4).apply(0)
    txt = g.render()
    assert len(txt.splitlines()) == 3
    assert "X" in txt and "O" in txt
