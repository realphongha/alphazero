"""Detailed unit tests for game logic (MNKGame)."""
import numpy as np

from alphazero.games import make_game
from alphazero.games.mnk import MNKGame


def test_turn_alternates_and_move_count():
    g = make_game("gomoku9")
    assert g.current_player() == 1
    assert g.move_count == 0
    for i, mv in enumerate([0, 1, 2]):
        g = g.apply(mv)
        assert g.move_count == i + 1
        # after (i+1) moves the to-move player alternates
        assert g.current_player() == (2 if (i + 1) % 2 == 1 else 1)
    assert g.current_player() == 2  # 3 moves -> player 2 to move


def test_reward_perspective():
    g = make_game("tictactoe")
    for mv in [0, 3, 1, 4, 2]:  # X (p1) wins top row
        g = g.apply(mv)
    assert g.winner() == 1
    # value from each player's perspective
    assert g.reward(1) == 1.0
    assert g.reward(2) == -1.0


def test_last_move_tracked():
    g = make_game("tictactoe")
    assert g.last_move == -1
    g = g.apply(7)
    assert g.last_move == 7


def test_is_move_legal_boundaries():
    g = make_game("tictactoe")
    assert g.is_move_legal(0)
    assert g.is_move_legal(8)
    assert not g.is_move_legal(-1)
    assert not g.is_move_legal(9)
    assert not g.is_move_legal(100)
    g = g.apply(0)
    assert not g.is_move_legal(0)  # now occupied


def test_copy_is_deep_and_independent():
    g = make_game("gomoku9")
    g = g.apply(4).apply(3)
    c = g.copy()
    c2 = c.apply(0)
    assert g.p1 != c2.p1  # mutating the copy does not affect the original
    assert c.p1 == g.p1
    assert g.turn == c.turn


def _win_at(game_n, m, start_rc, dr, dc):
    """Build a game where the current player just completed a line of m from
    start_rc in direction (dr,dc); return the state + the winning player."""
    g = MNKGame(n=game_n, m=m)
    # place opponent stones on a far row/col to consume turns without forming a line,
    # then place the winning line for the player to move.
    wins = []
    cells = [(start_rc[0] + dr * k, start_rc[1] + dc * k) for k in range(m)]
    # play the winning stones as player 1, filler as player 2 on disjoint cells
    p1 = p2 = 0
    used = set(cells)
    for r in range(game_n):
        for c in range(game_n):
            if (r, c) in used:
                continue
            if p2 == 0 and (r == game_n - 1):
                p2 |= 1 << (r * game_n + c)
                used.add((r, c))
    for (r, c) in cells:
        p1 |= 1 << (r * game_n + c)
    # reconstruct a legal-ish state directly (for terminal/winner checks only)
    st = MNKGame(n=game_n, m=m)
    st.p1 = p1
    st.p2 = p2
    return st, 1


def test_horizontal_win_all_rows():
    for row in range(9):
        st, p = _win_at(9, 5, (row, 0), 0, 1)
        assert st.winner() == p, row
        assert st.is_terminal()


def test_vertical_win_all_cols():
    for col in range(9):
        st, p = _win_at(9, 5, (0, col), 1, 0)
        assert st.winner() == p, col


def test_diagonal_win_both_directions():
    st, p = _win_at(9, 5, (0, 0), 1, 1)
    assert st.winner() == p
    st, p = _win_at(9, 5, (0, 4), 1, -1)
    assert st.winner() == p


def test_wins_on_15x15_board_edges():
    st, p = _win_at(15, 5, (14, 0), 0, 1)  # bottom row
    assert st.winner() == p
    st, p = _win_at(15, 5, (0, 14), 1, 0)  # right col
    assert st.winner() == p
    st, p = _win_at(15, 5, (0, 0), 1, 1)  # main diagonal
    assert st.winner() == p


def test_tictactoe_all_win_lines():
    lines = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
        (0, 4, 8), (2, 4, 6),             # diags
    ]
    for line in lines:
        p1 = sum(1 << i for i in line)
        st = MNKGame(n=3, m=3)
        st.p1 = p1
        assert st.winner() == 1, line
        assert st.is_terminal()


def test_winner_uniqueness_and_zero_ongoing():
    g = make_game("tictactoe")
    assert g.winner() == 0 and not g.is_terminal()
    g = g.apply(0)
    assert g.winner() == 0
