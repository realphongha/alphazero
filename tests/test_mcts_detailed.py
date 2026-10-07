"""Detailed unit tests for MCTS: PUCT selection, policy legality, edge cases,
and determinism (via a fixed, tool-free policy provider)."""
import numpy as np
import torch

from alphazero.games import make_game
from alphazero.inference import PolicyProvider
from alphazero.mcts import AlphaZeroMCTS, MCTSConfig, _Node


class FixedPolicyProvider(PolicyProvider):
    """Deterministic provider: fixed (non-uniform) priors, fixed value. No RNG."""

    def __init__(self, game, prior_center=0.5, value=0.0):
        super().__init__(game)
        self.prior_center = prior_center
        self.value = value
        self.game = game
        self.M = game.num_moves()

    def forward_batch(self, states):
        M = self.M
        B = len(states)
        priors = np.full((B, M), 1.0 / M, dtype=np.float32)
        # emphasise the centre (cell N*N//2)
        c = M // 2
        priors[:, c] = self.prior_center
        priors /= priors.sum(axis=1, keepdims=True)
        values = np.full(B, self.value, dtype=np.float32)
        return priors, values


def test_select_puct_formula():
    game = make_game("tictactoe")
    mcts = AlphaZeroMCTS(game, FixedPolicyProvider(game), MCTSConfig(c_puct=1.0, batch_size=4))
    root = _Node(None, game, -1)
    kids = [game.apply(i) for i in range(3)]
    for i, kg in enumerate(kids):
        root.children.append(_Node(root, kg, i))
    root.moves = np.array([0, 1, 2], dtype=np.int64)
    root.priors = np.array([0.1, 0.5, 0.4], dtype=np.float32)
    root.child_Q = np.array([0.0, 0.9, 0.0])   # child1 great FOR the child (bad for us)
    root.child_N = np.array([1, 100, 1])
    root.N = 102
    # independent PUCT re-implementation
    P = root.priors; Qc = root.child_Q; Nc = root.child_N
    safe = np.where(Nc > 0, Nc, 1)
    scores = np.where(Nc > 0, -Qc / safe, 0.0) + 1.0 * P * np.sqrt(root.N + 1) / (1 + Nc)
    expected = int(np.argmax(scores))
    got = mcts._select(root)
    assert root.children.index(got) == expected
    # the unvisited, high-prior child 2 must beat the heavily-negative-for-us child 1
    assert expected == 2


def test_policy_zero_on_illegal_moves():
    game = make_game("tictactoe")
    prov = FixedPolicyProvider(game, prior_center=0.4)
    mcts = AlphaZeroMCTS(game, prov, MCTSConfig(num_simulations=60, batch_size=8, seed=0))
    st = game
    for mv in [0, 1, 2, 3]:
        st = st.apply(mv)
    pol = mcts.search(st)
    illegal = [i for i in range(9) if not st.is_move_legal(i)]
    legal = [i for i in range(9) if st.is_move_legal(i)]
    assert np.allclose(pol[illegal], 0.0, atol=1e-7)
    assert np.all(pol[legal] > 0)
    assert abs(pol.sum() - 1.0) < 1e-6


def test_terminal_root_returns_zeros():
    game = make_game("tictactoe")
    g = game
    for mv in [0, 3, 1, 4, 2]:
        g = g.apply(mv)
    assert g.is_terminal()
    prov = FixedPolicyProvider(game)
    mcts = AlphaZeroMCTS(game, prov, MCTSConfig(num_simulations=20, batch_size=4))
    pol = mcts.search(g)
    assert np.allclose(pol, 0.0)


def test_root_value_bounded():
    game = make_game("gomoku9")
    prov = FixedPolicyProvider(game, value=0.3)
    mcts = AlphaZeroMCTS(game, prov, MCTSConfig(num_simulations=40, batch_size=8, seed=1))
    mcts.search(game.copy())
    assert -1.0 <= mcts.last_root_value <= 1.0


def test_deterministic_with_fixed_provider_and_seed():
    game = make_game("tictactoe")
    prov = FixedPolicyProvider(game)
    cfg = MCTSConfig(num_simulations=50, batch_size=8, seed=99, dirichlet_epsilon=0.0)
    m1 = AlphaZeroMCTS(game, prov, cfg)
    m2 = AlphaZeroMCTS(game, prov, cfg)
    p1 = m1.search(game.copy())
    p2 = m2.search(game.copy())
    assert np.allclose(p1, p2, atol=1e-7)  # same seed + fixed provider -> reproducible
    assert abs(p1.sum() - 1.0) < 1e-6


def test_more_sims_concentrates_policy():
    game = make_game("tictactoe")
    prov = FixedPolicyProvider(game, prior_center=0.6)
    mcts = AlphaZeroMCTS(game, prov, MCTSConfig(num_simulations=300, batch_size=16, seed=0,
                                                 dirichlet_epsilon=0.0))
    pol = mcts.search(game.copy())
    assert pol.argmax() == 4  # centre dominates
