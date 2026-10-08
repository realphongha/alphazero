"""Tests for player-perspective-normalised observations and multi-source self-play."""
import numpy as np
import torch

from alphazero.games import make_game
from alphazero.games.mnk import _bits_to_array
from alphazero.models import build_model
from alphazero.inference import NeuralPolicyProvider
from alphazero.mcts import AlphaZeroMCTS, MCTSConfig
from alphazero.selfplay import run_mixed_selfplay, SelfplayConfig


def test_observation_channel0_is_mover():
    g = make_game("gomoku9")
    o = g.observation()
    assert o.shape == (3, 9, 9)
    assert o[0].sum() == 0.0 and o[1].sum() == 0.0  # empty, P1 to move
    g1 = g.apply(0)  # P1 plays cell 0 -> now P2 to move
    o1 = g1.observation()
    assert o1[0].sum() == 0.0            # mover is P2 (no stones yet)
    assert o1[1].sum() == 1.0            # opponent is P1 (one stone)
    assert o1[1].reshape(-1)[0] == 1.0   # that stone sits at cell 0
    # the legal mask (ch2) is always the mover's, unchanged by the swap
    assert o1[2].sum() == float(len(g1.legal_moves()))


def test_observation_mover_opponent_invariant():
    """Across many random positions: ch0 == mover's stones, ch1 == opponent's stones."""
    g = make_game("gomoku9")
    rng = np.random.default_rng(0)
    size = 81
    for _ in range(40):
        legal = g.legal_moves()
        if len(legal) == 0:
            break
        g = g.apply(int(rng.choice(legal)))
        o = g.observation()
        mover = g.p1 if g.turn == 1 else g.p2
        opp = g.p2 if g.turn == 1 else g.p1
        assert np.allclose(o[0].reshape(-1), _bits_to_array(mover, size))
        assert np.allclose(o[1].reshape(-1), _bits_to_array(opp, size))
        assert o[2].sum() == float(len(g.legal_moves()))


def test_run_mixed_selfplay_records_valid_net_data():
    dev = torch.device("cpu")
    game = make_game("tictactoe")
    model = build_model(game, "resnet", dev, cfg={"channels": 16, "blocks": 1})
    prov = NeuralPolicyProvider(model, None)
    mcts = AlphaZeroMCTS(game, prov, MCTSConfig(num_simulations=4, batch_size=4, seed=0))
    rng = np.random.default_rng(0)
    opp = lambda s: int(rng.choice(s.legal_moves()))
    samples = run_mixed_selfplay(game.copy, mcts, opp, 3, net_player=2,
                                 cfg=SelfplayConfig(resign_min_moves=999), seed=0)
    assert len(samples) > 0
    for s in samples:
        assert s.obs.shape == (3, 3, 3)
        assert abs(s.policy.sum() - 1.0) < 1e-5
        assert s.value in (-1.0, 0.0, 1.0)
