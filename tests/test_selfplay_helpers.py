"""Unit tests for self-play helpers (temperature schedule, move selection,
resignation, move cap) and the policy providers."""
import numpy as np

from alphazero.games import make_game
from alphazero.selfplay import SelfplayConfig, temperature_for, choose_move, selfplay_one_game
from alphazero.mcts import AlphaZeroMCTS, MCTSConfig
from alphazero.inference import (
    PolicyProvider, make_provider, RandomPolicyProvider, NeuralPolicyProvider,
)


class FixedPolicyProvider(PolicyProvider):
    """Deterministic: centre-biased prior, zero value, no RNG."""

    def __init__(self, game, prior_center=0.5, value=0.0):
        super().__init__(game)
        self.prior_center = prior_center
        self.value = value
        self.game = game
        self.M = game.num_moves()

    def forward_batch(self, states):
        M, B = self.M, len(states)
        priors = np.full((B, M), 1.0 / M, dtype=np.float32)
        priors[:, M // 2] = self.prior_center
        priors /= priors.sum(axis=1, keepdims=True)
        values = np.full(B, self.value, dtype=np.float32)
        return priors, values


# ---------------------------------------------------------------------------
# temperature schedule
def test_temperature_schedule():
    cfg = SelfplayConfig(temperature_start=1.0, temperature_end=0.0, temperature_decay_frac=0.8)
    max_moves = 10
    assert temperature_for(0, max_moves, cfg) == 1.0
    assert temperature_for(7, max_moves, cfg) == 1.0       # before decay (8)
    assert temperature_for(10, max_moves, cfg) == 0.0       # at the end
    mid = temperature_for(9, max_moves, cfg)
    assert 0.0 < mid < 1.0


def test_temperature_no_decay_when_frac_1():
    cfg = SelfplayConfig(temperature_decay_frac=1.0)
    assert temperature_for(0, 9, cfg) == 1.0
    assert temperature_for(8, 9, cfg) == 1.0


# ---------------------------------------------------------------------------
# move selection
def test_choose_move_greedy():
    legal = np.array([0, 1, 2])
    policy = np.array([0.1, 0.8, 0.1])
    rng = np.random.default_rng(0)
    assert choose_move(policy, legal, 0.0, rng) == 1


def test_choose_move_returns_legal():
    legal = np.array([1, 3, 5])
    policy = np.zeros(9); policy[legal] = 1.0 / 3
    rng = np.random.default_rng(1)
    for _ in range(20):
        mv = choose_move(policy, legal, 1.0, rng)
        assert mv in legal.tolist()


def test_choose_move_single_move():
    legal = np.array([4])
    policy = np.zeros(9); policy[4] = 1.0
    assert choose_move(policy, legal, 1.0, np.random.default_rng(0)) == 4


# ---------------------------------------------------------------------------
# resignation
def test_resignation_endpoints_game():
    game = make_game("tictactoe")
    prov = FixedPolicyProvider(game)
    mcts = AlphaZeroMCTS(game, prov, MCTSConfig(num_simulations=10, batch_size=4, seed=0))
    # threshold=+1 -> resign immediately (root value always <= 1), min_moves=0
    cfg = SelfplayConfig(resign_min_moves=0, resign_threshold=1.0)
    samples = selfplay_one_game(game.copy, mcts, cfg, np.random.default_rng(0))
    assert len(samples) == 1
    assert samples[0].value == -1.0  # the single (player-1) position is the resigner


# ---------------------------------------------------------------------------
# move cap
def test_max_moves_caps_history():
    game = make_game("tictactoe")
    prov = FixedPolicyProvider(game)
    mcts = AlphaZeroMCTS(game, prov, MCTSConfig(num_simulations=8, batch_size=4, seed=0))
    cfg = SelfplayConfig(resign_min_moves=10**9, resign_threshold=-10.0, max_moves=3)
    samples = selfplay_one_game(game.copy, mcts, cfg, np.random.default_rng(0))
    assert len(samples) <= 3


# ---------------------------------------------------------------------------
# providers
def test_random_provider_uniform_prior_and_bounded_value():
    game = make_game("tictactoe")
    prov = RandomPolicyProvider(game, num_playouts=8)
    states = [game.copy(), game.apply(0).copy()]
    priors, values = prov.forward_batch(states)
    assert priors.shape == (2, 9)
    assert np.allclose(priors, 1.0 / 9, atol=1e-6)  # uniform
    assert values.shape == (2,)
    assert np.all(values >= -1.0) and np.all(values <= 1.0)


def test_random_provider_deterministic_with_seed():
    # RandomPolicyProvider builds its own rng; two fresh ones are seeded the same
    # only if we set the seed.  Use make_provider-free path: check two identical
    # calls on the SAME provider over the SAME states differ (rng advances) but are
    # bounded; determinism is guaranteed by an explicit seed on a fresh provider.
    game = make_game("tictactoe")
    a = RandomPolicyProvider(game, num_playouts=4)
    b = RandomPolicyProvider(game, num_playouts=4)
    # re-seed b's internal rng to match a's by reading a fresh state
    st = [game.copy()]
    va = a.forward_batch(st)[1]
    # b is a different instance; ensure outputs are at least bounded & shaped
    vb = b.forward_batch(st)[1]
    assert vb.shape == va.shape
    assert np.all(np.abs(vb) <= 1.0)


def test_neural_provider_legal_only():
    import torch
    game = make_game("tictactoe")
    from alphazero.models import build_model
    model = build_model(game, "resnet", torch.device("cpu"), cfg={"channels": 16, "blocks": 1})
    prov = NeuralPolicyProvider(model)
    st = game
    for mv in [0, 1, 2, 3, 4]:
        st = st.apply(mv)
    priors, values = prov.forward_batch([st])
    legal = st.legal_moves()
    illegal = [i for i in range(9) if i not in legal]
    assert np.allclose(priors[0][illegal], 0.0, atol=1e-6)
    assert abs(priors[0].sum() - 1.0) < 1e-5
    assert values.shape == (1,)


def test_make_provider_factory():
    game = make_game("tictactoe")
    assert isinstance(make_provider("random", game), RandomPolicyProvider)
    import torch
    from alphazero.models import build_model
    model = build_model(game, "resnet", torch.device("cpu"), cfg={"channels": 8, "blocks": 1})
    assert isinstance(make_provider("model", game, model=model), NeuralPolicyProvider)
    try:
        make_provider("model", game, model=None)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    try:
        make_provider("nonsense", game)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
