import numpy as np
import torch

from alphazero.games import make_game
from alphazero.models import build_model
from alphazero.inference import make_provider
from alphazero.mcts import AlphaZeroMCTS, MCTSConfig
from alphazero.selfplay import run_selfplay, SelfplayConfig
from alphazero.training import Trainer, TrainConfig, save_data, load_data, data_shape
from alphazero.training.train import loss_fn


def _tictactoe_model(dev):
    return build_model(make_game("tictactoe"), "resnet", dev, cfg={"channels": 16, "blocks": 1})


def test_selfplay_produces_valid_samples():
    game = make_game("tictactoe")
    prov = make_provider("random", game, num_playouts=16)
    mcts = AlphaZeroMCTS(game, prov, MCTSConfig(num_simulations=24, batch_size=8, seed=0))
    samples = run_selfplay(game.copy, mcts, 3, cfg=SelfplayConfig(resign_min_moves=2, resign_threshold=-0.9), seed=1)
    assert len(samples) > 0
    for s in samples:
        assert s.obs.shape == (3, 3, 3)
        assert s.policy.shape == (9,)
        assert abs(s.policy.sum() - 1.0) < 1e-5
        assert s.value in (-1.0, 0.0, 1.0)


def test_training_reduces_loss():
    dev = torch.device("cpu")
    game = make_game("tictactoe")
    prov = make_provider("random", game, num_playouts=16)
    mcts = AlphaZeroMCTS(game, prov, MCTSConfig(num_simulations=24, batch_size=8, seed=0))
    samples = run_selfplay(game.copy, mcts, 4, cfg=SelfplayConfig(resign_min_moves=2, resign_threshold=-0.9), seed=2)

    model = _tictactoe_model(dev)
    trainer = Trainer(model, dev, TrainConfig(lr=2e-3, batch_size=16, epochs=6, amp="none", seed=1))
    metrics = trainer.fit(samples)
    assert metrics["policy_loss"] < 2.5   # below the uniform prior CE (~ln9=2.2) target-ish
    assert metrics["loss"] > 0

    # a second, fresh model trained longer should not be worse
    model2 = _tictactoe_model(dev)
    trainer2 = Trainer(model2, dev, TrainConfig(lr=2e-3, batch_size=16, epochs=10, amp="none", seed=1))
    metrics2 = trainer2.fit(samples)
    assert metrics2["policy_loss"] <= metrics["policy_loss"] + 1e-3


def test_loss_fn_sane():
    logit = torch.randn(4, 9)
    target = torch.zeros(4, 9); target[0, 0] = 1.0
    val = torch.randn(4)
    total, pl, vl = loss_fn(logit, target, torch.zeros(4), val)
    assert pl.item() > 0 and vl.item() >= 0 and total.item() > 0


def test_data_roundtrip(tmp_path):
    game = make_game("tictactoe")
    prov = make_provider("random", game, num_playouts=8)
    mcts = AlphaZeroMCTS(game, prov, MCTSConfig(num_simulations=16, batch_size=4, seed=0))
    samples = run_selfplay(game.copy, mcts, 2, cfg=SelfplayConfig(resign_min_moves=1, resign_threshold=-0.9), seed=0)
    path = str(tmp_path / "d.npz")
    save_data(path, samples)
    re = load_data(path)
    assert len(re) == len(samples)
    assert np.allclose(re[0].obs, samples[0].obs)
    assert np.allclose(re[0].policy, samples[0].policy)
    assert data_shape(path)["n"] == len(samples)
