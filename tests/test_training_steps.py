"""Detailed unit tests for the training step (AMP, clipping, scheduler, learning)."""
import math

import numpy as np
import torch

from alphazero.games import make_game
from alphazero.models import build_model
from alphazero.training.data import Sample
from alphazero.training.train import Trainer, TrainConfig, loss_fn


def _model(dev):
    return build_model(make_game("tictactoe"), "resnet", dev, cfg={"channels": 16, "blocks": 1})


def _data(n=60, policy_at=4, value=0.0, seed=0):
    rng = np.random.default_rng(seed)
    game = make_game("tictactoe")
    obs = game.observation()
    samples = []
    for _ in range(n):
        pol = np.zeros(9, dtype=np.float32)
        pol[policy_at] = 1.0
        samples.append(Sample(obs.copy(), pol, value))
    return samples


def test_weights_change_after_fit():
    dev = torch.device("cpu")
    m = _model(dev)
    before = {n: p.clone() for n, p in m.named_parameters()}
    t = Trainer(m, dev, TrainConfig(lr=1e-2, batch_size=32, epochs=3, amp="none", seed=1))
    t.fit(_data())
    after = dict(m.named_parameters())
    assert any(not torch.equal(before[n], after[n]) for n in before)


def test_loss_is_finite_and_below_uniform_prior():
    dev = torch.device("cpu")
    m = _model(dev)
    t = Trainer(m, dev, TrainConfig(lr=1e-2, batch_size=32, epochs=20, amp="none", seed=1))
    met = t.fit(_data(n=80, policy_at=4, value=0.0))
    assert math.isfinite(met["loss"]) and math.isfinite(met["policy_loss"])
    # uniform prior over 9 moves has CE ~ ln(9) ~= 2.197; learning one-hot must beat it
    assert met["policy_loss"] < 1.5


def test_model_learns_one_hot_policy():
    dev = torch.device("cpu")
    m = _model(dev)
    t = Trainer(m, dev, TrainConfig(lr=1e-2, batch_size=32, epochs=40, amp="none", seed=1))
    t.fit(_data(n=100, policy_at=4, value=0.0))
    m.eval()
    with torch.no_grad():
        p, _ = m.predict_batch([make_game("tictactoe")])
    assert p[0].argmax() == 4
    assert p[0][4] > 0.5


def test_model_learns_value():
    dev = torch.device("cpu")
    m = _model(dev)
    t = Trainer(m, dev, TrainConfig(lr=1e-2, batch_size=32, epochs=40, amp="none", seed=1))
    t.fit(_data(n=100, policy_at=0, value=1.0))
    m.eval()
    with torch.no_grad():
        _, v = m.predict_batch([make_game("tictactoe")])
    assert v[0] > 0.3  # learned to predict a win (+1)


def test_gradient_clipping_keeps_finite():
    dev = torch.device("cpu")
    m = _model(dev)
    # absurdly large LR would normally blow up; clipping must keep it finite
    t = Trainer(m, dev, TrainConfig(lr=50.0, batch_size=16, epochs=2, amp="none",
                                    grad_clip=1.0, seed=1))
    met = t.fit(_data(n=40))
    assert math.isfinite(met["loss"])
    for p in m.parameters():
        assert torch.isfinite(p).all()


def test_scheduler_applies_lr_to_optimizer():
    dev = torch.device("cpu")
    m = _model(dev)
    base_lr = 3e-3
    t = Trainer(m, dev, TrainConfig(lr=base_lr, batch_size=32, epochs=2, amp="none",
                                    warmup_steps=3, seed=1))
    t.fit(_data(n=48))
    cur = t.optimizer.param_groups[0]["lr"]
    assert 0 < cur <= base_lr


def test_grad_accum_effective_batch():
    dev = torch.device("cpu")
    m = _model(dev)
    # batch 8, accum 2 -> micro-batch 4 -> 16 samples / 8 per opt step = 2 steps
    t = Trainer(m, dev, TrainConfig(lr=1e-2, batch_size=8, grad_accum=2, epochs=1, amp="none", seed=1))
    met = t.fit(_data(n=16))
    assert math.isfinite(met["loss"])
    assert met["optimizer_steps"] == 2


def test_loss_fn_components():
    pol = torch.log_softmax(torch.randn(4, 9), dim=1)
    target = torch.zeros(4, 9); target[:, 0] = 1.0
    pred_v = torch.zeros(4, 1)
    target_v = torch.ones(4)
    total, pl, vl = loss_fn(torch.exp(pol), target, pred_v, target_v)
    assert pl.item() > 0
    assert abs(vl.item() - 1.0) < 1e-6  # (0 - 1)^2 = 1
    assert abs(total.item() - (pl.item() + vl.item())) < 1e-6
