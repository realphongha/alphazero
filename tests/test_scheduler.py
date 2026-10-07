import torch

from alphazero.training import WarmupCosineScheduler, make_scheduler


def test_warmup_cosine_shape():
    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.0)
    base = 1e-3
    sched = WarmupCosineScheduler(opt, base, total_steps=100, warmup_steps=20, min_lr_frac=0.05)
    # warmup is monotonic increasing to base
    lrs = [sched.get_lr(s)[0] for s in range(21)]
    assert lrs[0] < lrs[10] < lrs[20]
    assert abs(lrs[20] - base) < 1e-9  # peak at end of warmup
    # then cosine decay down to min
    after = [sched.get_lr(s)[0] for s in range(21, 101)]
    assert after[-1] < after[0]
    assert abs(after[-1] - base * 0.05) < 1e-9


def test_step_advances_and_bounded():
    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.0)
    sched = WarmupCosineScheduler(opt, 1e-3, total_steps=50, warmup_steps=5, min_lr_frac=0.1)
    seen = []
    for _ in range(50):
        seen.append(sched.step())
    assert all(0 < v <= 1e-3 for v in seen)
    assert seen[5] >= seen[0]  # ramping during warmup


def test_factory():
    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.0)
    s = make_scheduler("warmup-cosine", opt, 1e-3, 10, 2)
    assert isinstance(s, WarmupCosineScheduler)
