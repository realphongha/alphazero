"""Learning-rate schedules (2026-flavoured: warmup + cosine decay)."""
from __future__ import annotations

import math
from typing import List

import torch


class WarmupCosineScheduler:
    """Linear warmup to the peak LR, then cosine decay to ``min_lr_frac``.

    The peak LR is ``base_lr`` (per parameter group, scaled by each group's
    relative weight).  During warmup the LR ramps *positively* from the first
    step (``(step+1)/warmup``) so the first optimiser step is never zero, and
    peaks exactly at ``base_lr``; after warmup it decays smoothly to
    ``min_lr_frac * base_lr`` by ``total_steps``.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        total_steps: int,
        warmup_steps: int = 0,
        min_lr_frac: float = 0.05,
        start_lr_frac: float = 0.0,
    ) -> None:
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = int(warmup_steps)
        self.min_lr_frac = float(min_lr_frac)
        self.start_lr_frac = float(start_lr_frac)
        # Per-group relative scale (default 1.0).  Captured from the optimiser's
        # initial LRs; if they are all zero (common when the scheduler owns the
        # LR), every group is treated as equal.
        init = [g["lr"] for g in optimizer.param_groups]
        ref = sum(init) / len(init) if init else 0.0
        self.scales = [(lr / ref) if ref > 0 else 1.0 for lr in init]
        self.step_count = 0

    def _coefficient(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            frac = (step + 1) / self.warmup_steps
            return self.start_lr_frac + (1.0 - self.start_lr_frac) * frac
        remaining = self.total_steps - self.warmup_steps
        x = (step - self.warmup_steps) / max(1, remaining)
        x = min(max(x, 0.0), 1.0)
        return self.min_lr_frac + 0.5 * (1.0 - self.min_lr_frac) * (1.0 + math.cos(math.pi * x))

    def get_lr(self, step: int) -> List[float]:
        c = self._coefficient(step)
        return [self.base_lr * s * c for s in self.scales]

    def step(self) -> float:
        lrs = self.get_lr(self.step_count)
        for g, lr in zip(self.optimizer.param_groups, lrs):
            g["lr"] = lr
        self.step_count += 1
        return lrs[0]

    def set_initial(self) -> None:
        """Set the optimiser LR to the value for step 0 (call before training)."""
        for g, lr in zip(self.optimizer.param_groups, self.get_lr(0)):
            g["lr"] = lr

    @property
    def lr(self) -> float:
        return self.get_lr(self.step_count)[0]


def make_scheduler(
    name: str,
    optimizer: torch.optim.Optimizer,
    base_lr: float,
    total_steps: int,
    warmup_steps: int = 0,
    min_lr_frac: float = 0.05,
) -> WarmupCosineScheduler:
    """Tiny factory; 'warmup-cosine' is the only (default) schedule."""
    name = (name or "warmup-cosine").lower()
    if name in ("warmup-cosine", "cosine", "warmup_cosine"):
        return WarmupCosineScheduler(optimizer, base_lr, total_steps, warmup_steps, min_lr_frac)
    raise ValueError(f"unknown scheduler '{name}'")
