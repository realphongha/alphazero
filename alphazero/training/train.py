"""PyTorch trainer with the modern (2026) optimisation stack:

* mixed-precision (``torch.amp.autocast``) — bf16 by default on GPU (no scaler
  needed, no fp16 overflow), fp16 with ``GradScaler`` on request, bf16 on CPU;
* AdamW with decoupled weight decay;
* warmup + cosine LR schedule (see :mod:`training.scheduler`);
* gradient clipping;
* optional gradient accumulation;
* optional ``torch.compile``.

Every phase (data / forward / backward / optimizer) is timed through the global
profiler so ``az latency`` can show exactly where training time goes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from ..models.nn import amp_autocast, resolve_amp_dtype
from ..profiling import prof
from .data import Sample, minibatches, samples_to_tensors
from .scheduler import WarmupCosineScheduler


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 1
    amp: str = "auto"            # auto | none | fp16 | bf16
    grad_clip: float = 1.0
    warmup_steps: int = 0        # in *optimizer* steps
    min_lr_frac: float = 0.05
    weight_policy: float = 1.0
    weight_value: float = 1.0
    grad_accum: int = 1
    compile: bool = False
    seed: Optional[int] = None


def _resolve_amp(amp: str, device: torch.device) -> Optional[torch.dtype]:
    amp = (amp or "auto").lower()
    if amp == "auto":
        if device.type == "cuda":
            return torch.bfloat16
        return None
    return resolve_amp_dtype(device, amp)


def loss_fn(
    pred_pol: torch.Tensor,
    target_pol: torch.Tensor,
    pred_val: torch.Tensor,
    target_val: torch.Tensor,
    weight_policy: float = 1.0,
    weight_value: float = 1.0,
):
    logp = torch.log_softmax(pred_pol, dim=1)
    pol_loss = -(target_pol * logp).sum(dim=1).mean()
    val_loss = ((pred_val.reshape(-1) - target_val) ** 2).mean()
    total = weight_policy * pol_loss + weight_value * val_loss
    return total, pol_loss, val_loss


class Trainer:
    def __init__(self, model, device: torch.device, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.device = device
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
            np.random.seed(cfg.seed)
        self.model = model.to(device)
        self.compiled = False
        if cfg.compile:
            try:
                self.model = torch.compile(self.model, mode="default")
                self.compiled = True
            except Exception:
                self.compiled = False
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
        )
        self.amp_dtype = _resolve_amp(cfg.amp, device)
        self.use_scaler = (self.amp_dtype == torch.float16 and device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_scaler)
        self.scheduler: Optional[WarmupCosineScheduler] = None

    # -- one optimiser step ------------------------------------------------
    def _step(self, clip: bool) -> None:
        if self.use_scaler:
            self.scaler.unscale_(self.optimizer)
            if clip and self.cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if clip and self.cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()

    def fit(
        self,
        samples: List[Sample],
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        total_steps: Optional[int] = None,
    ) -> dict:
        cfg = self.cfg
        epochs = epochs or cfg.epochs
        batch_size = batch_size or cfg.batch_size
        micro_bs = max(1, batch_size // max(1, cfg.grad_accum))
        n = len(samples)
        steps_per_epoch = max(1, math.ceil(n / max(1, micro_bs)) // max(1, cfg.grad_accum))
        if total_steps is None:
            total_steps = steps_per_epoch * epochs
        self.scheduler = WarmupCosineScheduler(
            self.optimizer, cfg.lr, total_steps, cfg.warmup_steps, cfg.min_lr_frac,
        )
        self.scheduler.set_initial()  # first optimiser step uses the warmup LR

        self.model.train()
        rng = np.random.default_rng(cfg.seed)
        agg = {"loss": 0.0, "pol_loss": 0.0, "val_loss": 0.0, "steps": 0}
        for epoch in range(epochs):
            with prof.track("train.data_preprocess"):
                micros = list(minibatches(samples, micro_bs, shuffle=True, rng=rng))
            self.optimizer.zero_grad(set_to_none=True)
            accum = 0
            for micro in micros:
                obs, pol, val = samples_to_tensors(micro)
                obs = obs.to(self.device, non_blocking=True)
                pol = pol.to(self.device, non_blocking=True)
                val = val.to(self.device, non_blocking=True)
                with prof.track("train.forward"):
                    with amp_autocast(self.device, self.amp_dtype):
                        p, v = self.model(obs)
                        loss, pl, vl = loss_fn(
                            p, pol, v, val, cfg.weight_policy, cfg.weight_value,
                        )
                        loss = loss / max(1, cfg.grad_accum)
                with prof.track("train.backward"):
                    if self.use_scaler:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                accum += 1
                if accum % max(1, cfg.grad_accum) == 0:
                    with prof.track("train.optimizer_step"):
                        self._step(clip=True)
                        self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)
                    agg["steps"] += 1
                agg["loss"] += float(loss.item())
                agg["pol_loss"] += float(pl.item())
                agg["val_loss"] += float(vl.item())
        if agg["steps"]:
            s = agg["steps"]
            return {
                "loss": agg["loss"] / s,
                "policy_loss": agg["pol_loss"] / s,
                "value_loss": agg["val_loss"] / s,
                "optimizer_steps": agg["steps"],
                "epochs": epochs,
                "amp": str(self.amp_dtype) if self.amp_dtype else "none",
                "compiled": self.compiled,
                "final_lr": self.scheduler.lr,
            }
        return {"loss": float("nan"), "epochs": epochs}
