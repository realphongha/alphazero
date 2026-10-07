"""Shared building blocks for the neural Zero networks.

Contains the small cross-cutting helpers (device selection, observation
stacking, an autocast context) plus the ``ZeroNet`` base class that defines the
``predict_batch`` contract both MCTS (inference) and self-play rely on.
"""
from __future__ import annotations

import contextlib
from typing import List, Optional

import numpy as np
import torch
from torch import nn


def get_device(preference: Optional[str] = None) -> torch.device:
    """Resolve a torch device. ``preference`` in {'auto','cuda','cpu'}.

    Default ('auto') uses CUDA when available, else CPU.
    """
    preference = (preference or "auto").lower()
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("cuda requested but unavailable")
        return torch.device("cuda")
    if not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def resolve_amp_dtype(device: torch.device, dtype_name: str) -> Optional[torch.dtype]:
    """Map an AMP name to a torch dtype (None disables mixed precision)."""
    dtype_name = (dtype_name or "none").lower()
    if dtype_name in ("none", "off", "no"):
        return None
    if dtype_name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype_name in ("fp16", "half", "float16"):
        return torch.float16
    raise ValueError(f"unknown AMP dtype '{dtype_name}'")


@contextlib.contextmanager
def amp_autocast(device: torch.device, amp_dtype: Optional[torch.dtype]):
    """Return an autocast context (or a no-op) for the given device/dtype."""
    if amp_dtype is None:
        yield
        return
    try:
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
            yield
    except (AssertionError, RuntimeError):
        # Fall back to a plain forward if this build/device can't autocast.
        yield


def stack_obs(states: List) -> torch.Tensor:
    """Stack a list of games' ``(C,H,W)`` obs into a single ``(B,C,H,W)`` tensor."""
    return torch.from_numpy(np.stack([s.observation() for s in states], axis=0))


class ZeroNet(nn.Module):
    """Base class: turns a batched board into policy logits + a scalar value.

    Subclasses implement :meth:`forward`.  :meth:`predict_batch` is the single
    entry-point used by the inference providers and MCTS: it converts a list of
    game states into model outputs in *one* batched forward pass (the
    "batch inference for AlphaZero nodes" requirement), masks the policy to
    legal moves, and returns plain numpy arrays.
    """

    obs_channels: int = 3

    def __init__(self, num_moves: int, device: torch.device) -> None:
        super().__init__()
        self.num_moves = num_moves
        self.device = device

    def _finalize(self, memory_format: bool = False) -> None:
        """Move to device (after submodules exist), set memory format, eval."""
        self.to(self.device)
        if memory_format and self.device.type == "cuda":
            self.to(memory_format=torch.channels_last)
        self.eval()

    def forward(self, obs: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        raise NotImplementedError

    @torch.no_grad()
    def predict_batch(self, states: List, amp_dtype: Optional[torch.dtype] = None) -> (np.ndarray, np.ndarray):
        obs = stack_obs(states)
        B = obs.shape[0]
        with amp_autocast(self.device, amp_dtype):
            obs_d = obs.to(self.device, non_blocking=True)
            pol_logits, val = self(obs_d)
            legal = obs_d[:, 2].reshape(B, self.num_moves)
            masked = pol_logits.masked_fill(legal == 0, -1e9)
            priors = torch.softmax(masked, dim=1)
        # autocast may leave bf16/fp16 outputs; cast to fp32 before numpy.
        return priors.detach().float().cpu().numpy(), val.detach().float().cpu().numpy().reshape(-1)
