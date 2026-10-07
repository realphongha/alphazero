"""Training data structures: a single (observation, policy, value) sample,
batches, and a shuffled iterable used by the trainer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Tuple

import numpy as np
import torch


@dataclass
class Sample:
    """One supervised training example.

    * ``obs``    : float32 (C, H, W)
    * ``policy`` : float32 (M,)  target distribution over all move slots
    * ``value``  : float32 ()    game result from the mover's perspective
    """
    obs: np.ndarray
    policy: np.ndarray
    value: float

    def __post_init__(self) -> None:
        self.obs = np.asarray(self.obs, dtype=np.float32)
        self.policy = np.asarray(self.policy, dtype=np.float32)
        self.value = float(self.value)


def samples_to_tensors(samples: List[Sample]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    obs = torch.from_numpy(np.stack([s.obs for s in samples], axis=0))
    pol = torch.from_numpy(np.stack([s.policy for s in samples], axis=0))
    val = torch.tensor([s.value for s in samples], dtype=torch.float32)
    return obs, pol, val


def minibatches(
    samples: List[Sample],
    batch_size: int,
    shuffle: bool = True,
    rng: np.random.Generator | None = None,
) -> Iterator[List[Sample]]:
    rng = rng or np.random.default_rng()
    idx = np.arange(len(samples))
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        yield [samples[i] for i in idx[start:start + batch_size]]


def aggregate_loss(samples: List[Sample]) -> dict:
    """Cheap stats for logging (no model): mean |value|, policy entropy, etc."""
    if not samples:
        return {"n": 0}
    vals = np.array([s.value for s in samples])
    return {
        "n": len(samples),
        "mean_value": float(vals.mean()),
        "win_rate_p1": float((vals > 0).mean()),
    }


def save_data(path: str, samples: List[Sample]) -> str:
    """Persist samples to a compressed ``.npz`` (obs, policy, value)."""
    obs = np.stack([s.obs for s in samples], axis=0)
    pol = np.stack([s.policy for s in samples], axis=0)
    val = np.asarray([s.value for s in samples], dtype=np.float32)
    np.savez_compressed(path, obs=obs, policy=pol, value=val)
    return path


def load_data(path: str) -> List[Sample]:
    """Load samples written by :func:`save_data`."""
    with np.load(path) as d:
        obs, pol, val = d["obs"], d["policy"], d["value"]
        return [Sample(obs[i], pol[i], float(val[i])) for i in range(len(val))]


def data_shape(path: str) -> dict:
    """Peek at a ``.npz`` without loading it fully."""
    with np.load(path) as d:
        return {"n": int(d["value"].shape[0]), "obs": list(d["obs"].shape)}
