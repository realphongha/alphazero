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


# ---------------------------------------------------------------------------
# D4 (square-board) symmetry augmentation  [the paper uses 8 symmetries for Go]
# ---------------------------------------------------------------------------
_D4_PERMS = {}


def _d4_cell_perms(n: int) -> List[np.ndarray]:
    """8 cell permutations of the D4 group.  ``perm[dest] = source`` so that
    ``arr[perm]`` warps an image/policy by the transform (content at source moves
    to dest)."""
    if n in _D4_PERMS:
        return _D4_PERMS[n]

    def idx_of(f):
        idx = np.zeros(n * n, dtype=np.int64)
        for r in range(n):
            for c in range(n):
                rp, cp = f(r, c)
                idx[rp * n + cp] = r * n + c
        return idx

    fns = (
        lambda r, c: (r, c),                 # identity
        lambda r, c: (c, n - 1 - r),         # rot90
        lambda r, c: (n - 1 - r, n - 1 - c), # rot180
        lambda r, c: (n - 1 - c, r),         # rot270
        lambda r, c: (r, n - 1 - c),         # flip left-right
        lambda r, c: (n - 1 - r, c),         # flip top-bottom
        lambda r, c: (c, r),                 # flip main diagonal
        lambda r, c: (n - 1 - c, n - 1 - r), # flip anti-diagonal
    )
    perms = [idx_of(f) for f in fns]
    _D4_PERMS[n] = perms
    return perms


def augment_d4(obs: np.ndarray, policy: np.ndarray):
    """Yield the 8 D4-transformed ``(obs, policy)`` pairs (value is unchanged)."""
    n = obs.shape[1]  # obs is (C, N, N); the board dimension is obs.shape[1]
    perms = _d4_cell_perms(n)
    o2 = obs.reshape(obs.shape[0], -1)
    p2 = policy.reshape(-1)
    for perm in perms:
        yield o2[:, perm].reshape(obs.shape), p2[perm].reshape(policy.shape)


def augment_samples(samples: List[Sample]) -> List[Sample]:
    """Expand ``samples`` 8x under the D4 symmetry group (value preserved)."""
    out: List[Sample] = []
    for s in samples:
        for no, np_ in augment_d4(s.obs, s.policy):
            out.append(Sample(no, np_, s.value))
    return out
