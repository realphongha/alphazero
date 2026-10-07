"""Training package: data (torch-free) + the PyTorch scheduler/trainer (lazy).

The pure-data helpers (``Sample``, batching, persistence, D4 augmentation) are
imported eagerly and require only numpy.  The torch-dependent symbols
(``Trainer``, ``TrainConfig``, ``loss_fn``, the scheduler) are loaded lazily
(PEP 562) so importing the data module does **not** require PyTorch -- this is
what lets the inference-only path run without torch.
"""
from __future__ import annotations

import importlib

from .data import (
    Sample, minibatches, samples_to_tensors, aggregate_loss, save_data, load_data,
    data_shape, augment_d4, augment_samples,
)

_LAZY = {
    "WarmupCosineScheduler": (".scheduler", "WarmupCosineScheduler"),
    "make_scheduler": (".scheduler", "make_scheduler"),
    "Trainer": (".train", "Trainer"),
    "TrainConfig": (".train", "TrainConfig"),
    "loss_fn": (".train", "loss_fn"),
}


def __getattr__(name):
    if name in _LAZY:
        module_name, attr = _LAZY[name]
        return getattr(importlib.import_module(module_name, __name__), attr)
    raise AttributeError(f"module 'alphazero.training' has no attribute {name!r}")


__all__ = [
    "Sample", "minibatches", "samples_to_tensors", "aggregate_loss",
    "save_data", "load_data", "data_shape", "augment_d4", "augment_samples",
    "WarmupCosineScheduler", "make_scheduler",
    "Trainer", "TrainConfig", "loss_fn",
]
