"""Policy/value providers: the pluggable "brain" behind MCTS.

``PolicyProvider.forward_batch(states)`` returns ``(priors, values)`` for a list
of states, so the *same* MCTS works with:

* ``NeuralPolicyProvider``  — a ZeroNet, one batched forward pass per round
  (this is where "batch inference for AlphaZero nodes" lives).
* ``RandomPolicyProvider``  — uniform prior + random rollouts to the end, i.e.
  classical / *pure* MCTS with no learned model (used for a model-less bot).

Both return numpy arrays so MCTS never has to touch torch directly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING

import numpy as np

from .games.base import Game
from .profiling import prof

if TYPE_CHECKING:  # type hints only -- never imported at runtime (keeps this torch-free)
    import torch
    from .models.nn import ZeroNet


class PolicyProvider(ABC):
    def __init__(self, game: Game) -> None:
        self.game = game

    @abstractmethod
    def forward_batch(self, states: List[Game]) -> (np.ndarray, np.ndarray):
        """Return ``(priors (B, M), values (B,))`` for ``states``."""

    def describe(self) -> str:
        return type(self).__name__


class NeuralPolicyProvider(PolicyProvider):
    def __init__(
        self,
        model: ZeroNet,
        amp_dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(game=None)  # model already carries the shape
        self.model = model
        self.amp_dtype = amp_dtype

    def forward_batch(self, states: List[Game]):
        with prof.track("infer.model_forward_batch"):
            priors, values = self.model.predict_batch(states, amp_dtype=self.amp_dtype)
        return priors, values

    def describe(self) -> str:
        return f"NeuralPolicyProvider({type(self.model).__name__})"


class RandomPolicyProvider(PolicyProvider):
    """Pure MCTS: uniform prior, value = average of ``num_playouts`` random games."""

    def __init__(self, game: Game, num_playouts: int = 16) -> None:
        super().__init__(game)
        self.num_playouts = num_playouts
        self._rng = np.random.default_rng()

    def forward_batch(self, states: List[Game]):
        M = self.game.num_moves()
        B = len(states)
        priors = np.full((B, M), 1.0 / M, dtype=np.float32)
        values = np.zeros(B, dtype=np.float32)
        with prof.track("infer.random_playouts"):
            for i, st in enumerate(states):
                acc = 0.0
                for _ in range(self.num_playouts):
                    g = st
                    while not g.is_terminal():
                        legal = g.legal_moves()
                        if len(legal) == 0:
                            break
                        g = g.apply(int(self._rng.choice(legal)))
                    acc += g.reward(st.current_player())
                values[i] = acc / max(1, self.num_playouts)
        return priors, values

    def describe(self) -> str:
        return f"RandomPolicyProvider(playouts={self.num_playouts})"


class OnnxPolicyProvider(PolicyProvider):
    """ONNX Runtime (CPU) inference for an exported ZeroNet -- no torch needed.

    Runs the exported raw ``forward`` (policy logits + value) and applies the
    same legal-move masking + softmax as ``ZeroNet.predict_batch``.
    """

    def __init__(self, onnx_path: str, game: Optional[Game] = None, num_threads: int = 0) -> None:
        try:
            import onnxruntime as ort
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "onnxruntime is not installed. Install it with:  "
                "pip install onnxruntime   (or: uv sync --extra onnx)") from e
        super().__init__(game)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if num_threads and num_threads > 0:
            opts.intra_op_num_threads = num_threads
            opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.num_threads = num_threads

    def forward_batch(self, states: List[Game]):
        with prof.track("infer.onnx_forward"):
            obs = np.stack([s.observation() for s in states], axis=0).astype(np.float32)
            out = self.session.run(None, {self.input_name: obs})
            pol, val = out[0], out[1]
            legal = obs[:, 2].reshape(len(states), -1)
            masked = np.where(legal == 0, -1e9, pol.astype(np.float32))
            masked = masked - masked.max(axis=1, keepdims=True)
            e = np.exp(masked)
            priors = e / e.sum(axis=1, keepdims=True)
            return priors.astype(np.float32), val.reshape(-1).astype(np.float32)

    def describe(self) -> str:
        t = f", threads={self.num_threads}" if self.num_threads else ""
        return f"OnnxPolicyProvider(CPU{t})"


def make_provider(
    kind: str,
    game: Game,
    *,
    model: Optional["ZeroNet"] = None,
    amp_dtype: Optional["torch.dtype"] = None,
    num_playouts: int = 16,
    onnx_path: Optional[str] = None,
    onnx_threads: int = 0,
) -> PolicyProvider:
    """Build a provider: 'model' (ZeroNet), 'onnx' (ONNX CPU), or 'random'."""
    kind = (kind or "random").lower()
    if kind in ("model", "neural", "net"):
        if model is None:
            raise ValueError("kind='model' requires a ZeroNet ``model``")
        return NeuralPolicyProvider(model, amp_dtype=amp_dtype)
    if kind in ("onnx", "onnxruntime", "ort"):
        if not onnx_path:
            raise ValueError("kind='onnx' requires onnx_path")
        return OnnxPolicyProvider(onnx_path, game=game, num_threads=onnx_threads)
    if kind in ("random", "mcts", "pure", "playout"):
        return RandomPolicyProvider(game, num_playouts=num_playouts)
    raise ValueError(f"unknown provider kind '{kind}' (use 'model', 'onnx', or 'random')")
