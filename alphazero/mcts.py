"""MCTS: an abstract base + a batched AlphaZero concrete implementation.

``MCTS`` (ABC) pins the contract (``search(root) -> policy``).  ``AlphaZeroMCTS``
implements it with the classic PUCT formula and, crucially, *batched* model
evaluation: each search round collects up to ``batch_size`` leaves and asks the
``PolicyProvider`` for all of their priors/values in one call, so the GPU never
sits idle on tiny forwards.  Node children are stored in numpy arrays so child
selection is a single vectorised PUCT + ``argmax``.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .games.base import Game
from .inference import PolicyProvider
from .profiling import prof


@dataclass
class MCTSConfig:
    num_simulations: int = 200
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.03
    dirichlet_epsilon: float = 0.25
    batch_size: int = 16
    seed: Optional[int] = None


def _terminal_value(game: Game) -> float:
    """Value from the perspective of the player to move at a terminal state."""
    w = game.winner()
    if w == 0:
        return 0.0
    return 1.0 if w == game.current_player() else -1.0


class _Node:
    __slots__ = (
        "parent", "game", "children", "moves", "priors", "child_Q", "child_N",
        "Q", "N", "terminal", "terminal_value", "expanded", "_idx",
    )

    def __init__(self, parent, game: Game, idx: int) -> None:
        self.parent = parent
        self.game = game
        self.children: List["_Node"] = []
        self.moves: Optional[np.ndarray] = None
        self.priors: Optional[np.ndarray] = None
        self.child_Q: Optional[np.ndarray] = None
        self.child_N: Optional[np.ndarray] = None
        self._idx = idx
        self.Q = 0.0
        self.N = 0
        self.terminal = game.is_terminal()
        self.terminal_value = _terminal_value(game) if self.terminal else 0.0
        self.expanded = False


class MCTS(ABC):
    """Abstract Monte-Carlo tree search."""

    @abstractmethod
    def search(self, root: Game) -> np.ndarray:
        """Run a full search from ``root`` and return a policy (len = num_moves)."""

    @abstractmethod
    def describe(self) -> str:
        ...


class AlphaZeroMCTS(MCTS):
    def __init__(self, game: Game, provider: PolicyProvider, cfg: MCTSConfig) -> None:
        self.game = game
        self.provider = provider
        self.cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)

    def describe(self) -> str:
        c = self.cfg
        return (f"AlphaZeroMCTS(sims={c.num_simulations}, puct={c.c_puct}, "
                f"batch={c.batch_size}, provider={self.provider.describe()})")

    # -- selection --------------------------------------------------------
    def _select(self, node: _Node) -> _Node:
        P = node.priors
        Qc = node.child_Q
        Nc = node.child_N
        safe_n = np.where(Nc > 0, Nc, 1)
        # Q is stored from the *child's* to-move perspective, which is the
        # opponent of the player choosing here, so we negate the value term:
        # the current player wants the child's value to be as negative as
        # possible for the opponent.  The exploration term uses the prior from
        # the current player's own perspective (positive).
        val = np.where(Nc > 0, -Qc / safe_n, 0.0)
        scores = val + self.cfg.c_puct * P * np.sqrt(node.N + 1) / (1 + Nc)
        idx = int(np.argmax(scores))
        return node.children[idx]

    # -- expansion --------------------------------------------------------
    def _expand(self, node: _Node, prior_vec: np.ndarray, noise: Optional[np.ndarray],
                is_root: bool) -> None:
        g = node.game
        legal = g.legal_moves()
        moves = []
        priors = []
        kids = []
        for j, m in enumerate(legal):
            mi = int(m)
            p = float(prior_vec[mi])
            if is_root:
                p = (1.0 - self.cfg.dirichlet_epsilon) * p + self.cfg.dirichlet_epsilon * float(noise[mi])
            moves.append(mi)
            priors.append(p)
            kids.append(g.apply(mi))
        node.moves = np.asarray(moves, dtype=np.int64)
        node.priors = np.asarray(priors, dtype=np.float32)
        node.child_Q = np.zeros(len(kids), dtype=np.float64)
        node.child_N = np.zeros(len(kids), dtype=np.int64)
        node.children = [_Node(node, kg, i) for i, kg in enumerate(kids)]
        node.expanded = True

    # -- backup -----------------------------------------------------------
    def _backup(self, node: _Node, value: float, count: int = 1) -> None:
        v = value
        while node is not None:
            node.Q += v * count
            node.N += count
            if node.parent is not None:
                node.parent.child_Q[node._idx] = node.Q
                node.parent.child_N[node._idx] = node.N
            v = -v
            node = node.parent

    def _walk_to_leaf(self, root: _Node) -> _Node:
        node = root
        while node.expanded and not node.terminal:
            node = self._select(node)
        return node

    # -- search -----------------------------------------------------------
    def search(self, root_game: Game) -> np.ndarray:
        cfg = self.cfg
        M = root_game.num_moves()
        root = _Node(None, root_game, -1)
        if root.terminal:
            return np.zeros(M, dtype=np.float32)

        noise = self._rng.dirichlet(np.full(M, cfg.dirichlet_alpha)).astype(np.float32)

        # Pre-expand the root (one small batch) so every walk descends into the
        # tree; this keeps the policy a clean distribution over root's moves.
        t0 = time.perf_counter()
        p0, v0 = self.provider.forward_batch([root_game])
        self._expand(root, p0[0], noise, is_root=True)
        self._backup(root, float(v0[0]), count=1)

        t_select = t_expand = t_backup = 0.0
        pending: List[_Node] = []

        def flush() -> None:
            if not pending:
                return
            counts: dict = {}
            for n in pending:
                counts[id(n)] = counts.get(id(n), 0) + 1
            uniq_nodes = []
            seen_ids = set()
            for n in pending:
                if id(n) not in seen_ids:
                    seen_ids.add(id(n))
                    uniq_nodes.append(n)
            priors, values = self.provider.forward_batch([n.game for n in uniq_nodes])
            for node, p, v in zip(uniq_nodes, priors, values):
                if not node.expanded:
                    self._expand(node, p, noise, is_root=False)
                self._backup(node, float(v), count=counts[id(node)])
            pending.clear()

        n_rounds, remainder = divmod(cfg.num_simulations, cfg.batch_size)
        for _ in range(n_rounds):
            t1 = time.perf_counter()
            for _ in range(cfg.batch_size):
                node = self._walk_to_leaf(root)
                if node.terminal:
                    self._backup(node, node.terminal_value, count=1)
                else:
                    pending.append(node)
            t_select += time.perf_counter() - t1
            t2 = time.perf_counter()
            flush()
            t_expand += time.perf_counter() - t2
        for _ in range(remainder):
            t1 = time.perf_counter()
            node = self._walk_to_leaf(root)
            if node.terminal:
                self._backup(node, node.terminal_value, count=1)
            else:
                pending.append(node)
            t_select += time.perf_counter() - t1
        t3 = time.perf_counter()
        flush()
        t_backup += time.perf_counter() - t3

        prof.record("mcts.select", t_select)
        prof.record("mcts.expand", t_expand)
        prof.record("mcts.backup", t_backup)

        self.last_root_value = (root.Q / root.N) if root.N else 0.0
        total = float(root.child_N.sum())
        policy = np.zeros(M, dtype=np.float32)
        if total > 0:
            for i, m in enumerate(root.moves):
                policy[int(m)] = root.child_N[i] / total
        else:  # pragma: no cover - safety net
            legal = root_game.legal_moves()
            if len(legal):
                policy[legal] = 1.0 / len(legal)
        return policy

    def best_action(self, root_game: Game) -> int:
        pol = self.search(root_game)
        legal = root_game.legal_moves()
        if len(legal) == 0:
            return -1
        return int(legal[int(np.argmax(pol[legal]))])
