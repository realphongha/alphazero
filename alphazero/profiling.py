"""Lightweight latency profiler for the "break down latency per step" goal.

A process-wide singleton accumulates ``(count, total_seconds)`` per named phase
so any component can time itself with a context manager::

    from alphazero.profiling import prof
    with prof.track("mcts.search"):
        ...

The CLI can then print a ranked table, dump JSON, or render a bar chart with
matplotlib (optional).  Everything is a no-op when profiling is disabled, so
the overhead is a single ``perf_counter`` call per tracked region.
"""
from __future__ import annotations

import contextlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class _Stat:
    count: int = 0
    total: float = 0.0

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0


class LatencyProfiler:
    def __init__(self, name: str = "profile") -> None:
        self.name = name
        self.enabled = True
        self._stats: Dict[str, _Stat] = defaultdict(_Stat)
        self._epoch: int = 0

    # -- timing -----------------------------------------------------------
    @contextlib.contextmanager
    def track(self, label: str):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.record(label, time.perf_counter() - t0)

    def record(self, label: str, seconds: float) -> None:
        if not self.enabled:
            return
        s = self._stats[label]
        s.count += 1
        s.total += seconds

    def event(self, label: str, seconds: float) -> None:
        self.record(label, seconds)

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for k, s in self._stats.items():
            out[k] = {"count": s.count, "total_s": s.total, "mean_ms": 1e3 * s.mean}
        return out

    # -- aggregation ------------------------------------------------------
    def group_prefix(self, n: int = 1) -> Dict[str, float]:
        """Sum totals into dotted prefixes (``n`` components), for top-down view."""
        agg: Dict[str, float] = defaultdict(float)
        for label, s in self._stats.items():
            prefix = ".".join(label.split(".")[:n])
            agg[prefix] += s.total
        return dict(agg)

    def rank(self, key: str = "total") -> List[Tuple[str, float]]:
        order = {"total": lambda s: s.total, "count": lambda s: s.count,
                 "mean_ms": lambda s: 1e3 * s.mean}
        fn = order.get(key, order["total"])
        return sorted(((k, fn(s)) for k, s in self._stats.items()), key=lambda x: -x[1])

    def reset(self) -> None:
        self._stats.clear()

    def start_epoch(self) -> int:
        self._epoch += 1
        return self._epoch

    # -- reporting --------------------------------------------------------
    def report(self, top: int = 40) -> str:
        lines = []
        if not self._stats:
            return f"[{self.name}] no events recorded"
        lines.append(f"=== {self.name}: latency breakdown (top {top} by total) ===")
        lines.append(f"{'phase':<34} {'count':>8} {'total_ms':>12} {'mean_ms':>10}")
        lines.append("-" * 70)
        for label, total in self.rank("total")[:top]:
            s = self._stats[label]
            lines.append(f"{label:<34} {s.count:>8d} {1e3 * total:>12.3f} {1e3 * s.mean:>10.3f}")
        lines.append("-" * 70)
        grand = sum(s.total for s in self._stats.values())
        lines.append(f"{'TOTAL (all phases)':<34} {sum(s.count for s in self._stats.values()):>8d} {1e3 * grand:>12.3f} {'-':>10}")
        return "\n".join(lines)

    def dump_json(self, path: str) -> str:
        payload = {
            "name": self.name,
            "wall_total_s": sum(s.total for s in self._stats.values()),
            "phases": self.snapshot(),
            "by_prefix_1": {k: 1e3 * v for k, v in self.group_prefix(1).items()},
            "by_prefix_2": {k: 1e3 * v for k, v in self.group_prefix(2).items()},
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        return path

    def plot(self, path: str, top: int = 25) -> Optional[str]:
        """Render a horizontal bar chart via matplotlib (returns path or None)."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return None
        ranked = [x for x in self.rank("total") if x[1] > 0][:top][::-1]
        if not ranked:
            return None
        labels = [r[0] for r in ranked]
        vals_ms = [1e3 * r[1] for r in ranked]
        fig, ax = plt.subplots(figsize=(11, max(4, 0.4 * len(labels) + 1)))
        ax.barh(labels, vals_ms, color="#3a7bd5")
        ax.set_xlabel("total time (ms)")
        ax.set_title(f"{self.name}: latency by phase")
        ax.tick_params(axis="y", labelsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path


# Global singleton used across the codebase.
prof = LatencyProfiler("alphazero")
