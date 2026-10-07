"""Full AlphaZero training loop: bootstrap -> (self-play -> train -> evaluate)*.

This is the "full training" driver.  It:

* bootstraps a replay buffer from strong pure-MCTS self-play (no model needed),
* then iterates: self-play with the *current* network -> augment (D4) -> train
  for a few epochs -> evaluate the network against a random and a strong
  pure-MCTS opponent,
* keeps the best network (by head-to-head winrate vs the pure-MCTS baseline) and
  a latest network, and logs every iteration to a JSONL file.

The pure-MCTS baseline doubles as the strength target: a network that does not
lose to it is "at least same level as pure MCTS".
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np
import torch

from .checkpoint import ModelMeta, save_model
from .games import make_game
from .inference import NeuralPolicyProvider, RandomPolicyProvider
from .mcts import AlphaZeroMCTS, MCTSConfig
from .models import build_model
from .models.nn import get_device, resolve_amp_dtype
from .profiling import prof
from .selfplay import SelfplayConfig, run_selfplay
from .training.data import Sample, augment_samples
from .training.train import TrainConfig, Trainer


@dataclass
class LoopConfig:
    # game + hardware
    game_name: str = "tictactoe"
    device: str = "auto"
    seed: Optional[int] = None
    # architecture
    backbone: str = "resnet"
    arch: Dict = field(default_factory=lambda: {"channels": 64, "blocks": 3})
    # self-play
    sims: int = 200
    mcts_batch: int = 16
    c_puct: float = 1.5
    playouts: int = 32          # pure-MCTS playouts (bootstrap + baseline)
    # bootstrap (pure MCTS)
    bootstrap_games: int = 40
    bootstrap_sims: int = 300
    # self-improvement loop
    iterations: int = 40
    games_per_iter: int = 20
    epochs_per_iter: int = 2
    # training
    lr: float = 2e-3
    warmup_steps: int = 20
    batch_size: int = 256
    amp: str = "auto"
    weight_decay: float = 1e-4
    buffer_size: int = 20000    # replay buffer cap (augmented samples)
    augment: bool = True
    # evaluation
    eval_every: int = 5
    eval_games: int = 40
    eval_sims: int = 120        # network's search budget when evaluating
    baseline_sims: int = 300    # strong pure-MCTS baseline search budget
    # challenger-vs-best (self-play ladder) acceptance gate
    eval_best_every: int = 0    # 0 = off (legacy vs_pure-score selection); else every N iters
    best_eval_games: int = 12   # games in the challenger-vs-best match
    promote_winrate: float = 0.5  # promote if challenger winrate vs best > this
    eval_best_pure_ref: bool = False  # also run a (CPU-heavy) vs-pure reference in the gate
    # outputs
    out_dir: str = "runs"
    start_model: Optional[str] = None
    profile: bool = False
    # wandb (opt-in; needs the 'wandb' extra:  uv sync --extra wandb)
    wandb: bool = False
    wandb_project: str = "alphazero"
    wandb_run_name: Optional[str] = None
    wandb_offline: bool = False
    wandb_tags: tuple = ()


class AlphaZeroLoop:
    def __init__(self, cfg: LoopConfig) -> None:
        self.cfg = cfg
        self.device = get_device(cfg.device)
        self.amp_dtype = resolve_amp_dtype(self.device, cfg.amp) if cfg.amp != "auto" else None
        if cfg.amp == "auto":
            self.amp_dtype = torch.bfloat16 if self.device.type == "cuda" else None
        self.game = make_game(cfg.game_name)
        self.model = build_model(self.game, cfg.backbone, self.device, cfg=cfg.arch)
        if cfg.start_model:
            from .checkpoint import load_model
            self.model = load_model(cfg.start_model, self.device)
        self.provider = NeuralPolicyProvider(self.model, self.amp_dtype)
        self.buffer: List[Sample] = []
        os.makedirs(cfg.out_dir, exist_ok=True)
        self.log_path = os.path.join(cfg.out_dir, "train.jsonl")
        self._t_start = time.time()
        self._wandb = None
        self._wandb_module = None
        self.champion_model = None
        self.champion_provider = None

    # ------------------------------------------------------------------
    def _selfplay_mcts(self, sims: int, seed: Optional[int] = None) -> AlphaZeroMCTS:
        return AlphaZeroMCTS(
            self.game, self.provider,
            MCTSConfig(num_simulations=sims, c_puct=self.cfg.c_puct,
                       batch_size=self.cfg.mcts_batch,
                       seed=seed if seed is not None else self.cfg.seed),
        )

    def _baseline_mcts(self, sims: int) -> AlphaZeroMCTS:
        return AlphaZeroMCTS(
            self.game, RandomPolicyProvider(self.game, self.cfg.playouts),
            MCTSConfig(num_simulations=sims, c_puct=self.cfg.c_puct,
                       batch_size=max(4, self.cfg.mcts_batch // 2), seed=self.cfg.seed),
        )

    # ------------------------------------------------------------------
    def bootstrap(self) -> None:
        cfg = self.cfg
        print(f"[bootstrap] {cfg.bootstrap_games} pure-MCTS games "
              f"({cfg.bootstrap_sims} sims, {cfg.playouts} playouts) ...", flush=True)
        t0 = time.time()
        mcts = self._baseline_mcts(cfg.bootstrap_sims)
        samples = run_selfplay(self.game.copy, mcts, cfg.bootstrap_games, seed=self.cfg.seed)
        self._ingest(samples)
        # quick initial train so the network is not random before iteration 1
        self._train(2)
        print(f"[bootstrap] {len(samples)} positions ({time.time()-t0:.1f}s) "
              f"-> buffer {len(self.buffer)}", flush=True)

    def _ingest(self, samples: List[Sample]) -> None:
        if self.cfg.augment:
            samples = augment_samples(samples)
        self.buffer.extend(samples)
        if self.cfg.buffer_size and len(self.buffer) > self.cfg.buffer_size:
            self.buffer = self.buffer[-self.cfg.buffer_size:]

    def _train(self, epochs: int) -> dict:
        trainer = Trainer(
            self.model, self.device,
            TrainConfig(lr=self.cfg.lr, batch_size=self.cfg.batch_size, epochs=epochs,
                        amp=str(self.cfg.amp), warmup_steps=self.cfg.warmup_steps,
                        weight_decay=self.cfg.weight_decay, seed=self.cfg.seed),
        )
        return trainer.fit(self.buffer, epochs=epochs)

    # ------------------------------------------------------------------
    def _opponent_move(self, kind: str, state, baseline, rng: np.random.Generator) -> int:
        if kind == "random":
            legal = state.legal_moves()
            return int(rng.choice(legal))
        return baseline.best_action(state)

    def _eval_vs(self, model_mcts: AlphaZeroMCTS, num_games: int, kind: str) -> Dict[str, int]:
        """``model_mcts`` (player 1) vs a 'random' or 'pure' MCTS opponent (player 2)."""
        baseline = self._baseline_mcts(self.cfg.baseline_sims) if kind == "pure" else None
        rng = np.random.default_rng(self.cfg.seed)
        r = {"win": 0, "draw": 0, "loss": 0}
        t0 = time.time()
        for _ in range(num_games):
            state = self.game.copy()
            while not state.is_terminal():
                if state.current_player() == 1:
                    mv = model_mcts.best_action(state)
                else:
                    mv = self._opponent_move(kind, state, baseline, rng)
                if mv < 0:
                    break
                state = state.apply(mv)
            r[{1: "win", 0: "draw", 2: "loss"}[state.winner()]] += 1
        r["time_s"] = round(time.time() - t0, 2)  # type: ignore[index]
        return r

    def evaluate(self, num_games: int) -> Dict[str, Dict[int, int]]:
        """Model (player 1) vs {random, pure-MCTS}.  Returns win/draw/loss counts."""
        model_mcts = self._selfplay_mcts(self.cfg.eval_sims)
        res = {"vs_random": self._eval_vs(model_mcts, num_games, "random"),
               "vs_pure": self._eval_vs(model_mcts, num_games, "pure")}
        res["eval_time_s"] = round(res["vs_random"]["time_s"] + res["vs_pure"]["time_s"], 2)
        return res

    def _score(self, ev: Dict) -> float:
        """Legacy best-selection score: wins vs pure MCTS - losses vs pure MCTS."""
        vs = ev["vs_pure"]
        return float(vs["win"]) - float(vs["loss"])

    # ---- challenger-vs-best (self-play ladder) acceptance --------------
    def _init_champion(self) -> None:
        """Seed the champion (current best) with the model's weights at loop start."""
        self.champion_model = build_model(self.game, self.cfg.backbone, self.device,
                                          cfg=self.cfg.arch)
        self.champion_model.load_state_dict(
            {k: v.detach().clone() for k, v in self.model.state_dict().items()})
        self.champion_model.eval()
        self.champion_provider = NeuralPolicyProvider(self.champion_model, self.amp_dtype)
        self._save("best", model=self.champion_model)

    def _best_mcts(self, sims: int) -> AlphaZeroMCTS:
        return AlphaZeroMCTS(
            self.game, self.champion_provider,
            MCTSConfig(num_simulations=sims, c_puct=self.cfg.c_puct,
                       batch_size=self.cfg.mcts_batch, seed=self.cfg.seed),
        )

    def _evaluate_best(self, num_games: int) -> Dict:
        """Current model (challenger) vs current best (champion); sides alternate.

        Returns challenger win/draw/loss, a winrate (wins / decisive), and a
        pure-MCTS reference so the fixed-baseline trend is still visible.
        """
        chall = self._selfplay_mcts(self.cfg.eval_sims)
        champ = self._best_mcts(self.cfg.eval_sims)
        c = {"win": 0, "draw": 0, "loss": 0}
        t0 = time.time()
        for g in range(num_games):
            state = self.game.copy()
            chall_p1 = (g % 2 == 0)
            while not state.is_terminal():
                p = state.current_player()
                if chall_p1:
                    mv = chall.best_action(state) if p == 1 else champ.best_action(state)
                else:
                    mv = champ.best_action(state) if p == 1 else chall.best_action(state)
                if mv < 0:
                    break
                state = state.apply(mv)
            w = state.winner()
            if w == 0:
                c["draw"] += 1
            elif w == (1 if chall_p1 else 2):
                c["win"] += 1
            else:
                c["loss"] += 1
        decisive = c["win"] + c["loss"]
        c["winrate_vs_best"] = round(c["win"] / decisive, 3) if decisive else 0.5
        c["time_s"] = round(time.time() - t0, 2)
        if self.cfg.eval_best_pure_ref:      # optional (CPU-heavy) pure-MCTS reference
            c["vs_pure"] = self._eval_vs(chall, self.cfg.eval_games, "pure")
        return c

    def _promote(self) -> None:
        """Champion <- current model; persist the new best."""
        self.champion_model.load_state_dict(
            {k: v.detach().clone() for k, v in self.model.state_dict().items()})
        self._save("best", model=self.champion_model)

    def _save(self, tag: str, model=None) -> str:
        model = model or self.model
        meta = ModelMeta(backbone=self.cfg.backbone, n=self.game.n, m=self.game.m,
                         num_moves=self.game.num_moves(), arch=self.cfg.arch)
        return save_model(model, os.path.join(self.cfg.out_dir, f"{tag}.pt"), meta)

    def _log(self, row: Dict) -> None:
        with open(self.log_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    # ---- wandb (opt-in experiment logging) ----------------------------
    def _wandb_init(self) -> None:
        if not self.cfg.wandb:
            return
        try:
            import wandb
        except ImportError:
            raise RuntimeError(
                "wandb is enabled (--wandb) but not installed. Install with:  "
                "uv sync --extra wandb   (or: pip install wandb)")
        name = self.cfg.wandb_run_name or (
            f"{self.cfg.game_name}_{self.cfg.backbone}_sims{self.cfg.sims}_{int(time.time())}")
        try:
            self._wandb = wandb.init(
                project=self.cfg.wandb_project, name=name,
                tags=list(self.cfg.wandb_tags) or None,
                mode="offline" if self.cfg.wandb_offline else "online",
                dir=self.cfg.out_dir,
            )
            self._wandb_module = wandb
        except Exception as e:  # never let wandb kill a long run
            print(f"[wandb] init failed ({e}); continuing without wandb", flush=True)
            self._wandb = None
            return
        self._wandb.config.update({
            "game": self.cfg.game_name, "backbone": self.cfg.backbone, "arch": self.cfg.arch,
            "device": str(self.device), "amp": self.cfg.amp, "iterations": self.cfg.iterations,
            "games_per_iter": self.cfg.games_per_iter, "sims": self.cfg.sims,
            "epochs_per_iter": self.cfg.epochs_per_iter, "lr": self.cfg.lr,
            "batch_size": self.cfg.batch_size, "buffer_size": self.cfg.buffer_size,
            "augment": self.cfg.augment, "c_puct": self.cfg.c_puct,
            "bootstrap_games": self.cfg.bootstrap_games, "bootstrap_sims": self.cfg.bootstrap_sims,
            "eval_games": self.cfg.eval_games, "eval_sims": self.cfg.eval_sims,
            "baseline_sims": self.cfg.baseline_sims, "seed": self.cfg.seed,
        })
        print(f"[wandb] logging -> project '{self.cfg.wandb_project}' run '{name}' "
              f"({'offline' if self.cfg.wandb_offline else 'online'})", flush=True)

    def _wandb_log(self, row: Dict) -> None:
        if self._wandb is None:
            return
        try:
            d: Dict = {"iter": row.get("iter", -1)}
            for k in ("loss", "policy_loss", "value_loss", "final_lr", "buffer"):
                if k in row and row[k] is not None:
                    d[f"train/{k}"] = row[k]
            ev = row.get("eval")
            if ev:
                for key in ("vs_random", "vs_pure"):
                    e = ev[key]
                    tot = e["win"] + e["draw"] + e["loss"]
                    d[f"eval/{key}_win_rate"] = e["win"] / tot if tot else 0.0
                    d[f"eval/{key}_loss_rate"] = e["loss"] / tot if tot else 0.0
                    d[f"eval/{key}_draw_rate"] = e["draw"] / tot if tot else 0.0
                    d[f"eval/{key}_score"] = float(e["win"] - e["loss"])
                    d[f"eval/{key}_games"] = tot
                if "score" in row:
                    d["score"] = row["score"]
            evb = row.get("eval_best")
            if evb:
                d["eval/winrate_vs_best"] = evb.get("winrate_vs_best", 0.0)
                d["eval/challenger_win"] = evb.get("win", 0)
                d["eval/challenger_draw"] = evb.get("draw", 0)
                d["eval/challenger_loss"] = evb.get("loss", 0)
                if "vs_pure" in evb:
                    e = evb["vs_pure"]; tot = e["win"] + e["draw"] + e["loss"]
                    d["eval/vs_pure_win_rate"] = e["win"] / tot if tot else 0.0
            if "promoted" in row:
                d["eval/promoted"] = 1.0
            self._wandb.log(d)
        except Exception as e:  # best-effort
            print(f"[wandb] log failed ({e}); continuing", flush=True)

    def _wandb_finish(self) -> None:
        if self._wandb is None:
            return
        try:
            best = os.path.join(self.cfg.out_dir, "best.pt")
            if os.path.exists(best) and self._wandb_module is not None:
                artifact = self._wandb_module.Artifact(f"{self.cfg.game_name}_best", type="model")
                artifact.add_file(best)
                self._wandb.log_artifact(artifact)
            self._wandb.finish()
        except Exception as e:
            print(f"[wandb] finish failed ({e})", flush=True)
        finally:
            self._wandb = None

    # ------------------------------------------------------------------
    def run(self) -> None:
        cfg = self.cfg
        prof.enabled = cfg.profile
        print(f"[loop] {cfg.game_name} backbone={cfg.backbone} arch={cfg.arch} "
              f"device={self.device} amp={self.cfg.amp} iters={cfg.iterations} "
              f"games/iter={cfg.games_per_iter} sims={cfg.sims} "
              f"buffer={cfg.buffer_size} augment={cfg.augment}", flush=True)
        print(f"[loop] log -> {self.log_path}", flush=True)
        self._wandb_init()

        if not self.buffer:
            self.bootstrap()
        self._init_champion()
        legacy = cfg.eval_best_every == 0   # gate off -> legacy vs_pure-score selection
        best_score = -1e9
        best_iter = -1
        n_promotions = 0

        for it in range(cfg.iterations):
            t_iter = time.time()
            sp_cfg = SelfplayConfig(resign_min_moves=max(4, int(0.05 * self.game.num_moves())),
                                    resign_threshold=-0.9)
            with prof.track("loop.selfplay"):
                # advance the seed per iteration so the Dirichlet + move-sampling
                # streams are independent across iterations (not just within one).
                it_seed = (cfg.seed + it) if cfg.seed is not None else None
                mcts = self._selfplay_mcts(cfg.sims, seed=it_seed)
                samples = run_selfplay(self.game.copy, mcts, cfg.games_per_iter,
                                       seed=it_seed)
            self._ingest(samples)
            with prof.track("loop.train"):
                metrics = self._train(cfg.epochs_per_iter)
            it_time = time.time() - t_iter
            row = {
                "iter": it, "buffer": len(self.buffer), "it_time_s": round(it_time, 2),
                "loss": round(metrics.get("loss", float("nan")), 4),
                "policy_loss": round(metrics.get("policy_loss", float("nan")), 4),
                "value_loss": round(metrics.get("value_loss", float("nan")), 4),
                "final_lr": metrics.get("final_lr"),
            }
            last = (it == cfg.iterations - 1)
            extra = ""
            # (a) reference eval vs {random, pure} -- logged; legacy selection source
            if cfg.eval_every and (it % cfg.eval_every == 0 or last):
                with prof.track("loop.eval"):
                    ev = self.evaluate(cfg.eval_games)
                row["eval"] = ev
                extra += f"  vs_pure {ev['vs_pure']}  vs_rnd {ev['vs_random']}"
                if legacy:
                    row["score"] = round(self._score(ev), 3)
                    if self._score(ev) > best_score:
                        best_score = self._score(ev)
                        best_iter = it
                        self._save("best")
                self._save("latest")
            # (b) challenger-vs-best gate -- promotion authority when enabled
            if cfg.eval_best_every and (it % cfg.eval_best_every == 0 or last):
                with prof.track("loop.eval_best"):
                    evb = self._evaluate_best(cfg.best_eval_games)
                row["eval_best"] = evb
                row["winrate_vs_best"] = evb["winrate_vs_best"]
                extra += (f"  vs_best {evb['winrate_vs_best']:.2f} "
                          f"({evb['win']}-{evb['draw']}-{evb['loss']})")
                if evb["winrate_vs_best"] > cfg.promote_winrate:
                    self._promote()
                    best_iter = it
                    best_score = evb["winrate_vs_best"]   # winrate that earned the promotion
                    n_promotions += 1
                    row["promoted"] = True
                    extra += "  *PROMOTED*"
            print(f"[iter {it:03d}] {it_time:6.1f}s  buf={row['buffer']:>7d}  "
                  f"loss={row['loss']:.3f} pol={row['policy_loss']:.3f} "
                  f"val={row['value_loss']:.3f}{extra}", flush=True)
            self._log(row)
            self._wandb_log(row)
            if cfg.profile and prof.snapshot():
                self._log({"iter": it, "profile": prof.snapshot()})
                prof.reset()

        # final eval + summary
        ev = self.evaluate(cfg.eval_games)
        summary = {
            "game": cfg.game_name, "backbone": cfg.backbone, "arch": cfg.arch,
            "iterations": cfg.iterations, "total_time_s": round(time.time() - self._t_start, 1),
            "selection": "challenger-vs-best" if not legacy else "vs_pure_score",
            "eval_best_every": cfg.eval_best_every, "best_eval_games": cfg.best_eval_games,
            "promote_winrate": cfg.promote_winrate, "promotions": n_promotions,
            "best_iter": best_iter, "best_score": best_score, "final_eval": ev,
            "log": self.log_path,
        }
        self._log({"summary": summary})
        with open(os.path.join(cfg.out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        self._wandb_log({"iter": cfg.iterations, "eval": ev, "score": round(self._score(ev), 3)})
        self._wandb_finish()
        print("\n=== SUMMARY ===", flush=True)
        print(json.dumps(summary, indent=2), flush=True)
        print(f"best model -> {os.path.join(cfg.out_dir, 'best.pt')}", flush=True)
