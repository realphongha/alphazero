"""Command-line entry point: ``python -m alphazero <cmd> ...``.

Commands
--------
selfplay    generate supervised data with the current policy (neural or pure-MCTS)
train       train a ResNet/ViT ZeroNet from saved data (AMP, warmup-cosine, ...)
play        text-mode game: human vs bot, or bot vs bot
gui         pygame GUI: player vs bot, or bot vs bot (different bots allowed)
latency     benchmark + detailed per-phase latency breakdown (table/json/plot)
benchmark   quick bot-vs-bot match to sanity-check strength
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--game", default="tictactoe", help="tictactoe | gomoku9 | gomoku15")
    p.add_argument("--device", default="auto", help="auto | cuda | cpu")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--playouts", type=int, default=16,
                   help="random playouts for pure-MCTS bots/policy (stronger bootstrap if higher)")


def _bot_args(p: argparse.ArgumentParser, prefix: str, default_kind: str = "random") -> None:
    p.add_argument(f"--{prefix}-kind", default=default_kind, choices=["model", "random"])
    p.add_argument(f"--{prefix}-model", default=None, help="checkpoint .pt path (kind=model)")
    p.add_argument(f"--{prefix}-name", default=None)


def build_parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(prog="alphazero", description=__doc__.split("\n\n")[0])
    sub = top.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("selfplay", help="generate supervised data")
    _common(sp)
    _bot_args(sp, "policy", default_kind="random")
    sp.add_argument("--games", type=int, default=4)
    sp.add_argument("--sims", type=int, default=100)
    sp.add_argument("--batch", type=int, default=16)
    sp.add_argument("--amp", default="none", choices=["none", "bf16", "fp16"])
    sp.add_argument("--out", default="data.npz")

    tr = sub.add_parser("train", help="train a model")
    _common(tr)
    tr.add_argument("--data", required=True)
    tr.add_argument("--backbone", default="resnet", choices=["resnet", "vit"])
    tr.add_argument("--channels", type=int, default=128, help="resnet channels")
    tr.add_argument("--blocks", type=int, default=5, help="resnet resblocks")
    tr.add_argument("--dim", type=int, default=128, help="vit hidden dim")
    tr.add_argument("--depth", type=int, default=4, help="vit depth")
    tr.add_argument("--heads", type=int, default=4, help="vit heads")
    tr.add_argument("--patch", type=int, default=1, help="vit patch size")
    tr.add_argument("--epochs", type=int, default=1)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--batch", type=int, default=64)
    tr.add_argument("--amp", default="auto", choices=["auto", "none", "bf16", "fp16"])
    tr.add_argument("--warmup", type=int, default=0, help="warmup steps (optimiser steps)")
    tr.add_argument("--grad-accum", type=int, default=1)
    tr.add_argument("--compile", action="store_true", help="enable torch.compile")
    tr.add_argument("--out", default="model.pt")

    pl = sub.add_parser("play", help="text game")
    _common(pl)
    pl.add_argument("--botvbot", action="store_true")
    pl.add_argument("--sims", type=int, default=100)
    pl.add_argument("--batch", type=int, default=16)
    pl.add_argument("--amp", default="none", choices=["none", "bf16", "fp16"])
    _bot_args(pl, "p1", default_kind="random")
    _bot_args(pl, "p2", default_kind="random")

    gi = sub.add_parser("gui", help="pygame GUI")
    _common(gi)
    gi.add_argument("--mode", default="pvb", choices=["pvb", "botvbot"])
    gi.add_argument("--sims", type=int, default=100)
    gi.add_argument("--batch", type=int, default=16)
    gi.add_argument("--amp", default="auto", choices=["auto", "none", "bf16", "fp16"])
    gi.add_argument("--p2-human", action="store_true")
    gi.add_argument("--selftest", type=int, default=0, help="run N frames headless then exit")
    _bot_args(gi, "p1", default_kind="random")
    _bot_args(gi, "p2", default_kind="random")

    la = sub.add_parser("latency", help="latency breakdown")
    _common(la)
    la.add_argument("--backbone", default="resnet", choices=["resnet", "vit"])
    la.add_argument("--sims", type=int, default=80)
    la.add_argument("--batch", type=int, default=16)
    la.add_argument("--games", type=int, default=2, help="selfplay games for the profile")
    la.add_argument("--train-samples", type=int, default=0, help="train on N samples (0=skip)")
    la.add_argument("--epochs", type=int, default=1)
    la.add_argument("--amp", default="auto", choices=["auto", "none", "bf16", "fp16"])
    la.add_argument("--json", dest="json_out", default=None, help="dump json to path")
    la.add_argument("--plot", default=None, help="save png chart to path")

    be = sub.add_parser("benchmark", help="bot vs bot match")
    _common(be)
    be.add_argument("--games", type=int, default=1, help="number of matches")
    be.add_argument("--sims", type=int, default=100)
    be.add_argument("--batch", type=int, default=16)
    be.add_argument("--amp", default="none", choices=["none", "bf16", "fp16"])
    _bot_args(be, "p1", default_kind="random")
    _bot_args(be, "p2", default_kind="random")

    return top


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _device(args):
    from .models.nn import get_device
    return get_device(args.device)


def _provider_from_args(args, game, prefix="policy"):
    from .checkpoint import load_model
    from .inference import NeuralPolicyProvider, RandomPolicyProvider
    from .models.nn import resolve_amp_dtype
    kind = getattr(args, f"{prefix}_kind")
    model_path = getattr(args, f"{prefix}_model")
    if kind == "model":
        dev = _device(args)
        model = load_model(model_path, dev)
        return NeuralPolicyProvider(model, resolve_amp_dtype(dev, args.amp)), model
    return RandomPolicyProvider(game, getattr(args, "playouts", 16)), None


def _bot_from_args(args, game, prefix):
    from .bot import Bot, BotConfig
    kind = getattr(args, f"{prefix}_kind")
    model = getattr(args, f"{prefix}_model")
    name = getattr(args, f"{prefix}_name") or f"{prefix}-{kind}"
    cfg = BotConfig(name=name, kind=kind, model_path=model, num_sims=args.sims,
                    batch_size=args.batch, num_playouts=getattr(args, "playouts", 16),
                    device=args.device, amp=args.amp, seed=args.seed)
    return Bot(cfg, game)


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_selfplay(args) -> int:
    from .games import make_game
    from .mcts import AlphaZeroMCTS, MCTSConfig
    from .selfplay import run_selfplay
    from .training.data import save_data
    game = make_game(args.game)
    provider, _ = _provider_from_args(args, game, "policy")
    mcts = AlphaZeroMCTS(game, provider, MCTSConfig(args.sims, batch_size=args.batch, seed=args.seed))
    t0 = time.time()
    def cb(done, total):
        print(f"  selfplay {done}/{total}", flush=True)
    samples = run_selfplay(game.copy, mcts, args.games, seed=args.seed, progress_cb=cb)
    save_data(args.out, samples)
    print(f"selfplay: {len(samples)} samples in {time.time()-t0:.1f}s -> {args.out}")
    return 0


def cmd_train(args) -> int:
    from .games import make_game
    from .models import build_model
    from .training import Trainer, TrainConfig, load_data
    from .checkpoint import ModelMeta, save_model
    game = make_game(args.game)
    dev = _device(args)
    samples = load_data(args.data)
    print(f"train: {len(samples)} samples from {args.data}")
    arch = {"resnet": {"channels": args.channels, "blocks": args.blocks},
            "vit": {"dim": args.dim, "depth": args.depth, "heads": args.heads, "patch": args.patch}}[args.backbone]
    model = build_model(game, args.backbone, dev, cfg=arch)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  backbone={args.backbone} params={n_params/1e6:.2f}M device={dev} arch={arch}")
    trainer = Trainer(model, dev, TrainConfig(lr=args.lr, batch_size=args.batch, epochs=args.epochs,
                                              amp=args.amp, warmup_steps=args.warmup,
                                              grad_accum=args.grad_accum, compile=args.compile, seed=args.seed))
    t0 = time.time()
    metrics = trainer.fit(samples)
    print(f"  trained in {time.time()-t0:.1f}s: "
          + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()))
    meta = ModelMeta(backbone=args.backbone, n=game.n, m=game.m, num_moves=game.num_moves(), arch=arch)
    save_model(model, args.out, meta)
    print(f"  saved -> {args.out}")
    return 0


def _text_play(args, bots, game, human_player=1) -> int:
    rng = np.random.default_rng(args.seed)
    state = game.copy()
    print(f"Game: {args.game}  |  P1={bots[1].describe() if bots[1] else 'human'}  "
          f"|  P2={bots[2].describe() if bots[2] else 'human'}")
    while not state.is_terminal():
        print("\n" + state.render())
        turn = state.current_player()
        bot = bots.get(turn)
        if bot is not None:
            print(f"[{bot.cfg.name} thinking...]", flush=True)
            mv = bot.best_action(state)
            print(f"[{bot.cfg.name}] -> {mv}")
        else:
            while True:
                legal = state.legal_moves()
                try:
                    raw = input(f"You play ({turn}). Legal: {sorted(legal.tolist())} [q=quit] > ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n(quitting)")
                    return 0
                if raw.lower() == "q":
                    return 0
                try:
                    mv = int(raw)
                except ValueError:
                    print("  enter a number"); continue
                if mv in legal.tolist():
                    break
                print("  illegal move")
        state = state.apply(mv)
    print("\n" + state.render())
    print("Draw." if state.winner() == 0 else f"Player {state.winner()} wins!")
    return 0


def cmd_play(args) -> int:
    from .games import make_game
    game = make_game(args.game)
    bots = {1: None, 2: None}
    bots[2] = _bot_from_args(args, game, "p2")
    if args.botvbot:
        bots[1] = _bot_from_args(args, game, "p1")
    return _text_play(args, bots, game)


def cmd_gui(args) -> int:
    try:
        import pygame  # noqa: F401
    except Exception:
        print("pygame is not installed. Install it with:  pip install pygame")
        return 1
    from .games import make_game
    from .gui import AlphaZeroGUI, make_gui_bots
    game = make_game(args.game)
    if args.selftest:
        import os
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    bots = make_gui_bots(game, args)
    gui = AlphaZeroGUI(game, bots, title=f"AlphaZero {args.game} [{args.mode}]")
    gui.run(max_frames=(args.selftest or None))
    return 0


def cmd_latency(args) -> int:
    from .games import make_game
    from .models import build_model
    from .models.nn import resolve_amp_dtype
    from .inference import NeuralPolicyProvider
    from .mcts import AlphaZeroMCTS, MCTSConfig
    from .selfplay import run_selfplay
    from .training import Trainer, TrainConfig
    from .profiling import prof
    game = make_game(args.game)
    dev = _device(args)
    arch = {"resnet": {"channels": 64, "blocks": 3}, "vit": {"dim": 96, "depth": 3, "heads": 3}}[args.backbone]
    model = build_model(game, args.backbone, dev, cfg=arch)
    amp_dtype = resolve_amp_dtype(dev, args.amp)
    provider = NeuralPolicyProvider(model, amp_dtype)
    mcts = AlphaZeroMCTS(game, provider, MCTSConfig(args.sims, batch_size=args.batch, seed=args.seed))

    # warmup (first call compiles kernels / moves weights)
    provider.forward_batch([game.copy()])
    model.predict_batch([game.copy()])

    prof.reset()
    prof.enabled = True
    print(f"latency benchmark: {args.game} backbone={args.backbone} device={dev} amp={args.amp}")
    print(f"  selfplay: {args.games} games @ {args.sims} sims...")
    samples = run_selfplay(game.copy, mcts, args.games, seed=args.seed)
    if args.train_samples:
        n = min(args.train_samples, len(samples))
        print(f"  train: {n} samples, {args.epochs} epoch(s)...")
        trainer = Trainer(model, dev, TrainConfig(lr=1e-3, batch_size=args.batch, epochs=args.epochs,
                                                  amp=args.amp, warmup_steps=5, seed=args.seed))
        trainer.fit(samples[:n])
    print()
    print(prof.report(40))
    if args.json_out:
        p = prof.dump_json(args.json_out)
        print(f"\njson -> {p}")
    if args.plot:
        p = prof.plot(args.plot)
        print(f"plot -> {p}" if p else "plot -> (matplotlib not available)")
    return 0


def _play_match(b1, b2, game):
    state = game.copy()
    while not state.is_terminal():
        bot = b1 if state.current_player() == 1 else b2
        state = state.apply(bot.best_action(state))
    return state.winner()


def cmd_benchmark(args) -> int:
    from .games import make_game
    game = make_game(args.game)
    b1 = _bot_from_args(args, game, "p1")
    b2 = _bot_from_args(args, game, "p2")
    t0 = time.time()
    wins = {1: 0, 2: 0, 0: 0}
    for _ in range(args.games):
        wins[_play_match(b1, b2, game)] += 1
    dt = time.time() - t0
    label = f"{b1.cfg.name} vs {b2.cfg.name}"
    if args.games == 1:
        print(f"{label}: winner={1 if wins[1] else (2 if wins[2] else 'draw')}")
    else:
        print(f"{label}: over {args.games} games -> P1={wins[1]}, P2={wins[2]}, draws={wins[0]}")
    print(f"  time={dt:.1f}s ({dt/max(1, args.games):.2f}s/game)")
    return 0


_COMMANDS = {
    "selfplay": cmd_selfplay,
    "train": cmd_train,
    "play": cmd_play,
    "gui": cmd_gui,
    "latency": cmd_latency,
    "benchmark": cmd_benchmark,
}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    return _COMMANDS[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
