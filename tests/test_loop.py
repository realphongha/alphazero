import json
import os

import numpy as np
import torch

from alphazero.loop import AlphaZeroLoop, LoopConfig


def _tiny_cfg(out_dir):
    return LoopConfig(
        game_name="tictactoe", device="cpu", seed=1,
        backbone="resnet", arch={"channels": 16, "blocks": 1},
        sims=30, mcts_batch=8, playouts=4, c_puct=1.5,
        bootstrap_games=2, bootstrap_sims=30,
        iterations=2, games_per_iter=3, epochs_per_iter=1,
        lr=1e-2, warmup_steps=2, batch_size=64, amp="none",
        buffer_size=500, augment=True,
        eval_every=1, eval_games=4, eval_sims=20, baseline_sims=20,
        out_dir=out_dir,
    )


def test_loop_runs_end_to_end(tmp_path):
    cfg = _tiny_cfg(str(tmp_path))
    loop = AlphaZeroLoop(cfg)
    before = {n: p.clone() for n, p in loop.model.named_parameters()}
    loop.run()
    # outputs exist
    assert os.path.exists(os.path.join(str(tmp_path), "best.pt"))
    assert os.path.exists(os.path.join(str(tmp_path), "latest.pt"))
    assert os.path.exists(os.path.join(str(tmp_path), "summary.json"))
    assert os.path.exists(os.path.join(str(tmp_path), "train.jsonl"))
    # training actually updated the weights
    changed = any(
        not torch.equal(before[n], loop.model.get_parameter(n))
        for n in before if n in dict(loop.model.named_parameters())
    )
    assert changed
    # summary is valid JSON and eval produced finite counts
    with open(os.path.join(str(tmp_path), "summary.json")) as f:
        s = json.load(f)
    vr = s["final_eval"]["vs_random"]
    assert vr["win"] + vr["draw"] + vr["loss"] == 4
    assert s["best_iter"] >= 0


def test_loop_no_augment_half_data(tmp_path):
    cfg = _tiny_cfg(str(tmp_path))
    cfg.augment = False
    loop = AlphaZeroLoop(cfg)
    loop.run()
    # without augmentation the buffer grows 4x slower; still completes
    assert os.path.exists(os.path.join(str(tmp_path), "best.pt"))


def test_wandb_disabled_is_noop(tmp_path):
    cfg = _tiny_cfg(str(tmp_path))
    loop = AlphaZeroLoop(cfg)
    loop._wandb_init()                       # no-op when wandb disabled
    assert loop._wandb is None
    loop._wandb_log({"iter": 0, "loss": 1.0})  # safe no-op when not initialised
    loop._wandb_finish()                      # safe no-op


def test_wandb_enabled_requires_wandb_package(tmp_path):
    import importlib.util
    import pytest
    if importlib.util.find_spec("wandb") is not None:
        pytest.skip("wandb is installed; missing-package path not applicable")
    cfg = _tiny_cfg(str(tmp_path))
    cfg.wandb = True
    loop = AlphaZeroLoop(cfg)
    with pytest.raises(RuntimeError, match="not installed"):
        loop._wandb_init()


def test_loop_respects_start_model(tmp_path):
    cfg = _tiny_cfg(str(tmp_path))
    loop = AlphaZeroLoop(cfg)
    loop._save("seed")  # save a seed checkpoint
    cfg2 = _tiny_cfg(str(tmp_path))
    cfg2.start_model = os.path.join(str(tmp_path), "seed.pt")
    cfg2.iterations = 1
    loop2 = AlphaZeroLoop(cfg2)
    # loaded model should match the saved one before any training
    a = {n: p.clone() for n, p in loop2.model.named_parameters()}
    ref = {n: p.clone() for n, p in loop.model.named_parameters()}
    for n in ref:
        assert torch.allclose(a[n], ref[n], atol=1e-6), n


def test_champion_gate_methods(tmp_path):
    cfg = _tiny_cfg(str(tmp_path))
    loop = AlphaZeroLoop(cfg)
    loop._init_champion()
    # champion matches the model at init; best.pt written
    for n, p in loop.model.named_parameters():
        assert torch.allclose(p, loop.champion_model.get_parameter(n), atol=1e-6), n
    assert os.path.exists(os.path.join(str(tmp_path), "best.pt"))
    # perturb the model, then _promote copies it into the champion
    with torch.no_grad():
        for p in loop.model.parameters():
            p.add_(0.01)
    before = {n: p.clone() for n, p in loop.model.named_parameters()}
    loop._promote()
    for n, p in loop.champion_model.named_parameters():
        assert torch.allclose(before[n], p, atol=1e-6), n
    # _evaluate_best returns a consistent structure; pure ref off by default
    evb = loop._evaluate_best(2)
    assert evb["win"] + evb["draw"] + evb["loss"] == 2
    assert 0.0 <= evb["winrate_vs_best"] <= 1.0
    assert "vs_pure" not in evb
    # enabling the (cheaper-to-omit) pure-MCTS reference adds a vs_pure block
    cfg.eval_best_pure_ref = True
    evb2 = loop._evaluate_best(2)
    vp = evb2["vs_pure"]
    assert vp["win"] + vp["draw"] + vp["loss"] == cfg.eval_games


def test_loop_runs_with_best_gate(tmp_path):
    cfg = _tiny_cfg(str(tmp_path))
    cfg.eval_best_every = 1
    cfg.best_eval_games = 2
    cfg.eval_every = 0   # disable the reference/legacy eval to isolate the gate
    loop = AlphaZeroLoop(cfg)
    loop.run()
    with open(os.path.join(str(tmp_path), "summary.json")) as f:
        s = json.load(f)
    assert s["selection"] == "challenger-vs-best"
    assert isinstance(s["promotions"], int)
    rows = [json.loads(l) for l in open(os.path.join(str(tmp_path), "train.jsonl"))
            if "eval_best" in json.loads(l)]
    assert len(rows) >= 1
    for r in rows:
        assert 0.0 <= r["winrate_vs_best"] <= 1.0
        eb = r["eval_best"]
        assert eb["win"] + eb["draw"] + eb["loss"] == cfg.best_eval_games
    assert os.path.exists(os.path.join(str(tmp_path), "best.pt"))
