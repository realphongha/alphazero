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
