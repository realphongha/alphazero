# alphazero

A compact, optimized **AlphaZero** library in PyTorch, built to learn to play
**connect‑M (MNK) games** — tic‑tac‑toe, 9×9 Gomoku and 15×15 Gomoku — using
MCTS + self‑play + a neural policy/value network.

Everything in the original wishlist is implemented and tested:

| Wishlist item | Where |
|---|---|
| MCTS (abstract) | `alphazero/mcts.py` — `MCTS` (ABC) + batched `AlphaZeroMCTS` |
| AlphaZero (abstract) | `alphazero/alphazero.py` — `AlphaZero` (ABC, template `run`) + `AlphaZeroAgent` |
| MNK game: tic‑tac‑toe, 9×9, 15×15 Gomoku | `alphazero/games/` — `MNKGame` + `presets.py` |
| Bitboard efficient ops | `alphazero/games/bitboard.py` (int bitboards, shift‑and‑intersect win test) |
| Batch inference for MCTS nodes | `AlphaZeroMCTS.search` (buffered batched `provider.forward_batch`) + `ZeroNet.predict_batch` |
| Training script (PyTorch, 2026 stack) | `alphazero/training/train.py` — AMP, warmup‑cosine, AdamW, clip, `torch.compile`, grad‑accum |
| ResNet‑style + ViT‑style backbone | `alphazero/models/resnet.py`, `alphazero/models/vit.py` |
| Simple GUI (pygame) | `alphazero/gui/` — player‑vs‑bot and bot‑vs‑bot |
| player‑vs‑bot, bot‑vs‑bot (different bots / pure MCTS) | GUI + `play`/`benchmark` CLI + `alphazero/bot.py` |
| Detailed latency breakdown | `alphazero/profiling.py` + `python -m alphazero latency` (table / JSON / PNG) |

---

## Install

Python ≥ 3.10 and PyTorch (CUDA recommended) are required.

```bash
pip install torch numpy
pip install pygame          # only for the GUI
pip install matplotlib      # optional, for latency charts
# optional: for a fast compile path
pip install -e .            # installs the package (see pyproject.toml)
```

To run without installing, just add the repo to your `PYTHONPATH`; everything is
plain Python under the `alphazero/` package.

---

## Quickstart (reproduces a strong bot)

```bash
cd alphazero

# 1. Bootstrap: pure‑MCTS self‑play (no model needed)
python -m alphazero selfplay --game tictactoe --policy-kind random \
    --games 30 --sims 300 --playouts 32 --out data0.npz

# 2. Train a ResNet policy/value net from the data (AMP + warmup‑cosine)
python -m alphazero train --game tictactoe --data data0.npz --backbone resnet \
    --channels 64 --blocks 3 --epochs 12 --lr 2e-3 --batch 32 --warmup 20 --out model.pt

# 3. Show it is strong: play it against a random opponent
python -m alphazero benchmark --game tictactoe --games 20 \
    --p1-kind model --p1-model model.pt --p1-name az \
    --p2-kind random --sims 30 --amp bf16
#   -> az vs random: over 20 games -> P1=18, P2=1, draws=1   (with this data)
```

The self‑improvement loop (repeat 1–2 using the trained model as the policy):

```bash
python -m alphazero selfplay --game tictactoe --policy-kind model \
    --policy-model model.pt --games 40 --sims 100 --amp bf16 --out data1.npz
python -m alphazero train --game tictactoe --data data1.npz --backbone resnet \
    --channels 64 --blocks 3 --epochs 10 --lr 1e-3 --batch 32 --warmup 20 --out model2.pt
```

> Quality scales with data + epochs. With only a handful of samples the net
> stays near‑uniform; with a few hundred good samples and ~10 epochs it reliably
> dominates a random opponent on tic‑tac‑toe and plays strong Gomoku.

---

## CLI

```
python -m alphazero selfplay    generate supervised data (neural or pure‑MCTS policy)
python -m alphazero train       train ResNet/ViT from a .npz (AMP, warmup‑cosine, ...)
python -m alphazero play        text game: human vs bot, or --botvbot
python -m alphazero gui         pygame GUI: --mode pvb | botvbot
python -m alphazero latency     benchmark + per‑phase latency breakdown (table/json/png)
python -m alphazero benchmark   bot vs bot match (--games N)
```

Common flags: `--game {tictactoe,gomoku9,gomoku15}`, `--device {auto,cuda,cpu}`,
`--seed`, `--playouts` (pure‑MCTS strength), `--sims`, `--batch`, `--amp {none,bf16,fp16}`.

### GUI

```bash
# you (X) vs a model bot (O)
python -m alphazero gui --game gomoku9 --mode pvb \
    --p2-kind model --p2-model model.pt --p2-name az --sims 200 --amp auto

# bot vs bot — the two bots can be completely different
python -m alphazero gui --game gomoku15 --mode botvbot \
    --p1-kind model --p1-model big.pt --p1-name big --sims 300 \
    --p2-kind random --p2-name mcts --playouts 32 --sims 80
```

Controls: **click** a cell to move, **N** = new game, **Q**/Esc = quit. A model
bot thinks in a background thread, so the window stays responsive. Headless
smoke test: add `--selftest 60`.

### Latency breakdown

```bash
python -m alphazero latency --game gomoku9 --backbone resnet --sims 100 \
    --batch 16 --games 3 --train-samples 0 --amp bf16 \
    --json latency.json --plot latency.png
```

The profiler (`alphazero/profiling.py`) accumulates `(count, total)` per named
phase. Instrumented phases include `mcts.select`, `mcts.expand`, `mcts.backup`,
`infer.model_forward_batch`, `selfplay.mcts_search`, and the training phases
`train.data_preprocess`, `train.forward`, `train.backward`,
`train.optimizer_step`. The output is a ranked table (count / total / mean ms),
a JSON dump with 1‑ and 2‑level prefix rollups, and an optional matplotlib bar
chart — the input for further optimization.

---

## Architecture

```
Game (base.Game, ABC)
   └─ MNKGame                    bitboard state (two Python ints), immutable
        └─ observation() (3,N,N) float32 = [p1, p2, legal]

PolicyProvider (inference.PolicyProvider, ABC)      forward_batch(states)->(priors, values)
   ├─ NeuralPolicyProvider      one batched ZeroNet forward (the "batch inference")
   └─ RandomPolicyProvider      pure MCTS: uniform prior + random playouts

ZeroNet (models.nn.ZeroNet, ABC)
   ├─ ResNetNet                 1x1 policy conv (per‑cell) + global‑pool value head
   └─ ViTNet                    patch embed -> transformer blocks -> per‑token policy

MCTS (mcts.MCTS, ABC)
   └─ AlphaZeroMCTS             PUCT, numpy‑vectorised selection, buffered batched model calls

AlphaZero (alphazero.AlphaZero, ABC)
   └─ AlphaZeroAgent            wires provider + MCTS + Trainer; run() = selfplay→train loop

Trainer (training.train.Trainer)   AMP + AdamW + warmup‑cosine + clip + optional torch.compile
```

### Design notes

* **Bitboards.** A board is two arbitrary‑precision Python ints (one per
  player). Legal moves, apply and win‑detection are a few int ops in C. Win
  detection uses *successive shifts + intersection*: a run of M in a direction
  is `AND_k (stones >> k*delta)`; per‑shift *wrap masks* stop runs from jumping
  rows. `tests/test_bitboard.py` property‑tests it against a naive reference on
  random boards for every supported size.
* **Batched MCTS.** Each search round performs `batch_size` independent
  selection walks, deduplicates the leaves, runs **one** batched model forward
  for all of them, then backs each leaf up with its **visit multiplicity**. The
  policy is normalised visit counts over the root's moves.
* **Value sign convention.** `Q` is stored from each node's *own* to‑move
  perspective (sign flips on backup); child selection therefore **negates** the
  value term and adds the (current‑player) prior exploration term.
* **2026 training.** `torch.amp.autocast` (bf16 by default on GPU — no
  `GradScaler` needed and no fp16 overflow; fp16+scaler on request; bf16 on
  CPU), AdamW with decoupled weight decay, linear‑warmup → cosine decay to a
  small floor, gradient clipping, optional gradient accumulation and optional
  `torch.compile`. `channels_last` is set for conv models on CUDA.
* **Immutability.** States are immutable and cheap to copy, so MCTS nodes just
  hold the state they reached.

### Observations & heads

* Input: 3 planes `[p1, p2, legal]` (the legal plane doubles as the masking
  source). Policy logits are masked to legal moves before softmax, so illegal
  moves get zero probability.
* **ResNet policy head** is a 1×1 conv with a single output channel → one logit
  per cell → flatten to (N·N). **Value head** is a 1×1 conv + spatial global
  average pool → MLP → `tanh`.
* **ViT** treats the board as tokens (one per cell by default, or a `patch`×`patch`
  block with nearest‑neighbour up‑sampling of the policy for big boards), with
  learned positional codes and pre‑norm transformer blocks.

---

## Testing

```bash
python -m pytest tests/ -q
```

Covers: bitboard correctness vs a naive reference, game invariants, model
shapes / legal‑masking / checkpoints, MCTS (distribution, strength, legality,
determinism), the full self‑play→train pipeline (loss actually decreases), the
warmup‑cosine schedule, the profiler, and a headless GUI self‑test.

---

## Performance notes / tuning

* Use `--device cuda` and `--amp bf16` (or `auto`) for self‑play and training.
* For big boards the MCTS search is dominated by **tree expansion** (applying
  moves) and, when using a model, by `infer.model_forward_batch`. Raising
  `--batch` improves GPU utilisation; the latency report tells you which phase
  to attack.
* `torch.compile` (`train --compile`) can reduce per‑step overhead but adds a
  one‑time compile cost; enable it for long training runs.
* Pure‑MCTS (`--policy-kind random`) scales with `--sims` *and* `--playouts`;
  for stronger bootstrap data raise both (e.g. `--sims 300 --playouts 32`).

---

## Layout

```
alphazero/
├── games/          base.Game ABC, bitboard.py, mnk.py, presets.py
├── models/         nn.py (ZeroNet, device/amp), resnet.py, vit.py, factory.py
├── inference.py    PolicyProvider ABC + neural / random providers
├── mcts.py         MCTS ABC + batched AlphaZeroMCTS
├── alphazero.py    AlphaZero ABC + AlphaZeroAgent
├── selfplay.py     data generation (temperature annealing, resignation)
├── training/       data.py, scheduler.py, train.py
├── bot.py          unified Bot (model or pure MCTS)
├── checkpoint.py   save/load model + metadata
├── profiling.py    latency breakdown (table/json/png)
├── gui/            pygame GUI
├── cli.py          command-line entry point
└── __main__.py
tests/              pytest suite
```
