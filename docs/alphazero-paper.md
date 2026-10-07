# AlphaZero paper notes

Source: [arXiv:1712.01815](https://arxiv.org/abs/1712.01815)
**"Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"**
David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez,
Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, Timothy Lillicrap, Karen Simonyan,
Demis Hassabis — DeepMind, 5 Dec 2017. (PDF: [`1712.01815.pdf`](./1712.01815.pdf))

> Note: this is the **AlphaZero** paper (the generalisation of AlphaGo Zero to
> chess/shogi/Go). The original Go-only paper is "Mastering the game of Go without
> human knowledge" (Silver et al., 2017); the MCTS/PUCT rule is shared and is
> referenced there. Our implementation follows AlphaZero.

---

## 1. The algorithm in one paragraph

A single network `f_θ(s) = (p, v)` maps a board position `s` to a move
distribution `p` and a scalar value `v ≈ E[z | s]` (the expected outcome).
AlphaZero plays games of self-play: at each position it runs a Monte‑Carlo tree
search (MCTS) guided by `(p, v)` to produce a **search policy** `π` (the
distribution over moves implied by the root visit counts). It stores the triple
`(s, π, z)` for every position of every self‑play game, and updates `θ` by
gradient descent on a combined cross‑entropy + mean‑squared + L² loss. The
updated network is then used for the next self‑play games. No human games, no
hand‑crafted features, no domain knowledge beyond the rules.

---

## 2. Network

- Input: an `N × N × (M·T + L)` image stack — `T` history steps of `M` binary
  feature planes (P1 pieces, P2 pieces, …) plus `L` constant planes (colour,
  total move count, repetition / no‑progress counters). `T = 8` in the paper.
  The board is oriented to the perspective of the **current player**.
- Output:
  - policy `p`: a flat distribution over the legal moves. For Go this is a flat
    distribution over `N² + 1` moves (placements + pass). Chess/shogi use a
    plane stack (`8×8×73`, `9×9×139`).
  - value `v`: a scalar in `[-1, 1]` (a `tanh` output) estimating `E[z|s]`.
- Architecture: a convolutional **ResNet** (for Go: 19 residual blocks, 192
  filters; a 256‑filter variant is used in the full runs). The policy head is a
  1×1 convolution producing one logit per move (flattened); the value head is a
  1×1 convolution → global pool → fully‑connected (256) → `tanh`.
- **Illegal moves are masked to probability 0 and the rest re‑normalised** —
  this is exactly what `ZeroNet.predict_batch` does with the legal plane.

## 3. Loss function (equation 1)

```
(p, v) = f_θ(s)
l = (z − v)²  −  π · log p  +  c ‖θ‖²
```

- `(z − v)²` — value mean‑squared error, where `z ∈ {−1, 0, +1}` (loss/draw/win).
- `−π · log p` — policy cross‑entropy against the **search** policy `π`.
- `c‖θ‖²` — L² weight regularisation (constant `c`).

## 4. MCTS (PUCT)

Standard PUCT, identical to AlphaGo Zero. At a state `s`, the child is chosen by

```
a* = argmax_a  [ Q(s,a)  +  P(s,a) · c_puct · √( Σ_b N(s,b) ) / ( 1 + N(s,a) ) ]
```

- `N(s,a)` visit count, `P(s,a)` the (noised) prior from the network, `c_puct`
  the exploration constant (`≈ 1.5` in AlphaGo Zero).
- `Q(s,a)`: the value of the child **from the perspective of the player to move
  at the child** (the parent's opponent). The value is **negated at each backup
  level** so that, at the parent, the search maximises its *own* expected
  outcome. Our `AlphaZeroMCTS` implements exactly this: `Q` is stored per‑node
  from that node's to‑move perspective and the selection term is negated.
- **Dirichlet noise** `Dir(α)` is mixed into the root priors to encourage
  exploration, scaled inversely to the typical number of legal moves:
  `α = {0.3 (chess), 0.15 (shogi), 0.03 (Go)}`.
- The search returns a root **policy** `π` ∝ the root visit counts.

## 5. Self‑play data generation

- A game is played by both players selecting moves from the MCTS policy `π_t`
  (during self‑play the paper selects **in proportion to the root visit count**,
  i.e. temperature 1).
- At the end the terminal position is scored to `z ∈ {−1, 0, +1}`.
- Every position `(s_t, π_t)` is stored, and `v_t` is set to the game outcome
  **from the perspective of the player to move at `s_t`**.
- At **evaluation** (play / tournament) moves are chosen **greedily** w.r.t. the
  root visit count (temperature 0), and **resignation** is enabled.

## 6. Hyperparameters (paper)

| Setting | Value |
|---|---|
| MCTS simulations | **800** |
| Mini‑batch size | **4096** |
| Training steps | 700,000 |
| Learning rate | **0.2**, dropped 3× → 0.02, 0.002, 0.0002 (step decay) |
| Dirichlet `α` | 0.3 / 0.15 / 0.03 (chess / shogi / Go) |
| `c_puct` | ≈ 1.5 (from AlphaGo Zero) |
| Optimiser | (SGD‑style, per AlphaGo Zero) |
| Data augmentation | 8 board symmetries for Go; **none** for chess/shogi (asymmetric) |
| Resignation | −900 cp for 10 moves (chess); 5 % winrate (AlphaZero); Tromp‑Taylor for Go |

---

## 7. How this implementation maps to the paper

| Paper | This repo | Status |
|---|---|---|
| `(p,v) = f_θ(s)` | `ZeroNet` → `(policy_logits, value)`; value `tanh` → `[-1,1]` | ✅ match |
| Loss `(z−v)² − π·log p + c‖θ‖²` | `training.train.loss_fn` = value MSE + policy cross‑entropy | ✅ match |
| `c‖θ‖²` L² in the loss | AdamW **decoupled weight decay** (the 2026 standard; equivalent regulariser) | ⚙️ modernised |
| `z ∈ {−1,0,+1}`, value = expected outcome | identical; `MNKGame.reward`, self‑play `v_t` from mover's perspective | ✅ match |
| PUCT + per‑node value sign (negate on backup) | `AlphaZeroMCTS._select` (negated value term) + `_backup` (sign flip) | ✅ match |
| `c_puct ≈ 1.5` | `MCTSConfig.c_puct = 1.5` (default) | ✅ match |
| Dirichlet noise at root, `α ∝ 1/#moves` | `MCTSConfig.dirichlet_alpha = 0.03` (the Go value), mixed into root priors | ✅ match |
| Policy = flat distribution over `N²` moves, illegal masked | `ZeroNet.predict_batch` masks illegal moves, softmax | ✅ match |
| ResNet backbone (1×1 policy head, global‑pool value head) | `models/resnet.py` | ✅ match |
| Self‑play: ∝ visit count, score `z` at end, store `(s,π,z)` | `selfplay.run_selfplay` | ✅ match |
| Eval greedy w.r.t. visits; resignation | `MCTS.best_action` = argmax (greedy); optional resignation in self‑play | ✅ match |
| Input = `T=8` history planes + L constant planes (colour, move count) | **3 planes** `[p1, p2, legal]`, current‑player‑oriented, no history | ⚙️ simplified |
| Go policy includes a **pass** move (`N²+1`) | **no pass** (pure placement) — pass is rarely useful for connect‑M | ⚙️ simplified |
| Optimiser: SGD‑style, step LR decay (0.2→0.0002) | **AdamW + warmup→cosine** (2026), AMP bf16/fp16 | ⚙️ modernised (per wishlist) |
| Mini‑batch 4096 | default 64, fully configurable (`--batch`) | ⚙️ scaled down |
| 8 symmetries data augmentation (Go) | not used (could be added for MNK symmetry) | ⬜ not implemented |

### Deliberate differences (and why)
- **Warmup‑cosine + AdamW + AMP** replace the paper's SGD + step decay, per the
  "2026 optimisation" requirement. Both are drop‑in; the loss is unchanged.
- **3‑plane input** vs the paper's history stack: for translation‑invariant
  connect‑M games the current board + legal mask carries essentially all the
  signal (unlike chess, where repetition/castle history matters), and it keeps
  the input cheap so self‑play scales. The paper explicitly notes the algorithm
  is "robust for many reasonable choices" of representation.
- **No pass move**: pass is nearly never optimal for a game won by connecting
  M stones; omitting it shrinks the action space by one.
- **Temperature annealing** in self‑play (start 1.0 → 0.0 over the last ~20 % of
  moves) is a practical addition (later used by MuZero) that improves endgame
  determinism; the paper's fixed proportional (temp 1) is our starting point.

### What a faithful "full" port would add (not yet)
- T=8 history planes + colour/move‑count constant planes in `observation()`.
- A `pass` action for Go‑style games.
- Optional 8‑symmetry data augmentation for the square MNK boards.
- Explicit `c‖θ‖²` term in the loss (instead of / in addition to weight decay).
- A `Stockfish`‑style Elo evaluation harness (we ship a simpler win‑rate
  `benchmark`).
