"""Self-play: MCTS generates the supervised data (s, pi, v) for training.

For each position the current policy is searched with MCTS to produce a
``pi`` (normalised visit counts); a move is then sampled with a temperature that
anneals to (near) greedy late in the game.  Once the game ends, every stored
position is labelled with ``v`` = the final result from that position's mover's
perspective.  An optional early *resignation* cuts off lost games, which speeds
up self-play dramatically on big boards.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from .games.base import Game
from .mcts import AlphaZeroMCTS
from .profiling import prof
from .training.data import Sample


@dataclass
class SelfplayConfig:
    temperature_start: float = 1.0
    temperature_end: float = 0.0
    temperature_decay_frac: float = 0.8  # anneal over the last 20% of moves
    resign_threshold: float = -0.95
    resign_min_moves: int = 20
    max_moves: Optional[int] = None      # None -> board size squared


def temperature_for(move_index: int, max_moves: int, cfg: SelfplayConfig) -> float:
    start_decay = int(cfg.temperature_decay_frac * max_moves)
    if move_index < start_decay or max_moves <= start_decay:
        return cfg.temperature_start
    frac = min(1.0, (move_index - start_decay) / (max_moves - start_decay))
    return cfg.temperature_start + (cfg.temperature_end - cfg.temperature_start) * frac


def choose_move(policy: np.ndarray, legal: np.ndarray, temp: float,
                rng: np.random.Generator) -> int:
    if temp <= 1e-6 or len(legal) <= 1:
        return int(legal[int(np.argmax(policy[legal]))])
    p = policy[legal] ** (1.0 / temp)
    p = p / p.sum()
    return int(legal[int(rng.choice(len(legal), p=p))])


def selfplay_one_game(
    make_game,
    mcts: AlphaZeroMCTS,
    cfg: SelfplayConfig,
    rng: np.random.Generator,
) -> List[Sample]:
    state: Game = make_game()
    n = state.num_moves()
    max_moves = cfg.max_moves or n
    history = []  # (state, policy)
    move_index = 0
    resigner: Optional[int] = None

    while not state.is_terminal() and move_index < max_moves:
        with prof.track("selfplay.mcts_search"):
            policy = mcts.search(state)
        history.append((state, policy))

        if move_index >= cfg.resign_min_moves and mcts.last_root_value <= cfg.resign_threshold:
            resigner = state.current_player()
            break

        legal = state.legal_moves()
        if len(legal) == 0:
            break
        temp = temperature_for(move_index, max_moves, cfg)
        m = choose_move(policy, legal, temp, rng)
        state = state.apply(m)
        move_index += 1

    w = state.winner()
    samples: List[Sample] = []
    for st, pol in history:
        if resigner is not None:
            v = -1.0 if st.current_player() == resigner else 1.0
        elif w == 0:
            v = 0.0
        else:
            v = 1.0 if w == st.current_player() else -1.0
        samples.append(Sample(st.observation(), pol, v))
    return samples


def run_selfplay(
    make_game,
    mcts: AlphaZeroMCTS,
    num_games: int,
    cfg: Optional[SelfplayConfig] = None,
    seed: Optional[int] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> List[Sample]:
    cfg = cfg or SelfplayConfig()
    rng = np.random.default_rng(seed)
    all_samples: List[Sample] = []
    for g in range(num_games):
        with prof.track("selfplay.game"):
            all_samples += selfplay_one_game(make_game, mcts, cfg, rng)
        if progress_cb is not None:
            progress_cb(g + 1, num_games)
    return all_samples


def run_mixed_selfplay(
    make_game,
    net_mcts: AlphaZeroMCTS,
    opponent_move,
    num_games: int,
    net_player: int = 1,
    cfg: Optional[SelfplayConfig] = None,
    seed: Optional[int] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> List[Sample]:
    """Self-play where ``net_mcts`` plays ``net_player`` and a non-neural
    ``opponent_move(state) -> int`` plays the other side.

    Only the *net's* own positions are returned: the opponent's moves are not the
    network's policy, so they must not be used as supervised targets.  Mixing this
    with self-vs-self (``run_selfplay``) exposes the net to a broader positional
    distribution (strong-opponent and weak-opponent positions alike), which sharpens
    tactical play -- one of the levers that made reference implementations stronger.
    """
    cfg = cfg or SelfplayConfig()
    rng = np.random.default_rng(seed)
    all_samples: List[Sample] = []
    for g in range(num_games):
        with prof.track("selfplay.game"):
            all_samples += _mixed_one_game(make_game, net_mcts, opponent_move,
                                           net_player, cfg, rng)
        if progress_cb is not None:
            progress_cb(g + 1, num_games)
    return all_samples


def _mixed_one_game(make_game, net_mcts: AlphaZeroMCTS, opponent_move, net_player: int,
                    cfg: SelfplayConfig, rng: np.random.Generator) -> List[Sample]:
    state: Game = make_game()
    n = state.num_moves()
    max_moves = cfg.max_moves or n
    history = []  # (state, policy) -- net_player's positions only
    move_index = 0
    resigned = False

    while not state.is_terminal() and move_index < max_moves:
        cur = state.current_player()
        legal = state.legal_moves()
        if len(legal) == 0:
            break
        temp = temperature_for(move_index, max_moves, cfg)
        if cur == net_player:
            with prof.track("selfplay.mcts_search"):
                policy = net_mcts.search(state)
            history.append((state, policy))
            if move_index >= cfg.resign_min_moves and net_mcts.last_root_value <= cfg.resign_threshold:
                resigned = True
                break
            m = choose_move(policy, legal, temp, rng)
        else:
            m = opponent_move(state)
        state = state.apply(m)
        move_index += 1

    w = state.winner()
    samples: List[Sample] = []
    for st, pol in history:  # every st is net_player's turn
        if resigned:
            v = -1.0
        elif w == 0:
            v = 0.0
        else:
            v = 1.0 if w == net_player else -1.0
        samples.append(Sample(st.observation(), pol, v))
    return samples
