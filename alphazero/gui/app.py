"""Pygame GUI: player vs bot and bot vs bot.

* A *Bot* (neural or pure-MCTS) thinks in a background thread so the window
  never freezes while MCTS runs.
* The two sides can be entirely different: different checkpoint files,
  different sim budgets, or neural vs pure-MCTS.
* Human side clicks a cell to move; the bot side plays automatically.
* ``--selftest`` runs a bounded, input-free loop under a dummy video driver so
  the render path can be smoke-tested headless.
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..bot import Bot, BotConfig
from ..games.base import Game
from ..profiling import prof

BOT_EVENT = "AZ_BOT_MOVE"


class _BotWorker(threading.Thread):
    def __init__(self, bot: Bot, state: Game, on_done) -> None:
        super().__init__(daemon=True)
        self.bot = bot
        self.state = state
        self.on_done = on_done

    def run(self) -> None:
        try:
            with prof.track("gui.bot_search"):
                action = self.bot.best_action(self.state)
            self.on_done(action, None)
        except Exception as e:  # pragma: no cover - defensive
            self.on_done(-1, e)


class _Theme:
    BG = (24, 26, 32)
    BOARD = (120, 90, 60)
    LINE = (40, 30, 20)
    STONE = {1: (235, 235, 240), 2: (30, 30, 30)}
    STONE_EDGE = {1: (200, 200, 200), 2: (90, 90, 90)}
    LAST = (255, 80, 60)
    HOVER = (80, 200, 120)
    TEXT = (230, 230, 235)
    ACCENT = (90, 160, 255)


class AlphaZeroGUI:
    def __init__(
        self,
        game: Game,
        bots: Dict[int, Optional[Bot]],
        human_player: int = 1,
        title: str = "AlphaZero",
    ) -> None:
        self.game = game
        self.n = game.n
        self.m = game.m
        self.bots = bots
        self.human_player = human_player
        self.title = title
        self.T = _Theme()

        self.state = game.copy()
        self.turn = self.state.current_player()
        self.winner = 0
        self.worker: Optional[_BotWorker] = None
        self.result_q: "queue.Queue" = queue.Queue()
        self.hover: int = -1
        self._closed = False

        # layout
        self.cell = 34 if self.n <= 9 else 26
        self.margin = 24
        self.board_px = self.cell * self.n
        self.win_w = self.board_px + 2 * self.margin
        self.win_h = self.board_px + 2 * self.margin + 46  # status strip

    # -- geometry ---------------------------------------------------------
    def cell_center(self, idx: int):
        r, c = divmod(idx, self.n)
        x = self.margin + (c + 0.5) * self.cell
        y = self.margin + 46 + (r + 0.5) * self.cell
        return int(x), int(y)

    def idx_at(self, x: int, y: int) -> int:
        x -= self.margin
        y -= self.margin + 46
        if x < 0 or y < 0 or x >= self.board_px or y >= self.board_px:
            return -1
        c = x // self.cell
        r = y // self.cell
        return r * self.n + c

    # -- turn logic -------------------------------------------------------
    def current_is_human(self) -> bool:
        return self.bots.get(self.turn) is None

    def apply_move(self, move: int) -> None:
        if self.winner:
            return
        if not self.state.is_move_legal(move):
            return
        self.state = self.state.apply(move)
        self.turn = self.state.current_player()
        if self.state.is_terminal():
            self.winner = self.state.winner()

    def maybe_start_bot(self) -> None:
        if self.winner or self.worker is not None:
            return
        if not self.current_is_human():
            st = self.state.copy()
            self.worker = _BotWorker(self.bots[self.turn], st, self._bot_done)
            self.worker.start()

    def _bot_done(self, action: int, err) -> None:
        self.result_q.put(("move", action, err))

    def drain_results(self) -> None:
        while True:
            try:
                item = self.result_q.get_nowait()
            except queue.Empty:
                break
            kind, action, err = item
            self.worker = None
            if err is not None:
                raise err
            if action >= 0:
                self.apply_move(action)
        self.maybe_start_bot()

    def random_legal(self, rng: np.random.Generator) -> int:
        legal = self.state.legal_moves()
        if len(legal) == 0:
            return -1
        return int(rng.choice(legal))

    # -- drawing ----------------------------------------------------------
    def draw(self, surface) -> None:
        T = self.T
        surface.fill(T.BG)
        # board
        import pygame
        pygame.draw.rect(surface, T.BOARD, (self.margin, 46 + self.margin,
                                            self.board_px, self.board_px))
        # grid lines
        for i in range(self.n + 1):
            x = self.margin + i * self.cell
            pygame.draw.line(surface, T.LINE, (x, 46 + self.margin),
                             (x, 46 + self.margin + self.board_px), 1)
            y = 46 + self.margin + i * self.cell
            pygame.draw.line(surface, T.LINE, (self.margin, y),
                             (self.margin + self.board_px, y), 1)
        # stones
        p1, p2 = self.state.p1, self.state.p2
        for idx in range(self.n * self.n):
            s = 1 << idx
            player = 1 if p1 & s else (2 if p2 & s else 0)
            if player:
                cx, cy = self.cell_center(idx)
                r = self.cell // 2 - 3
                pygame.draw.circle(surface, T.STONE[player], (cx, cy), r)
                pygame.draw.circle(surface, T.STONE_EDGE[player], (cx, cy), r, 2)
        # last move
        if self.state.last_move >= 0:
            cx, cy = self.cell_center(self.state.last_move)
            pygame.draw.circle(surface, T.LAST, (cx, cy), self.cell // 2 - 4, 2)
        # hover (only when it is the human turn and cell is empty)
        if (not self.winner) and self.current_is_human() and self.hover >= 0 \
                and self.state.is_move_legal(self.hover):
            cx, cy = self.cell_center(self.hover)
            pygame.draw.circle(surface, T.HOVER, (cx, cy), self.cell // 2 - 5, 2)
        # status
        font = pygame.font.SysFont(None, 22)
        who = "You" if self.current_is_human() else self.bots[self.turn].cfg.name
        if self.winner:
            if self.winner == 0:
                msg = "Draw."
            else:
                nm = "You" if self.bots.get(self.winner) is None else self.bots[self.winner].cfg.name
                msg = f"{nm} wins!"
        elif self.worker is not None:
            msg = f"{who} thinking...  (N=new, Q=quit)"
        else:
            msg = f"{who} to move  (N=new, Q=quit)"
        txt = font.render(msg, True, T.TEXT if not self.winner else T.ACCENT)
        surface.blit(txt, (self.margin, 12))
        pygame.display.flip()

    def _status_only(self, surface) -> None:
        pass

    # -- main loop --------------------------------------------------------
    def run(self, max_frames: Optional[int] = None,
            rng: Optional[np.random.Generator] = None) -> None:
        import pygame
        pygame.init()
        pygame.display.set_caption(self.title)
        screen = pygame.display.set_mode((self.win_w, self.win_h))
        clock = pygame.time.Clock()
        rng = rng or np.random.default_rng()
        frames = 0
        self.maybe_start_bot()
        while not self._closed:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self._closed = True
                elif e.type == pygame.MOUSEMOTION:
                    self.hover = self.idx_at(*e.pos)
                elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    self.hover = self.idx_at(*e.pos)
                    if self.winner:
                        self.state = self.game.copy(); self.turn = self.state.current_player(); self.winner = 0; self.maybe_start_bot()
                    elif self.current_is_human():
                        self.apply_move(self.hover)
                        self.maybe_start_bot()
                elif e.type == pygame.KEYDOWN:
                    if e.key in (pygame.K_q, pygame.K_ESCAPE):
                        self._closed = True
                    elif e.key == pygame.K_n:
                        self.state = self.game.copy(); self.turn = self.state.current_player(); self.winner = 0
                        self.worker = None; self.maybe_start_bot()
            # drain bot results
            self.drain_results()
            # headless self-test: auto-play for human turns & stop after N frames
            if max_frames is not None:
                if (not self.winner) and self.current_is_human() and self.worker is None:
                    self.apply_move(self.random_legal(rng))
                    self.maybe_start_bot()
                self.draw(screen)
                frames += 1
                if frames >= max_frames:
                    self._closed = True
            else:
                self.draw(screen)
            clock.tick(60)
        pygame.quit()


def make_gui_bots(game: Game, args) -> Dict[int, Optional[Bot]]:
    """Build the {player: Bot|None} map from GUI CLI args."""
    bots: Dict[int, Optional[Bot]] = {1: None, 2: None}
    # player 2 is always a bot unless --p2-human
    if getattr(args, "p2_human", False):
        bots[2] = None
    else:
        bots[2] = _bot_from_args(game, args.p2_name, args.p2_kind, args.p2_model,
                                 args.sims, args.batch, args.device, args.amp, getattr(args, "playouts", 16))
    if getattr(args, "mode", "pvb") == "botvbot":
        bots[1] = _bot_from_args(game, args.p1_name, args.p1_kind, args.p1_model,
                                 args.sims, args.batch, args.device, args.amp, getattr(args, "playouts", 16))
    else:
        bots[1] = None  # human
    return bots


def _bot_from_args(game, name, kind, model, sims, batch, device, amp, playouts=16) -> Bot:
    cfg = BotConfig(name=name or kind, kind=kind, model_path=model, num_sims=sims,
                    batch_size=batch, num_playouts=playouts, device=device, amp=amp)
    return Bot(cfg, game)
