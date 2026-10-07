import os

import pytest

pygame = pytest.importorskip("pygame")

from alphazero.games import make_game
from alphazero.bot import Bot, BotConfig
from alphazero.gui import AlphaZeroGUI


def test_gui_headless_selftest():
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    game = make_game("tictactoe")
    cfg = BotConfig(name="random", kind="random", num_sims=20, batch_size=4,
                    num_playouts=2, device="cpu")
    bot = Bot(cfg, game)
    gui = AlphaZeroGUI(game, {1: None, 2: bot}, title="selftest")
    # run a bounded number of frames headless; must not raise and should
    # let the (background-thread) bot actually move at least once.
    gui.run(max_frames=60)
