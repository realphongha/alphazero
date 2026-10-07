"""Tests for the torch-free ONNX inference path (onnxruntime CPU).

These prove the model exported from PyTorch reproduces the neural policy/value
exactly, that an ONNX bot plays legal moves, and -- critically -- that the whole
inference path (games + MCTS + ONNX provider) runs with ``import torch`` blocked,
which is what makes it deployable on a machine (e.g. an Apple M1 Mac) that has
only ``onnxruntime`` and no PyTorch.
"""
import importlib.abc
import subprocess
import sys

import numpy as np
import pytest

onnxruntime = pytest.importorskip("onnxruntime")

from alphazero.games import make_game
from alphazero.models import build_model
from alphazero.models.nn import get_device
from alphazero.export import export_to_onnx
from alphazero.inference import OnnxPolicyProvider, NeuralPolicyProvider
from alphazero.bot import Bot, BotConfig


def _model(tmp_path):
    game = make_game("tictactoe")
    m = build_model(game, "resnet", get_device("cpu"), cfg={"channels": 16, "blocks": 1})
    return game, m


def _some_states(game):
    states = [game.copy()]
    g = game.copy()
    for mv in (4, 0, 8, 2):
        if not g.is_terminal():
            g = g.apply(mv)
        states.append(g.copy())
    return states


def test_export_matches_torch(tmp_path):
    game, m = _model(tmp_path)
    p = str(tmp_path / "m.onnx")
    export_to_onnx(m, p, board_dim=game.n)
    states = _some_states(game)
    tp, tv = NeuralPolicyProvider(m, None).forward_batch(states)
    op, ov = OnnxPolicyProvider(p, game=game).forward_batch(states)
    assert np.abs(tp - op).max() < 1e-4
    assert np.abs(tv - ov).max() < 1e-4
    assert np.allclose(op.sum(1), 1.0, atol=1e-5)


def test_onnx_masking_zero_on_illegal(tmp_path):
    game, m = _model(tmp_path)
    p = str(tmp_path / "m.onnx")
    export_to_onnx(m, p, board_dim=game.n)
    prov = OnnxPolicyProvider(p, game=game)
    g = game.copy()
    g = g.apply(4)  # centre occupied -> only 8 legal
    pol, _ = prov.forward_batch([g.copy()])
    legal = g.legal_moves()
    assert len(legal) == 8
    for i in range(game.num_moves()):
        if i not in legal:
            assert pol[0][i] < 1e-6


def test_onnx_bot_makes_legal_move(tmp_path):
    game, m = _model(tmp_path)
    p = str(tmp_path / "m.onnx")
    export_to_onnx(m, p, board_dim=game.n)
    bot = Bot(BotConfig(name="b", kind="onnx", model_path=p, num_sims=20), game)
    assert game.is_move_legal(bot.best_action(game.copy()))


def test_onnx_path_is_torch_free(tmp_path):
    """Build + play an ONNX bot in a subprocess where `import torch` is blocked."""
    game, m = _model(tmp_path)
    p = str(tmp_path / "m.onnx")
    export_to_onnx(m, p, board_dim=game.n)
    code = (
        "import sys, importlib.abc\n"
        "class B(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, f, p=None, t=None):\n"
        "        if f == 'torch' or f.startswith('torch.'): raise ImportError('blocked '+f)\n"
        "        return None\n"
        "sys.meta_path.insert(0, B())\n"
        "import torch  # noqa: F401  -> must raise\n"
    )
    code_ok = (
        "import sys, importlib.abc\n"
        "class B(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, f, p=None, t=None):\n"
        "        if f == 'torch' or f.startswith('torch.'): raise ImportError('blocked '+f)\n"
        "        return None\n"
        "sys.meta_path.insert(0, B())\n"
        "try:\n"
        "    import torch  # noqa\n"
        "    raise SystemExit('torch NOT blocked')\n"
        "except ImportError:\n"
        "    pass\n"
        "from alphazero.games import make_game\n"
        "from alphazero.bot import Bot, BotConfig\n"
        "g = make_game('tictactoe')\n"
        f"b = Bot(BotConfig(name='b', kind='onnx', model_path={p!r}, num_sims=20), g)\n"
        "assert g.is_move_legal(b.best_action(g.copy()))\n"
        "assert 'torch' not in sys.modules\n"
        "print('TORCH_FREE_OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", code_ok], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "TORCH_FREE_OK" in r.stdout
