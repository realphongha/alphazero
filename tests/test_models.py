import numpy as np
import torch

from alphazero.games import make_game
from alphazero.models import build_model
from alphazero.checkpoint import ModelMeta, save_model, load_model


def _model(backbone="resnet", game=None):
    game = game or make_game("tictactoe")
    dev = torch.device("cpu")
    arch = {"resnet": {"channels": 16, "blocks": 1}, "vit": {"dim": 16, "depth": 1, "heads": 2}}[backbone]
    return build_model(game, backbone, dev, cfg=arch)


def test_forward_shapes():
    game = make_game("tictactoe")
    m = _model()
    states = [game.copy(), game.apply(0).copy(), game.apply(1).copy()]
    p, v = m.predict_batch(states)
    assert p.shape == (3, 9)
    assert v.shape == (3,)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)


def test_vit_forward_shapes():
    game = make_game("tictactoe")
    m = _model("vit")
    states = [game.copy(), game.apply(2).copy()]
    p, v = m.predict_batch(states)
    assert p.shape == (2, 9)
    assert v.shape == (2,)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)


def test_legal_masking():
    game = make_game("tictactoe")
    st = game
    for mv in [0, 1, 2, 3, 4, 5]:
        st = st.apply(mv)
    m = _model()
    p, _ = m.predict_batch([st])
    legal = st.legal_moves()
    illegal = [i for i in range(9) if i not in legal]
    assert len(legal) == 3 and len(illegal) == 6
    assert np.allclose(p[0][illegal], 0.0, atol=1e-6)
    assert np.all(p[0][legal] > 0)


def test_vit_patch_upsample():
    game = make_game("gomoku9")
    dev = torch.device("cpu")
    m = build_model(game, "vit", dev, cfg={"dim": 16, "depth": 1, "heads": 2, "patch": 3})
    states = [game.copy(), game.apply(4).copy()]
    p, v = m.predict_batch(states)
    assert p.shape == (2, 81)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)


def test_checkpoint_roundtrip(tmp_path):
    game = make_game("gomoku9")
    dev = torch.device("cpu")
    m = build_model(game, "resnet", dev, cfg={"channels": 16, "blocks": 1})
    path = str(tmp_path / "m.pt")
    meta = ModelMeta(backbone="resnet", n=9, m=5, num_moves=81,
                     arch={"channels": 16, "blocks": 1})
    save_model(m, path, meta)
    m2 = load_model(path, dev)
    a, _ = m.predict_batch([game.copy()])
    b, _ = m2.predict_batch([game.copy()])
    assert np.allclose(a, b, atol=1e-6)


def test_amp_bf16_cpu():
    game = make_game("tictactoe")
    dev = torch.device("cpu")
    m = build_model(game, "resnet", dev, cfg={"channels": 16, "blocks": 1})
    # predict_batch with bf16 autocast on CPU should still work
    p, v = m.predict_batch([game.copy()], amp_dtype=torch.bfloat16)
    assert p.shape == (1, 9)
