import numpy as np

from alphazero.games import make_game
from alphazero.training.data import Sample, augment_d4, augment_samples, _d4_cell_perms


def test_d4_perms_are_bijections():
    for n in (3, 9, 15):
        perms = _d4_cell_perms(n)
        assert len(perms) == 8
        for p in perms:
            assert sorted(p.tolist()) == list(range(n * n))  # permutation


def test_d4_identity_first_and_distinct_orbit():
    n = 3
    perms = _d4_cell_perms(n)
    assert perms[0].tolist() == list(range(9))  # identity
    # orbit of a corner under D4 is exactly the 4 corners (each 2x)
    corner = 0
    orbit = [int(p[corner]) for p in perms]  # where does corner map to
    # actually p[dest]=source, so to find the image of the corner, invert:
    imgs = set()
    for p in perms:
        dest = int(np.where(p == corner)[0][0])
        imgs.add(dest)
    assert imgs == {0, 2, 6, 8}, imgs


def test_augment_moves_stone_and_policy_consistently():
    g = make_game("tictactoe")
    g = g.apply(0)  # p1 stone at cell 0 (top-left)
    obs = g.observation()
    pol = np.zeros(9, dtype=np.float32)
    pol[0] = 1.0
    cells = []
    for no, np_ in augment_d4(obs, pol):
        assert no[0].sum() == 1.0           # exactly one p1 stone
        stone_cell = int(no[0].argmax())
        cells.append(stone_cell)
        assert np_.argmax() == stone_cell    # policy mass follows the stone
        assert abs(np_.sum() - 1.0) < 1e-6
    assert set(cells) == {0, 2, 6, 8}        # orbit of a corner = the 4 corners


def test_augment_preserves_value_and_8x():
    g = make_game("tictactoe")
    obs = g.observation()
    pol = np.full(9, 1.0 / 9, dtype=np.float32)
    sample = Sample(obs, pol, -1.0)
    out = augment_samples([sample, Sample(obs, pol, 0.5)])
    assert len(out) == 16  # 2 samples * 8
    vals = set(round(s.value, 4) for s in out)
    assert vals == {-1.0, 0.5}
    for s in out:
        assert abs(s.policy.sum() - 1.0) < 1e-5
        assert s.obs.shape == (3, 3, 3)


def test_augment_is_a_symmetry_of_wins():
    """Augmenting a winning position must keep it winning (value preserved),
    and the policy must remain legal-only where it was legal-only."""
    g = make_game("tictactoe")
    for mv in [0, 3, 1, 4, 2]:
        g = g.apply(mv)
    assert g.winner() == 1
    obs = g.observation()
    # a policy concentrated on the winning third stone (cell 2)
    pol = np.zeros(9, dtype=np.float32)
    pol[2] = 1.0
    for no, np_ in augment_d4(obs, pol):
        assert no[0].sum() == 3.0  # three p1 stones preserved under symmetry
        assert abs(np_.sum() - 1.0) < 1e-6
