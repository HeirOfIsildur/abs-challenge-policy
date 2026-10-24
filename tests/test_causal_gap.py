"""HONZA-100: the exact pitch-chain kernel of src/causal_gap.py (no data needed).
run: python -m pytest tests"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import causal_gap as CG


def _ev(kind):
    ev = np.zeros((12, 3, 8, 6)); ev[..., kind] = 1.0
    return ev


def _kip_single():
    kip = np.zeros((12, 3, 8, 160)); kip[..., 1 * 40 + 1 * 5 + 0] = 1.0       # in play -> 1 out, runner on first
    return kip


def test_rows_are_distributions():
    rng = np.random.default_rng(0)
    ev = rng.dirichlet(np.ones(6), size=(12, 3, 8)); kip = rng.dirichlet(np.ones(160), size=(12, 3, 8))
    T = CG.chain_T(ev, kip, np.full(12, 0.4))
    assert T.shape == (12, 3, 8, 4, 8, 5)
    assert np.allclose(T.sum((3, 4, 5)), 1.0, atol=1e-10)
    assert np.all(T[:, :, :, 3, 1:, :] == 0)                                  # three outs: bases cleared


def test_called_strikes_give_a_strikeout_and_balls_a_walk():
    T = CG.chain_T(_ev(0), _kip_single(), np.ones(12))                       # every pitch a called strike
    assert T[0, 0, 0, 1, 0, 0] == 1.0 and T[0, 2, 5, 3, 0, 0] == 1.0
    T = CG.chain_T(_ev(0), _kip_single(), np.zeros(12))                      # every pitch a called ball
    assert T[0, 1, 7, 1, 7, 1] == 1.0                                         # bases loaded walk: one run, still loaded
    assert T[9, 0, 2, 0, 3, 0] == 1.0                                         # 3-0, runner on 2nd: walk -> 1st and 2nd


def test_in_play_uses_the_count_kernel():
    kip = np.zeros((12, 3, 8, 160)); kip[..., 0 * 40 + 0 * 5 + 1] = 1.0       # default: a run scores, bases empty
    kip[3, ..., :] = 0; kip[3, ..., 1 * 40 + 0] = 1.0                         # at 1-0: an out
    ev = np.zeros((12, 3, 8, 6)); ev[..., 3] = 1.0; ev[0, :, :, :] = 0; ev[0, :, :, 5] = 1.0   # 0-0 is a ball, then in play
    T = CG.chain_T(ev, kip, np.zeros(12))
    assert T[0, 0, 0, 1, 0, 0] == 1.0                                         # 0-0 -> ball -> 1-0 -> out


def test_no_move_baserunning_is_a_no_op():
    rng = np.random.default_rng(1)
    ev = rng.dirichlet(np.ones(6), size=(12, 3, 8)); kip = rng.dirichlet(np.ones(160), size=(12, 3, 8))
    stay = np.zeros((3, 8, 160))
    for o in range(3):
        for b in range(8):
            stay[o, b, o * 40 + b * 5] = 1.0
    assert np.allclose(CG.chain_T(ev, kip, np.full(12, 0.3)), CG.chain_T(ev, kip, np.full(12, 0.3), stay), atol=1e-10)


def test_stolen_base_before_every_pitch():
    M = np.zeros((3, 8, 160))
    for o in range(3):
        for b in range(8):
            M[o, b, o * 40 + b * 5] = 1.0
    M[0, 1] = 0; M[0, 1, 0 * 40 + 2 * 5] = 1.0                                # runner on 1st always steals 2nd
    T = CG.chain_T(_ev(0), _kip_single(), np.ones(12), M)                    # then three called strikes
    assert np.isclose(T[0, 0, 1, 1, 2, 0], 1.0)
