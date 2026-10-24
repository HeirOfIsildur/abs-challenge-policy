"""HONZA-130 (Sol review): the causal simulator's state at a simulated pitch -- the score-aware DP's dbin and the
opportunity index after a half-inning change (extra innings included), and the fitted criterion and posterior of a
drawn pool pitch at the simulated inning and stake (no data needed).
run: python -m pytest tests"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import causal as C
import lik

DM = 6


def test_half_start_dbin_matches_add_dstart():
    assert C.half_start_dbin(3, 0, 2, DM) == 2 + DM
    assert C.half_start_dbin(9, 1, -9, DM) == 0                       # clipped
    assert C.half_start_dbin(10, 0, 0, DM) == DM + 1                  # extra top half: dp_score.topx()
    assert C.half_start_dbin(12, 0, 0, DM) == DM + 1
    assert C.half_start_dbin(10, 1, -2, DM) == DM - 2                 # extra bottom half: d + DM, d <= 0
    assert C.half_start_dbin(11, 1, 0, DM) == DM


def _toy(allow_dbin=None):
    """every pitch a called strike that is wrong (truth: ball); the away team (t = 0) challenges iff the score-aware
    threshold is 0, which D4 sets only at opportunity index k = 0 (and, optionally, only at one dbin of slot 18);
    the home team never challenges (signal -1e9). Nobody scores, so the game runs to maxinn."""
    ev = np.ones((12, 3, 8, 6)); kip = np.ones((12, 3, 8, 160))
    br = np.zeros((3, 8, 160))
    for o in range(3):
        for b in range(8):
            br[o, b, o * 40 + b * 5:] = 1.0                           # cumulative kernel: never move
    n = 12; pidx = np.arange(n, dtype=np.int64); pcnt = np.ones(12, np.int64); poff = np.arange(12, dtype=np.int64)
    yes = np.ones(n, np.bool_)
    D4 = np.full((19, 2 * DM + 1, 80, 3), 1e9)
    if allow_dbin is None:
        D4[:, :, 0, :] = 0.0
    else:
        D4[:, :, 0, :] = 0.0; D4[18] = 1e9; D4[18, allow_dbin, 0, :] = 0.0
    G = np.ones((10, 2, 3, 8, 21, 4, 3))                              # every flip worth 1: rb counts the successes
    z2 = np.zeros((19, 80, 3)); zc = np.zeros((19, 3)); card = np.zeros((4, 3, 2, 2))
    args = (z2, zc, card, 0.0, 0.0, 0.0, D4, ev, kip, br, poff, pcnt, pidx, yes, yes, yes,
            np.full((n, 3), 0.5), np.full(n, -1e9), np.zeros((n, 3)), G, G, 13, DM,
            yes, np.zeros((2, 6)), np.zeros(4), np.zeros((2, 2)))
    # start: top of the 10th, the branch pitch called a strike, the team past its first opportunity (kin = 1)
    w, rb, ob = C.play(10, 0, 0, 2, 0, 0, 0, 1, 2, 2, 0, 7, 1, DM + 1, *args)
    return w, rb, ob


def test_opportunity_index_restarts_in_every_extra_half():
    # k = 0 is reached again at the start of the tops of the 11th, 12th and 13th (it never reset within slot 18)
    w, rb, ob = _toy()
    assert w == 0.5 and ob == 0.0 and rb == 3.0


def test_extra_top_half_uses_the_dp_encoding():
    assert _toy(allow_dbin=DM + 1)[1] == 3.0                         # topx(), not d + DM = DM
    assert _toy(allow_dbin=DM)[1] == 0.0


def _frame(rng, n):
    return pd.DataFrame(dict(g=np.exp(rng.uniform(np.log(5e-5), np.log(0.3), n)), inning=rng.integers(1, 14, n),
                             left=rng.integers(1, 3, n), strikes=rng.integers(0, 3, n), balls=rng.integers(0, 4, n)))


def _lik_kappa(e, beta, kg, kgl, const):
    return lik.design(e) @ beta + lik.gfeat(e) @ kg + lik.glow(e) * kgl + const


def test_kappa_at_reproduces_the_fitted_criterion():
    rng = np.random.default_rng(0); n = 50
    e = _frame(rng, n); isoff = rng.random(n) < 0.5
    par = {s: (rng.normal(size=6), rng.normal(size=3), rng.normal(), rng.normal(size=n)) for s in (0, 1)}
    kcoef = np.array([[par[s][0][1], par[s][0][3], *par[s][1], par[s][2]] for s in (0, 1)])
    knots = np.array([*lik.G_KNOTS, lik.G_LOW])
    kp = np.zeros((n, 3))
    for c in (1, 2):                                                  # kappa_by_count: criterion at left = c
        ec = e.assign(left=c)
        for s in (0, 1):
            k = isoff == bool(s)
            kp[k, c] = _lik_kappa(ec, *par[s][:3], par[s][3])[k]
    kb = C.kappa_base(kp, e.g.values, e.inning.values.astype(np.int64), isoff, kcoef, knots)
    for j in range(n):
        for c in (1, 2):
            # at the row's own state: the row's criterion
            assert abs(C.kappa_at(j, c, int(e.inning[j]), float(e.g[j]), kb, isoff, kcoef, knots) - kp[j, c]) < 1e-10
            # at another inning and stake: the fitted formula at that state (row constants carried over)
            inn2, g2 = int(rng.integers(1, 14)), float(np.exp(rng.uniform(np.log(5e-5), np.log(0.3))))
            e2 = e.iloc[[j]].assign(left=c, inning=inn2, g=g2)
            s = int(isoff[j])
            want = _lik_kappa(e2, *par[s][:3], par[s][3][j])[0]
            assert abs(C.kappa_at(j, c, inn2, g2, kb, isoff, kcoef, knots) - want) < 1e-10


def test_posterior_follows_the_stake_tercile():
    import sim
    pt3 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]); isoff = np.array([True, False])
    cuts = np.array([[0.002, 0.01], [0.001, 0.005]])                  # [def, off]
    for j, sd in ((0, "off"), (1, "def")):
        cs = cuts[1 if sd == "off" else 0]
        for g in (0.0, cs[0] - 1e-9, cs[0], 0.5 * (cs[0] + cs[1]), cs[1], 0.2):
            q = int(np.searchsorted(cs, g, side="right"))             # sim.cal_groups' tercile
            assert C.pt_at(j, g, pt3, isoff, cuts) == pt3[j, q]
    # the cell names posterior_by_tercile rebuilds are sim.cal_groups'
    m = pd.DataFrame(dict(side=["off", "def", "off"], g=[0.0005, 0.003, 0.02], strikes=[2, 0, 1], balls=[0, 3, 1],
                          pgrp=["0", "2", "1"]))
    cal = sim.cal_groups(m, cuts={"off": cuts[1], "def": cuts[0]})
    assert list(cal) == ["0_s2_0", "2_b3_1", "1_o_2"]
    m["cal"] = cal
    maps = {f"{sd}|{p}_{q}": np.full(301, 0.1 * (q + 1) + (0.5 if sd == "def" else 0.0))
            for sd, p in (("off", "0_s2"), ("def", "2_b3"), ("off", "1_o")) for q in range(3)}
    p3, miss = C.posterior_by_tercile(m, np.zeros(3), np.array([0.1, 0.7, 0.3]), maps)
    assert miss == 0 and np.allclose(p3, [[0.1, 0.2, 0.3], [0.6, 0.7, 0.8], [0.1, 0.2, 0.3]])


def test_full_contrast_resimulates_later_divergences():
    """the toy game of _toy: the policy challenges (and wins) at the first opportunity of every half, the team's
    behavioural rule never challenges (signal -1e9). From a divergence at the top of the 10th where the policy flipped
    the call, world P gains at the tops of the 11th-25th (L_P = 15; full_branches plays to the 25th) and world B nothing, so the full RB contrast is
    g_b + 15; following the policy in both worlds (the first-divergence RB) would credit 0 after the branch."""
    ev = np.ones((12, 3, 8, 6)); kip = np.ones((12, 3, 8, 160)); br = np.zeros((3, 8, 160))
    for o in range(3):
        for b in range(8):
            br[o, b, o * 40 + b * 5:] = 1.0
    n = 12; ar = np.arange(n, dtype=np.int64); yes = np.ones(n, np.bool_)
    D4 = np.full((19, 2 * DM + 1, 80, 3), 1e9); D4[:, :, 0, :] = 0.0
    G = np.ones((10, 2, 3, 8, 21, 4, 3))
    I = lambda *x: np.array(x, np.int64)
    fo = C.full_branches(I(0), I(0), I(2), I(2), I(2), I(0), I(10), I(0), I(0), I(2), I(0), I(0), I(0), I(0), I(DM + 1), 7,
                         np.zeros((19, 80, 3)), np.zeros((19, 80, 3)), np.zeros((19, 3)), np.zeros((19, 3)),
                         np.zeros((4, 3, 2, 2)), 0.0, 0.0, 0.0, D4, D4, ev, kip, br, ar, np.ones(12, np.int64), ar, yes, yes,
                         yes, np.full((n, 3), 0.5), np.full(n, -1e9), np.zeros((n, 3)), G, G, 2, 1, DM, yes, np.zeros((2, 6)),
                         np.zeros(4), np.zeros((2, 2)))
    assert np.allclose(fo[0], [0.5, 15.0, 0.0, 0.5, 0.0, 0.0])
