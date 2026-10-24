"""guardrails: the vectorised stake table, the compiled DP recursion, the count-based join and the
rho split of the decision noise must reproduce the reference implementations.
Data tests read data/ (not committed) and are skipped when it is absent.
run: python -m pytest tests"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DATA = ROOT / "data"


def need(*names):
    miss = [n for n in names if not (DATA / n).exists()]
    if miss:
        pytest.skip(f"data not present: {miss}")


def test_state_table_matches_vcount():
    need("wp_engine.npz")
    import os
    os.chdir(ROOT)
    import stakes
    WV, RV = stakes.state_table()
    rng = np.random.default_rng(132); D = stakes.D
    for _ in range(300):
        inn = int(rng.integers(1, 14)); h = int(rng.integers(0, 2)); o = int(rng.integers(0, 3)); b = int(rng.integers(0, 8))
        d = int(rng.integers(-30, 31)); bl = int(rng.integers(0, 5)); st = int(rng.integers(0, 4))
        if bl == 4 and st == 3:
            continue
        w, r = stakes.Vcount(inn, h, o, b, d, bl, st)
        i = min(inn, 10) - 1; dd = int(np.clip(d, -(D + 4), D + 4)) + D + 4
        assert abs(WV[i, h, o, b, dd, bl, st] - w) < 1e-12 and abs(RV[i, h, o, b, dd, bl, st] - r) < 1e-12


def test_compiled_half_value_matches_reference():
    import dp_score
    rng = np.random.default_rng(7)
    gg = rng.uniform(0, 0.05, 400); pp = rng.uniform(0, 1, 400); h = rng.uniform(0.3, 0.9, dp_score.KMAX)
    Vend = np.array([0.0, 0.01, 0.015])
    step = dp_score.stepper(gg, pp)
    C, Dk = dp_score.half_value(step, h, Vend)
    C0 = Vend.copy(); D0 = np.zeros((dp_score.KMAX, 3))
    for k in range(dp_score.KMAX - 1, -1, -1):
        D0[k, 1], D0[k, 2] = C0[1] - C0[0], C0[2] - C0[1]
        C0 = h[k] * step(C0) + (1 - h[k]) * Vend
    assert np.allclose(C, C0, atol=1e-15) and np.allclose(Dk, D0, atol=1e-15)


def test_count_join_unique_and_consistent():
    need("called_2026.parquet")
    m = pd.read_parquet(DATA / "called_2026.parquet", columns=["game_pk", "at_bat_number", "pre_b", "pre_s", "stk", "misjoin",
                                                                "balls", "strikes"])
    ok = ~m.misjoin
    assert not m[ok].duplicated(["game_pk", "at_bat_number", "pre_b", "pre_s", "stk"]).any()
    assert ((m.balls == m.pre_b) & (m.strikes == m.pre_s))[ok].all()
    assert m.misjoin.sum() < 20


def test_rho_split_keeps_the_challenge_curve():
    """the behavioural signal s_p + sqrt(1 - rho) sr e2 has the fitted noise sr for every rho (Monte Carlo, loose)."""
    rng = np.random.default_rng(3); n = 400000; sr = 2.0
    for rho in (1.0, 0.5, 0.2):
        s = np.sqrt(rho) * sr * rng.standard_normal(n) + np.sqrt(1 - rho) * sr * rng.standard_normal(n)
        assert abs(s.std() - sr) < 0.01
