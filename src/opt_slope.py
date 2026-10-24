"""HONZA-114 (referee): the benchmark for the revealed-threshold slope. Teams' implied confidence thresholds p* are
regressed as logit(1 - p*) on log g with the state controls of kappa (postseason.implied_slope); a stake-rational
rule with the continuation value D held fixed has slope 1. D itself moves with g (late, close games), so the right
benchmark is the same regression run on the optimum's own thresholds p* = D/(g + D), D from the score-aware DP at the
recorded state (slot, score bin, opportunity index, challenges held).
usage: python src/opt_slope.py [R]  -> results/opt_slope.json"""
import json, sys
import numpy as np
sys.path.insert(0, "src")


def main(R=4):
    import sim, dp_score
    from headline import spec_variant, MAIN
    m, _ = sim.load(spec_variant(MAIN))
    m = dp_score.add_dstart(m)
    m = sim.opportunities(m)
    kin = sim.kin_index(m)
    rng = np.random.default_rng(1)
    D = np.zeros(len(m))
    for r in range(R):
        _, _, pt = sim.draw(m, rng, spec_variant(MAIN))
        D4h, _ = dp_score.solve_dp_score(m, pt, True); D4a, _ = dp_score.solve_dp_score(m, pt, False)
        c = np.clip(m.left.values, 1, 2); k = np.minimum(kin, D4h.shape[2] - 1)
        D += np.where(m.is_home_team.values, D4h[m.slot.values, m.dbin.values, k, c], D4a[m.slot.values, m.dbin.values, k, c]) / R
    out = {}
    for side in ("off", "def"):
        # HONZA-120 (GPT-6-Astra review): the same eligibility mask as the teams' regression (postseason.eligible:
        # side and challenges left); states with D = 0 stay in, their p* = 0 clipped to 1e-3 as in implied_slope
        e = (m.side == side).values & (m.left.values > 0)
        x = m[e]; g = np.clip(x.g.values, 1e-4, None)
        p = np.clip(D[e] / (g + D[e]), 1e-3, 1 - 1e-3)
        X = np.column_stack([np.ones(e.sum()), np.log(g), (x.left == 1).astype(float), np.clip(x.inning.values, 1, 10) / 9,
                             (x.strikes == 2).astype(float), (x.balls == 3).astype(float)])
        b = np.linalg.lstsq(X, np.log((1 - p) / p), rcond=None)[0]
        # the same with the score state as a further control, to show how much of the shortfall is the score
        Xs = np.column_stack([X, np.eye(13)[np.clip(x.dbin.values, 0, 12)][:, 1:]])
        bs = np.linalg.lstsq(Xs, np.log((1 - p) / p), rcond=None)[0]
        out[side] = dict(n=int(e.sum()), slope_log_g=float(b[1]), slope_log_g_score_controls=float(bs[1]), mean_pstar=float(p.mean()))
        print(side, out[side], flush=True)
    json.dump(dict(R=R, variant=MAIN, opt=out), open("results/opt_slope.json", "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
