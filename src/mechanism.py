"""HONZA-122 (Opus, Astra reviews): model-free and calibration diagnostics of the mechanism.
(1) Model-free: challenge rate on wrong calls (w > 0) by stake quintile, by side and distance band (no model).
(2) Revealed vs optimal confidence thresholds by stake quintile (the slope comparison of Section 4.5 without a
    regression): teams' p* = calibrated belief at kappa(x), the DP benchmark's p* = D/(g + D), at the recorded state.
(3) The one-line rule as a criterion on the signal, in inches: the signal threshold s* at which the calibrated belief
    reaches delta/(g + delta), against teams' kappa, by stake quintile (what a player would have to shift).
(4) Stake-weighted calibration on the pitches each policy selects: sum p g against sum W g (Astra's diagnostic of the
    posterior used by the DP; with calibration cells that include the count and the stake they should agree).
usage: python src/mechanism.py [R]  -> results/mechanism.json"""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")


def main(R=6):
    import sim, dp_score, policies as P
    from headline import spec_variant, MAIN, evaluate_all
    v = spec_variant(MAIN)
    m, _ = sim.load(v); m = dp_score.add_dstart(m); m = sim.opportunities(m)
    m["new_extra"] = sim.extra_starts(m); m["cal"] = sim.cal_groups(m)
    kin = sim.kin_index(m).astype(np.int64)
    el = (m.left.values > 0)
    out = {}
    # (1) model-free
    t = m[el & (m.w.values > 0)].copy()
    t["q"] = pd.qcut(t.g, 5, labels=False, duplicates="drop"); t["band"] = pd.cut(t.w, [0, 1, 2, 4, 100], right=False)
    tab = t.groupby(["side", "band", "q"], observed=True).agg(n=("ch", "size"), rate=("ch", "mean"), g=("g", "mean")).reset_index()
    out["model_free"] = [dict(side=r.side, band=str(r.band), q=int(r.q), n=int(r.n), rate=float(r.rate), g_mean=float(r.g)) for r in tab.itertuples()]
    # (2), (3): R draws of the calibration map and the DP
    rng = np.random.default_rng(1); qs = pd.qcut(m.g, 5, labels=False, duplicates="drop").values
    acc = {k: np.zeros(len(m)) for k in ("pt_team", "pt_opt", "s_rule")}
    for r in range(R):
        W, s, pt = sim.draw(m, rng, v)
        D4h, _ = dp_score.solve_dp_score(m, pt, True); D4a, _ = dp_score.solve_dp_score(m, pt, False)
        c = np.clip(m.left.values, 1, 2); k = np.minimum(kin, D4h.shape[2] - 1)
        D = np.where(m.is_home_team.values, D4h[m.slot.values, m.dbin.values, k, c], D4a[m.slot.values, m.dbin.values, k, c])
        acc["pt_opt"] += D / (m.g.values + D + 1e-12) / R
        # calibrated belief as a function of the signal, cell by cell: empirical map from this draw's (s, pt)
        pt_team = np.zeros(len(m)); s_rule = np.zeros(len(m))
        thr = 0.008 / (m.g.values + 0.008)
        for key in np.unique(np.char.add(m.side.values.astype(str), m.cal.values)):
            j = np.char.add(m.side.values.astype(str), m.cal.values) == key
            o = np.argsort(s[j]); ss, pp = s[j][o], np.maximum.accumulate(pt[j][o])
            pt_team[j] = pp[np.clip(np.searchsorted(ss, m.kappa.values[j]), 0, len(ss) - 1)]
            s_rule[j] = ss[np.clip(np.searchsorted(pp, thr[j]), 0, len(ss) - 1)]
        acc["pt_team"] += pt_team / R; acc["s_rule"] += s_rule / R
    rows = []
    for side in ("off", "def"):
        for q in range(5):
            j = el & (m.side.values == side) & (qs == q)
            rows.append(dict(side=side, q=q, g_median=float(np.median(m.g.values[j])), pstar_team=float(acc["pt_team"][j].mean()),
                             pstar_opt=float(acc["pt_opt"][j].mean()), kappa_team_in=float(m.kappa.values[j].mean()),
                             s_rule_in=float(acc["s_rule"][j].mean())))
    out["thresholds_by_stake_quintile"] = rows
    # (4) stake-weighted calibration on selected pitches
    _, _, res, draws = evaluate_all(v, R=R, seed=11, keep_draws=True)
    cal = {}
    for pol in ("opt_score", "stake_0.008", "obs_rule", "savant"):
        a = b = 0.0
        for dr in draws:
            pc, dc, card = dr["pols"][pol]
            base = (m.gidx.values, m.is_home_team.values, m.new_extra.values, m.slot.values.astype(np.int64), kin,
                    m.inning.values.astype(np.int64), m.balls.values.astype(np.int64), m.strikes.values.astype(np.int64),
                    m.dbin.values.astype(np.int64))
            sel = selected(base, dr, m, pc, dc, card, P, sim)
            a += (dr["pt"][sel] * m.g.values[sel]).sum() / 30 / R; b += (dr["W"][sel] * m.g.values[sel]).sum() / 30 / R
        cal[pol] = dict(sum_p_g=float(a), sum_W_g=float(b), ratio=float(b / a))
    out["selected_calibration"] = cal
    json.dump(dict(R=R, variant=MAIN, **out), open("results/mechanism.json", "w"), indent=1)
    print(json.dumps(out["thresholds_by_stake_quintile"], indent=1)); print(cal)


def selected(base, dr, m, pc, dc, card, P, sim):
    """rows at which the policy challenges along its own path (inventory as in run_path)."""
    from numba import njit
    return _sel(*base, dr["W"], dr["pt"], dr["s"], m.g.values, m.RV.values, sim.kappa_by_count(m), pc, dr["Dh"], dr["Da"],
                dr["Cnh"], dr["Cna"], card, dr["q"][0], dr["q"][1], dc, dr["D4h"], dr["D4a"])


from numba import njit
from policies import decide


@njit(cache=True)
def _sel(gidx, is_home_team, new_extra, slot, kin, inning, balls, strikes, dbin, W, pt, s, g, RV, kappa, pol, Dh, Da, Cnh, Cna,
         card, q1, q2, dconst, D4h, D4a):
    out = np.zeros(len(g), np.bool_); c = np.array([2, 2]); cur = -1
    for i in range(len(g)):
        if gidx[i] != cur:
            cur = gidx[i]; c[0] = 2; c[1] = 2
        if new_extra[i]:
            for t in range(2):
                if c[t] == 0: c[t] = 1
        t = 1 if is_home_team[i] else 0
        if c[t] == 0:
            continue
        D = Dh if t == 1 else Da; Cn = Cnh if t == 1 else Cna; D4 = D4h if t == 1 else D4a
        if decide(pol, pt[i], s[i], kappa[i, min(c[t], 2)], g[i], RV[i], slot[i], kin[i], c[t], inning[i], balls[i], strikes[i],
                  D, Cn, card, q1, q2, dconst, D4, dbin[i]):
            out[i] = True
            if not W[i]:
                c[t] -= 1
    return out


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
