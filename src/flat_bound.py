"""The flat-optimum theorem (theory.md, Theorem 3) evaluated on the calibrated 2026 pools.

Model world: the score-aware calibrated DP of dp_score.py restricted to regulation (no extra-inning block, so the MDP
is exact): exogenous states (slot, home differential d at the start of the half-inning), a random number N of
opportunities per half-inning with the empirical law of (slot, d) (it was the rounded mean), each drawn
i.i.d. from the pool of (slot, d) (g, calibrated posterior p), runs between half-innings from
P(r | slot), game over after the top of the 9th if the home team leads and after the 9th.  Per team side (home/away):
  V*            optimal value (backward induction), Delta_k(c) = W_k(c) - W_k(c-1) the continuation used at step k;
  V^delta       value of "challenge iff p >= delta/(g+delta)" (forward occupancy of c under the rule);
  identity (i)  V* - V^delta = sum_steps sum_{c>=1} P^delta(c) E_pool[|p(g+D) - D| 1{rules disagree}], D = Delta_k(c);
  bound (ii)    the same sum with the inner expectation replaced by (M/2) E_g[g^2 (D-delta)^2 / ((g+D)(g+delta)^2)],
                M = sup of the conditional density of p given g over the disagreement interval (histogram estimate,
                bin width HW, by g quintile within the pool). the calibrated posterior
                has atoms, so this is a smoothed approximation of the bound, not a certified bound for that law;
                the identity (i) is exact;
  local (ii')   M replaced by the histogram density at the interval midpoint (second-order approximation).
The real-stream loss of the best constant rule (opt_score vs best stake, headline.py) is the empirical counterpart.
usage: python src/flat_bound.py [R]  -> results/flat_bound.json"""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")

DELTAS = np.round(np.arange(0.004, 0.0201, 0.001), 4)
HW = 0.025; NB = int(round(1 / HW))


def pools(m, pt, home):
    """pools (g, p, g quintile, sup-density table [quintile, b0, b1]) and opportunities per half-inning n[slot, d]."""
    import dp_score
    sel = np.flatnonzero(m.is_home_team.values == home)
    sl_, db = m.slot.values, m.dbin.values
    key = pd.DataFrame(dict(game=m.game_pk.values, inn=m.inning.values, top=m.top.values, slot=sl_, d=db,
                            own=m.is_home_team.values == home))
    hk = key.groupby(["game", "inn", "top"]).agg(slot=("slot", "first"), d=("d", "first"), n=("own", "sum")).reset_index()
    hk = hk[hk.slot < 18]
    P, n = {}, {}
    for (s, d), grp in hk.groupby(["slot", "d"]):     # N law of the half-inning: hazard h_k and P(N >= k+1)
        h = dp_score.hazard(grp.n.values)
        n[(int(s), int(d))] = (h, np.cumprod(h))
    for (s, d), idx in pd.Series(sel).groupby([sl_[sel], db[sel]]):
        if s >= 18:
            continue
        idx = idx.values
        gg, pp = np.maximum(m.g.values[idx], 1e-12), pt[idx]     # G = 0 now occurs (non-monotone engine cells)
        qb = np.quantile(gg, [0.2, 0.4, 0.6, 0.8]) if len(gg) >= 50 else np.array([])
        gq = np.searchsorted(qb, gg)
        dens = np.zeros((5, NB))
        for q in range(5):
            x = pp[gq == q]
            if len(x):
                dens[q] = np.histogram(x, bins=NB, range=(0, 1))[0] / (len(x) * HW)
        sup = np.zeros((5, NB, NB))                       # sup[q, b0, b1] = max density over bins b0..b1
        for b0 in range(NB):
            sup[:, b0, b0:] = np.maximum.accumulate(dens[:, b0:], axis=1)
        P[(s, d)] = (gg, pp, gq, dens, sup)
    return P, n


def mdp(P, n, delta):
    """exact regulation MDP for one team side: V*, V^delta, regret identity (i), bound (ii), local version (ii')."""
    import dp_score
    DM, ND = dp_score.DM, dp_score.ND
    Pr = dp_score.half_runs(); RM = Pr.shape[1] - 1
    dv = np.arange(-DM, DM + 1)
    trans = []                                            # start-of-slot d -> start of slot s+1
    for s in range(18):
        T = np.zeros((ND, ND)); sgn = 1 if s % 2 == 1 else -1
        for r in range(RM + 1):
            T[np.arange(ND), np.clip(dv + sgn * r, -DM, DM) + DM] += Pr[s, r]
        trans.append(T)
    alive = [np.ones(ND) for _ in range(18)]              # game continues into slot s+1 from end-of-slot d
    alive[16] = (dv <= 0).astype(float); alive[17] = np.zeros(ND)
    K = dp_score.KMAX; zero = (np.zeros(K), np.zeros(K))
    Vstart = np.zeros((ND, 3)); cont = {}
    for s in range(17, -1, -1):
        Vend = (trans[s] * alive[s][None, :]) @ Vstart
        Vs = np.zeros((ND, 3))
        for d in range(ND):
            h, _ = n.get((s, d), zero)
            gg, pp = P[(s, d)][:2] if (s, d) in P else (np.zeros(0), np.zeros(0))
            C = Vend[d].copy(); seq = [None] * K
            for k in range(K - 1, -1, -1):
                seq[k] = C.copy()                         # continuation after opportunity k: C_{k+1}
                U = C.copy()
                if len(gg):
                    for c in (2, 1):
                        U[c] = C[c] + np.maximum(0, pp * gg - (1 - pp) * (C[c] - C[c - 1])).mean()
                C = h[k] * U + (1 - h[k]) * Vend[d]
            cont[(s, d)] = seq; Vs[d] = C
        Vstart = Vs
    vstar = Vstart[DM, 2]
    occ = np.zeros((ND, 3)); occ[DM, 2] = 1.0
    vdel = reg = bnd = loc = 0.0
    for s in range(18):
        for d in range(ND):
            if occ[d].sum() == 0 or (s, d) not in P:
                continue
            gg, pp, gq, dens, sup = P[(s, d)]
            go_d = pp >= delta / (gg + delta)
            pfail = (go_d * (1 - pp)).mean(); gain = (go_d * pp * gg).mean()
            h, S = n.get((s, d), (np.zeros(dp_score.KMAX), np.zeros(dp_score.KMAX)))
            start = occ[d].copy(); end = np.zeros(3)
            for k in range(dp_score.KMAX):
                if S[k] <= 0:
                    end += (S[k - 1] if k else 1.0) * occ[d]
                    break
                end += ((S[k - 1] if k else 1.0) - S[k]) * occ[d]          # P(N = k): the half ends before opportunity k
                W = cont[(s, d)][k]
                for c in (1, 2):
                    pc = S[k] * occ[d, c]                                  # opportunity k occurs, c held
                    if pc == 0:
                        continue
                    D = W[c] - W[c - 1]
                    go_o = pp * (gg + D) >= D
                    reg += pc * (np.abs(pp * (gg + D) - D) * (go_o != go_d)).mean()
                    q = gg ** 2 * (D - delta) ** 2 / ((gg + D) * (gg + delta) ** 2) / 2
                    lo = np.minimum(D / (gg + D), delta / (gg + delta)); hi = np.maximum(D / (gg + D), delta / (gg + delta))
                    b0 = np.clip((lo / HW).astype(int), 0, NB - 1); b1 = np.clip((hi / HW).astype(int), 0, NB - 1)
                    mid = np.clip(((lo + hi) / 2 / HW).astype(int), 0, NB - 1)
                    bnd += pc * (sup[gq, b0, b1] * q).mean(); loc += pc * (dens[gq, mid] * q).mean()
                vdel += S[k] * (occ[d, 1] + occ[d, 2]) * gain
                occ[d] = [occ[d, 0] + occ[d, 1] * pfail, occ[d, 1] * (1 - pfail) + occ[d, 2] * pfail, occ[d, 2] * (1 - pfail)]
            occ[d] = end if end.sum() > 0 else start
        if s < 17:
            occ = (trans[s] * alive[s][None, :]).T @ occ
    return dict(vstar=vstar, vdelta=vdel, loss=vstar - vdel, identity=reg, bound=bnd, local=loc)


if __name__ == "__main__":
    import sim, dp_score
    from headline import spec_variant, MAIN
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    v = spec_variant(MAIN)
    m, _ = sim.load(v); m = dp_score.add_dstart(m); m = sim.opportunities(m)
    rng = np.random.default_rng(5)
    rows = []
    for r in range(R):
        _, _, pt = sim.draw(m, rng, v)
        for home in (True, False):
            P, n = pools(m, pt, home)
            for dl in DELTAS:
                x = mdp(P, n, dl); x.update(r=r, home=home, delta=float(dl)); rows.append(x)
            print(r, home, flush=True)
    df = pd.DataFrame(rows)
    agg = df.groupby("delta")[["vstar", "vdelta", "loss", "identity", "bound", "local"]].sum() / R   # both sides, per game
    agg["loss_pct"] = 100 * agg.loss / agg.vstar; agg["bound_pct"] = 100 * agg.bound / agg.vstar
    agg["local_pct"] = 100 * agg.local / agg.vstar; agg["identity_gap"] = (agg.identity - agg.loss).abs()
    best = agg.loss.idxmin()
    out = dict(R=R, variant=MAIN, best_delta=float(best), table=agg.reset_index().round(7).to_dict("records"),
               best=agg.loc[best].round(7).to_dict(), max_identity_gap=float(agg.identity_gap.max()),
               note="model world = regulation-only score-aware DP; values per game, both team sides summed, WP points")
    json.dump(out, open("results/flat_bound.json", "w"), indent=1)
    print(agg.round(5).to_string())
