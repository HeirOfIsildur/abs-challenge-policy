"""Win-probability engine per format and the stake of every ball since Sep 2017: g = V(state, w) - V(state, w+1), the batting
team's loss of match value from one more wicket at the state just after the ball.
V = P(win) in ODI/T20 (ties count 1/2) and expected match points P(win) + P(draw)/2 in Tests.
Gradient boosting, monotone in wickets (-), runs (+) and lead (+); trained on men's matches since 2010 (no DLS
results), with one end-of-innings row per innings so that the all-out state is in the support.
An isotonic map fitted on the held-out fold (20 % of matches) recalibrates V.
Held-out Brier score and calibration go to results/drs/wp_check.json.  The recalibrated diagnostics (brier_recal,
calibration_recal) are cross-fitted: the held-out matches are split in two halves, the isotonic map is fitted
on one half and scored on the other, and both directions are pooled, so no match scores a map fitted on itself.  The
map used for the stakes is still the one fitted on the whole held-out fold.
usage: python src/drs/wp.py  ->  data/drs_balls_g.parquet"""
import json
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

STR = ["fmt", "comp", "bat", "bowl", "wk_kind", "rev_by", "rev_dec"]

def load(fmt):   # one format at a time, strings as categories (memory)
    d = pd.read_parquet("data/drs_balls.parquet", filters=[("fmt", "==", fmt)])
    for c in STR: d[c] = d[c].astype("category")
    d["wkts_c"] = d.wkts.clip(0, 10)
    return d

def feats(d, fmt, w):
    if fmt == "Test":   # innings no., wickets, runs this innings, lead (bat - bowl, match aggregate), match overs elapsed
        return np.c_[d.inn, w, d.runs, d.lead, d.match_balls / 6]
    bl = d.max_overs.fillna(50 if fmt == "ODI" else 20) * 6 - d.balls
    need = np.where(d.inn == 1, d.target - d.runs, np.nan)
    return np.c_[d.inn, w, d.runs, bl, need]

MONO = {"Test": [0, -1, 1, 1, 0], "ODI": [0, -1, 1, 1, -1], "T20": [0, -1, 1, 1, -1]}
chk, parts = {}, []
for fmt in ("ODI", "T20", "Test"):
    d = load(fmt)
    if fmt != "Test": d = d[d.inn <= 1]
    tr = d[(d.date >= "2010-01-01") & ~d.dl & d.pts.notna()]
    end = tr.groupby(["match", "inn"]).tail(1).copy()        # innings-end state (after the last ball)
    end["runs"] += end.r_ball; end["lead"] += end.r_ball; end["balls"] += 1; end["match_balls"] += 1
    end["wkts_c"] = (end.wkts + end.n_wk).clip(0, 10)
    tr = pd.concat([tr.sample(frac=0.15, random_state=1), end])
    mids = tr.match.unique(); rng = np.random.default_rng(0)
    test_m = set(rng.choice(mids, len(mids) // 5, replace=False))
    ho = tr.match.isin(test_m)
    mk = lambda: HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08, max_leaf_nodes=48,
                                               monotonic_cst=MONO[fmt], random_state=0)
    m = mk().fit(feats(tr[~ho], fmt, tr[~ho].wkts_c), tr[~ho].pts)
    p = np.clip(m.predict(feats(tr[ho], fmt, tr[ho].wkts_c)), 0, 1); y = tr[ho].pts.values
    bins = pd.cut(p, np.linspace(0, 1, 11), include_lowest=True)
    cal = pd.DataFrame(dict(p=p, y=y, b=bins)).groupby("b", observed=True).agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
    chk[fmt] = dict(brier=float(np.mean((p - y) ** 2)), brier_const=float(np.var(y)), n_test=int(len(y)),
                    calibration=cal.round(3).reset_index(drop=True).to_dict("list"))
    mk_iso = lambda: IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    hm = tr[ho].match.values; hmu = np.array(sorted(set(hm))); rng.shuffle(hmu)
    half = np.isin(hm, hmu[: len(hmu) // 2])                          # calibration / test halves of the held-out matches
    pi = np.empty_like(p)
    for a_, b_ in ((half, ~half), (~half, half)):                     # cross-fit: map fitted on one half, scored on the other
        pi[b_] = mk_iso().fit(p[a_], y[a_]).predict(p[b_])
    chk[fmt]["brier_recal_insample"] = float(np.mean((mk_iso().fit(p, y).predict(p) - y) ** 2))   # old (in-sample) number
    chk[fmt]["recal_scheme"] = "cross-fitted over two halves of the held-out matches"
    chk[fmt]["n_test_matches"] = int(len(hmu))
    iso = mk_iso().fit(p, y)                                          # production map: whole held-out fold
    cal_i = pd.DataFrame(dict(p=pi, y=y, b=pd.cut(pi, np.linspace(0, 1, 11), include_lowest=True)))
    chk[fmt]["brier_recal"] = float(np.mean((pi - y) ** 2))
    chk[fmt]["calibration_recal"] = cal_i.groupby("b", observed=True).agg(pred=("p", "mean"), obs=("y", "mean")).round(3).reset_index(drop=True).to_dict("list")
    m = mk().fit(feats(tr, fmt, tr.wkts_c), tr.pts)            # refit on all, then recalibrate
    d = d[d.date >= "2017-09-01"].copy()                       # stakes only needed in the DRS-analysis era
    a = d.copy()                                               # state just after the ball, w vs w+1
    a["runs"] += a.r_ball; a["lead"] += a.r_ball; a["balls"] += 1; a["match_balls"] += 1
    w0 = a.wkts.clip(0, 9)
    # no recalibration. Scored on held-out matches the isotonic map does not improve the Brier score (it
    # worsens it in Tests), so the stakes use the engine's own predictions, clipped to [0, 1]; the map is kept only as
    # a diagnostic in wp_check.json
    raw = lambda X: np.clip(m.predict(X), 0, 1)
    v0 = raw(feats(a, fmt, w0)); v1 = raw(feats(a, fmt, w0 + 1))
    chk[fmt]["stakes_recalibrated"] = False
    d["V0"], d["g"] = v0, np.maximum(v0 - v1, 0)
    parts.append(d)
    print(fmt, chk[fmt]["brier"], chk[fmt]["brier_const"], "median g", np.median(d.g))
out = pd.concat(parts)
for c in STR: out[c] = out[c].astype(str)
out.to_parquet("data/drs_balls_g.parquet")
json.dump(chk, open("results/drs/wp_check.json", "w"), indent=1)
