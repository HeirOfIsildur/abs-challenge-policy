"""is the Gaussian (probit) shape of the challenge curve right, not just best within a
parametric family?  Nonparametric check, by side and zone region: the observed challenge indicator on every eligible
called pitch is regressed on the truth margin w by isotonic regression (P(challenge | w) is monotone in w under any
signal-detection model with a location-family noise law), and compared with the fitted main model's challenge
probability Phi((w - kappa(x)) / sigma_r) averaged over the same pitches.  The covariates in kappa(x) vary from pitch to
pitch, so the model curve is the pitch average, smoothed in the same way (isotonic on w) for a like-for-like overlay.
Pointwise 95% bands for the isotonic fit come from a game bootstrap (B = 200, fit only, model held fixed).
usage: python src/isotonic_check.py -> results/isotonic_check.json (curves on a grid + discrepancy summaries)"""
import json, sys
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
sys.path.insert(0, "src")
import sim
from final_ci import spec_of
from headline import MAIN

REG = ["bottom", "top", "inside", "outside"]          # lik.REGION_NAMES (legacy lik.REGIONS keys: arm = inside, glove = outside; )
GRID = np.round(np.arange(-6.0, 6.01, 0.25), 2)


def iso(w, y):
    return IsotonicRegression(increasing=True, out_of_bounds="clip").fit(w, y).predict(GRID)


if __name__ == "__main__":
    m, _ = sim.load(dict(theta=f"data/th_{MAIN}.npy", spec=spec_of(MAIN)))
    m = m[(m.left > 0).values & (m.w.abs() < 6).values].reset_index(drop=True)
    m["chal"] = m.ch.astype(bool).astype(float)
    m["pm"] = norm.cdf((m.w.values - m.kappa.values) / m.sr.values)
    rng = np.random.default_rng(106)
    out = dict(grid=GRID.tolist(), spec=MAIN, B=200, panels={})
    for side in ("off", "def"):
        for r, nm in enumerate(REG):
            e = m[(m.side == side).values & (m.pgrp.astype(int) == r).values].reset_index(drop=True)
            w, y = e.w.values, e.chal.values
            emp, mod = iso(w, y), iso(w, e.pm.values)
            rows = [d.index.values for _, d in e.groupby("game_pk")]
            bs = []
            for _ in range(200):
                idx = np.concatenate([rows[i] for i in rng.integers(0, len(rows), len(rows))])
                bs.append(iso(w[idx], y[idx]))
            lo, hi = np.quantile(np.array(bs), [0.025, 0.975], axis=0)
            # binned calibration: observed vs predicted challenges in w-bins of 0.5 in, Pearson chi-square
            t = pd.DataFrame(dict(b=np.floor(w / 0.5), y=y, p=e.pm.values)).groupby("b").agg(n=("y", "size"), o=("y", "sum"), x=("p", "sum"))
            t = t[t.x > 5]
            chi2 = float((((t.o - t.x) ** 2) / (t.x * (1 - t.x / t.n))).sum())
            inside = (GRID >= np.quantile(w, 0.01)) & (GRID <= np.quantile(w, 0.99))
            p = out["panels"][f"{side}_{nm}"] = dict(
                n=int(len(e)), challenges=int(y.sum()), predicted=float(e.pm.sum()),
                empirical=emp.round(5).tolist(), model=mod.round(5).tolist(), lo=lo.round(5).tolist(), hi=hi.round(5).tolist(),
                max_abs_gap=float(np.abs(emp - mod)[inside].max()),
                share_grid_model_in_band=float(((mod >= lo - 1e-9) & (mod <= hi + 1e-9))[inside].mean()),
                bins=int(len(t)), chi2=chi2)
            print(side, nm, p["n"], p["challenges"], round(p["predicted"], 1), round(p["max_abs_gap"], 4),
                  round(p["share_grid_model_in_band"], 3), p["bins"], round(chi2, 1), flush=True)
    json.dump(out, open("results/isotonic_check.json", "w"), indent=1)
