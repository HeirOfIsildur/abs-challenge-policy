"""Calibration of the DRS WP engine against realised results (HONZA-89 audit, item 12; does not import src/).
V0 = engine value just after each ball (batting side; P(win), ties 1/2; Tests: expected points), from
data/drs_balls_g.parquet; y = realised points.  Binned by deciles of V0, by innings and by era, for the analysis
era (Sep 2017 on, no DLS).  Also the realised stake check: P(win | wicket on this ball) vs P(win | no wicket) at matched
V0 bins, against the mean engine g.
usage: .venv/bin/python src/drs/wp_calib.py  ->  results/drs/wp_calib.json"""
import json
import numpy as np, pandas as pd

b = pd.read_parquet("data/drs_balls_g.parquet")
b = b[~b.dl & b.pts.notna()]
out = {}
for fmt, d in b.groupby("fmt"):
    d = d.copy()
    # one ball per over is plenty (autocorrelated within match)
    d = d[d.balls % 6 == 0]
    d["bin"] = pd.cut(d.V0, np.linspace(0, 1, 11), include_lowest=True)
    t = d.groupby("bin", observed=True).agg(pred=("V0", "mean"), obs=("pts", "mean"), n=("pts", "size"))
    # match-cluster SE of obs per bin
    se = d.groupby("bin", observed=True).apply(lambda g: g.groupby("match").pts.mean().std(ddof=1) / np.sqrt(g.match.nunique()))
    t["se"] = se; t["z"] = (t.obs - t.pred) / t.se
    by_inn = d.groupby("inn").apply(lambda g: float((g.pts - g.V0).mean())).round(4).to_dict()
    by_era = d.groupby(d.date.dt.year >= 2022).apply(lambda g: float((g.pts - g.V0).mean())).round(4).to_dict()
    out[fmt] = dict(n=int(len(d)), matches=int(d.match.nunique()), brier=float(((d.pts - d.V0) ** 2).mean()),
                    mean_resid=float((d.pts - d.V0).mean()), max_abs_z=float(t.z.abs().max()),
                    bins=t.round(4).reset_index(drop=True).to_dict("list"),
                    resid_by_inn={int(k): v for k, v in by_inn.items()}, resid_2022plus={str(k): v for k, v in by_era.items()})
    print(fmt, "n", len(d), "brier", round(out[fmt]["brier"], 4), "mean resid", round(out[fmt]["mean_resid"], 4),
          "max|z|", round(out[fmt]["max_abs_z"], 2)); print(t.round(3).to_string())
    print(" resid by inn", out[fmt]["resid_by_inn"], "era>=2022", out[fmt]["resid_2022plus"])
json.dump(out, open("results/drs/wp_calib.json", "w"), indent=1)
