"""Known-truth recovery of the DRS selection model (HONZA-89 audit, item 12).
Simulate outcome counts from the fitted model on the real opportunity set (same cells, same kinds of non-overturned
decisions drawn from the fitted pi_lbw), refit with fit_full (profile in s + free fit), and report the recovered s,
kappa, mu and how often the 95 % LR set (continuous profile endpoints) covers the truth.  Run for the band model (main) and, as a contrast, the
constant-rho model at the same truth.
usage: python src/drs/identify.py [R] [fmt...]  ->  results/drs/identification.json"""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src/drs")
from model import fit_full, cell_probs, cells, OUT

R = int(sys.argv[1]) if len(sys.argv) > 1 else 20
FMTS = sys.argv[2:] or ["Test", "ODI", "T20"]
inv = {v: k for k, v in OUT.items()}

def simulate(d, P, rng):
    """kind ~ pi_lbw, outcome ~ cell probabilities of that kind; overturned decisions lose their kind label"""
    Pu = dict(P, tau=np.array(P["tau"]), mu=np.array(P["mu"]))
    cp = cell_probs(Pu, "band", P["rho"]); c = cells(d)
    kind = (rng.random(len(d)) > P["pi_lbw"]).astype(int)
    pr = np.stack([np.stack(cp[k], axis=1)[c] for k in (0, 1)])[kind, np.arange(len(d))]
    pr = np.clip(pr, 0, None); pr /= pr.sum(1, keepdims=True)
    o = (rng.random(len(d))[:, None] > np.cumsum(pr, 1)).sum(1)
    out = pd.Series(o).map(inv).values
    return d.assign(kind=np.where(out == "success", "", np.where(kind == 0, "lbw", "caught")), outcome=out)

res = {}
if __name__ == "__main__":
    o = pd.read_parquet("data/drs_opps.parquet"); o = o[o.side == "bat"]
    for fmt in FMTS:
        d = o[(o.fmt == fmt) & (o.c > 0)].reset_index(drop=True)
        P = json.load(open(f"results/drs/model_{fmt}.json")); rng = np.random.default_rng(89)
        reps = []
        for r in range(R):
            ds = simulate(d, P, rng); F = fit_full(ds, "band")
            lo, hi = F["s_95"]                       # continuous profile-LR endpoints (HONZA-122; was a grid with a 15 % slack)
            reps.append(dict(s=F["s"], kappa=F["kappa"], mu=F["mu"], conv=F["conv"], lr_set=[lo, hi], lr_open=F["s_95_open"],
                             covers=bool(lo <= P["s"] <= hi)))
        s = np.array([x["s"] for x in reps])
        res[fmt] = dict(truth=dict(s=P["s"], kappa=P["kappa"], mu=P["mu"]), R=R,
                        s_median=float(np.median(s)), s_q=[float(np.percentile(s, q)) for q in (5, 25, 75, 95)],
                        kappa_median=float(np.median([x["kappa"] for x in reps])),
                        lr_cover_rate=float(np.mean([x["covers"] for x in reps])), reps=reps)
        print(fmt, "truth s", round(P["s"], 3), "median", round(res[fmt]["s_median"], 3), "q5-95", np.round(res[fmt]["s_q"], 3)[[0, 3]],
              "cover", res[fmt]["lr_cover_rate"], flush=True)
    json.dump(res, open("results/drs/identification.json", "w"), indent=1)
