"""challenge counts of the co-primary model (player effects + lapse, 11 x 11 nodes) integrated over the
fitted population law and with each player's posterior mean plugged in, against the observed. -> results/re_calibration.json"""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
import lik
out = {}
m = pd.read_parquet("data/called_2026_w.parquet")
for name in ("re_lapse_gh11", "kappa_sr_region_g"):
    spec = lik.spec_of(name); th = np.load(f"data/th_{name}.npy")
    data = lik.prepare(m, players=bool(spec.get("re")))
    for side in ("off", "def"):
        pi, _ = lik.predict(data, th, spec, side)
        r = dict(observed=int(data[side]["nch"]), integrated=float(pi.sum()))
        if spec.get("re"):
            post = pd.read_parquet(f"data/{name}_{side}.parquet").set_index("player").reindex(data[side]["players"]).reset_index()
            r["plug_in"] = float(lik.predict(data, th, spec, side, re_post=post)[0].sum())
        out[f"{name}_{side}"] = r
json.dump(out, open("results/re_calibration.json", "w"), indent=1)
print(json.dumps(out, indent=1))
