"""known-truth recovery for the primary model (player
effects + lapse, 11 x 11 nodes). Replicate i simulates the challenges of the 2026 season from the fitted primary model
(coverage_stream.behave: truth from the zone rule and the mislabel rate, signal and criterion from the fitted
parameters with every player at his posterior mean, lapses, the challenges the simulated team holds), refits the
primary model from its estimate, and records the readable parameters (sigma_r, lapse rates, the player-effect SDs and
their correlation).
usage: python src/recovery_primary.py run <i0> <n>  -> data/recovery_primary_<i0>.json
       python src/recovery_primary.py summary       -> results/recovery_primary.json"""
import glob, json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "src")
NAME = "re_lapse_gh11"


def flat(P):
    out = {}
    for side in ("off", "def"):
        out[f"{side}_sr"] = P[f"{side}_sr"]; out[f"{side}_lapse"] = P[f"{side}_lapse"]
        out[f"{side}_sd_kappa"] = P[f"{side}_re"]["sd_kappa"]; out[f"{side}_sd_logsr"] = P[f"{side}_re"]["sd_logsr"]
        out[f"{side}_re_rho"] = P[f"{side}_re"]["rho"]
    return out


def run(i0, n):
    import lik
    from coverage_stream import behave
    spec = lik.spec_of(NAME); th = np.load(f"data/th_{NAME}.npy")
    path = f"data/recovery_primary_{i0}.json"
    st = json.load(open(path)) if os.path.exists(path) else dict(rows=[])
    done = {r["i"] for r in st["rows"]}
    m = pd.read_parquet("data/called_2026_w.parquet")
    for i in range(i0, i0 + n):
        if i in done:
            continue
        t = time.time()
        f, _ = behave(m, NAME, 31000 + i)
        r, _ = lik.fit(lik.prepare(f, players=True), spec, th0=th, maxiter=60)
        st["rows"].append(dict(i=i, ok=bool(r.success), secs=round(time.time() - t), est=flat(lik.readable(r.x, spec))))
        json.dump(st, open(path, "w"))
        print("recovery", i, st["rows"][-1]["secs"], st["rows"][-1]["est"], flush=True)


def summary():
    import lik
    spec = lik.spec_of(NAME)
    true = flat(lik.readable(np.load(f"data/th_{NAME}.npy"), spec))
    rows = [r for p in glob.glob("data/recovery_primary_*.json") for r in json.load(open(p))["rows"]]
    out = dict(model=NAME, replicates=len(rows), converged=sum(r["ok"] for r in rows), params={})
    for k, v0 in true.items():
        e = np.array([r["est"][k] for r in rows])
        out["params"][k] = dict(true=v0, mean=float(e.mean()), sd=float(e.std(ddof=1)) if len(e) > 1 else None,
                                bias=float(e.mean() - v0), rel_bias=float(e.mean() / v0 - 1) if v0 else None)
    json.dump(out, open("results/recovery_primary.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(int(sys.argv[2]), int(sys.argv[3]))
    else:
        summary()
