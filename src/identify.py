"""HONZA-73 step 4: known-truth recovery.  Simulate challenge decisions and ABS rulings from the fitted baseline
(region truth + ruling contamination, Gaussian perception) on the real 2026 opportunities (same w, state, stakes,
eligibility), refit from a deliberately wrong start (sigma_r x 1.5, sigma_m x 2), and compare.  Also two
misspecified data-generating processes fitted with the baseline model: t5 perception noise, and player-level
heterogeneity in kappa and log sigma_r (SDs from the random-effects fit).  Plus the profile likelihood of each
side's sigma_r on the real data.
HONZA-88: the baseline is the deterministic truth (W = 1{w > 0}, mislabel pi), so the recovery is against a known
truth; an extra misspecified DGP adds 0.3 in of tracking noise between the measured and the ABS location.
usage: python src/identify.py [R]  -> results/identification.json"""
import json, sys
import numpy as np, pandas as pd
import jax, jax.numpy as jnp
from scipy.optimize import minimize
sys.path.insert(0, "src")
import lik

SPEC = lik.spec_of("normal")      # HONZA-88: deterministic truth W = 1{w > 0} + ruling mislabel pi


def simulate(data, th, rng, law="normal", re_sd=None, coord_sd=0.0):
    """new (ch, y) per side under theta; returns a re-sorted copy of data (challenged rows first).
    coord_sd > 0 (in): misspecified DGP in which the tracked location is off the ABS location by N(0, coord_sd^2)."""
    P = lik.unpack(np.asarray(th), SPEC); pi = 1 / (1 + np.exp(-float(P["lpi"])))
    out = {}
    for side in ("off", "def"):
        D = data[side]; n = D["n"]
        mu = np.asarray(P["mu"])[D["reg"]]; sm = np.exp(np.asarray(P["lsm"]))[D["reg"]]
        dc = D["d"] - mu; w = np.where(D["os"] > 0, -dc, dc)
        kap = D["X"] @ np.asarray(P[side]["beta"]); sr = np.exp(float(P[side]["lsr"])) * np.ones(n)
        if re_sd is not None:
            npl = len(D["players"]); a = rng.normal(0, re_sd[side][0], npl); b = rng.normal(0, re_sd[side][1], npl)
            kap = kap + a[D["pidx"]]; sr = sr * np.exp(b[D["pidx"]])
        eta = rng.normal(0, 1, n) * np.sqrt(sm ** 2 + coord_sd ** 2)
        e = rng.standard_t(5, n) * np.sqrt(3 / 5) if law == "t5" else rng.normal(0, 1, n)
        W = (w + eta) > 0
        ch = (w + eta + sr * e) > kap
        y = np.where(rng.random(n) < pi, ~W, W)
        o = np.argsort(~ch, kind="stable")
        out[side] = {k: (v[o] if isinstance(v, np.ndarray) and len(v) == n else v) for k, v in D.items()}
        out[side].update(ch=ch[o], y=y[o], nch=int(ch.sum()))
    return out


def key(th):
    r = lik.readable(th, SPEC)
    return dict(off_sr=r["off_sr"], def_sr=r["def_sr"], off_k0=r["off_beta"][0], def_k0=r["def_beta"][0], pi=r["contam_pi"])


def bad_start(th):
    t = np.array(th, float); nt = lik.n_truth(SPEC)
    t[0] += 1.0                                                  # pi x ~2.7 on the odds scale
    t[nt + 6] += np.log(1.5); t[nt + 7 + 6] += np.log(1.5)       # sigma_r x 1.5
    return t


def profile(data, th, side, grid):
    """profile negative log-likelihood of sigma_r(side): fix log sr, minimise over everything else."""
    nt = lik.n_truth(SPEC); j = nt + 6 if side == "off" else nt + 13
    nll = lik.make_nll(data, SPEC)
    free = np.array([i for i in range(len(th)) if i != j])
    out = []
    for sr in grid:
        def f(x, sr=sr):
            return nll(jnp.asarray(th).at[free].set(x).at[j].set(np.log(sr)))
        vg = jax.jit(jax.value_and_grad(f))
        r = minimize(lambda x: tuple(np.asarray(v, float) for v in vg(jnp.asarray(x))), np.asarray(th)[free], jac=True,
                     method="L-BFGS-B", options=dict(maxiter=500))
        out.append((float(sr), float(r.fun)))
    return out


if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    m = pd.read_parquet("data/called_2026_w.parquet")
    data = lik.prepare(m, players=True)
    th = np.load("data/th_normal.npy")
    res = dict(truth=key(th))
    re_sd = None
    try:
        v = json.load(open("results/perception_variants.json"))["re"]["params"]
        re_sd = {s: (v[f"{s}_re"]["sd_kappa"], v[f"{s}_re"]["sd_logsr"]) for s in ("off", "def")}
        res["re_sd_used"] = re_sd
    except (FileNotFoundError, KeyError):
        pass
    rng = np.random.default_rng(2026)
    for dgp in ["correct", "t5_noise", "coord_noise_0.3in"] + (["player_heterogeneity"] if re_sd else []):
        rows = []
        for r in range(R if dgp == "correct" else max(R // 2, 4)):
            sim = simulate(data, th, rng, law="t5" if dgp == "t5_noise" else "normal",
                           re_sd=re_sd if dgp == "player_heterogeneity" else None,
                           coord_sd=0.3 if dgp.startswith("coord_noise") else 0.0)
            fr, _ = lik.fit(sim, SPEC, th0=bad_start(th))
            rows.append(dict(key(fr.x), converged=bool(fr.success)))
            print(dgp, r, {k: round(v, 3) for k, v in rows[-1].items() if k != "converged"}, flush=True)
        df = pd.DataFrame(rows)
        res[dgp] = dict(n=len(df), converged=int(df.converged.sum()),
                        mean={k: float(df[k].mean()) for k in res["truth"]}, sd={k: float(df[k].std()) for k in res["truth"]},
                        bias_pct={k: float(100 * (df[k].mean() / res["truth"][k] - 1)) for k in res["truth"]})
        json.dump(res, open("results/identification.json", "w"), indent=1)
    for side, centre in (("off", res["truth"]["off_sr"]), ("def", res["truth"]["def_sr"])):
        grid = centre * np.array([0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2])
        pr = profile(data, th, side, grid)
        best = min(v for _, v in pr)
        res[f"profile_{side}"] = [(s, round(v - best, 2)) for s, v in pr]
        print("profile", side, res[f"profile_{side}"], flush=True)
        json.dump(res, open("results/identification.json", "w"), indent=1)
