"""HONZA-73 steps 1 and 3: alternative perception laws and player heterogeneity, all fitted with the exact joint
likelihood of lik.py on the 2026 eligible opportunities.  Reports log-likelihood, AIC/BIC, sigma_r with a Wald CI,
and the calibration of the challenge rate and of the overturn rate given a challenge (Hosmer-Lemeshow-type
chi-square over pre-set bins: |w|, challenges left, inning, stake tercile, zone region).
usage: python src/robust_fit.py [variant ...]  -> data/th_<variant>.npy, data/re_<side>.parquet,
       results/perception_variants.json, results/perception_calibration.csv"""
import json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "src")
import lik

VARIANTS = {
    "sm_edge": {},                        # smooth truth model, offset and scale by edge (HONZA-71), no pi: LR check
    "sm_region": {},                      # smooth truth by zone region + pi (pre-HONZA-88 baseline): LR check
    "normal": {},                         # baseline (HONZA-88): deterministic truth W = 1{w > 0} + ruling mislabel pi
    "logistic": {"law": "logistic"},
    "t10": {"law": "t10"},
    "t5": {"law": "t5"},
    "t3": {"law": "t3"},
    "sr_edge": {"sr_edge": True},
    "sr_region": {"sr_region": True},
    "sr_slope": {"sr_slope": True},
    "sr_region_slope": {"sr_region": True, "sr_slope": True},
    "kappa_region": {"kappa_region": True},
    "kappa_sr_region": {"kappa_region": True, "sr_region": True},
    "kappa_sr_region_slope": {"kappa_region": True, "sr_region": True, "sr_slope": True},
    "kappa_sr_region_g": {"kappa_region": True, "sr_region": True, "kappa_g": True},   # HONZA-106: non-linear stake response
    "re": {"kappa_region": True, "sr_region": True, "kappa_g": True, "re": True, "gh": 5},   # player effects on the main fixed part (HONZA-122: + kappa_g)
    # HONZA-114: the noise-law tests on the main specification, and a lapse rate (challenge curve topping out below 1).
    # HONZA-122: "main_" = the new main (kappa_sr_region_g) + one extension; (Fable) a criterion that can say "nothing at
    # stake" (hinge below g = 0.001), alone and with the lapse rate, to test whether the lapse is a criterion effect at
    # zero stakes; (Sonnet) noise that varies with the stake; (Sonnet, Opus) the lapse and the player effects together
    "main_logistic": {"kappa_region": True, "sr_region": True, "kappa_g": True, "law": "logistic"},
    "main_t10": {"kappa_region": True, "sr_region": True, "kappa_g": True, "law": "t10"},
    "main_t5": {"kappa_region": True, "sr_region": True, "kappa_g": True, "law": "t5"},
    "main_t3": {"kappa_region": True, "sr_region": True, "kappa_g": True, "law": "t3"},
    "main_lapse": {"kappa_region": True, "sr_region": True, "kappa_g": True, "lapse": True},
    "main_glow": {"kappa_region": True, "sr_region": True, "kappa_g": True, "kappa_glow": True},
    "main_glow_lapse": {"kappa_region": True, "sr_region": True, "kappa_g": True, "kappa_glow": True, "lapse": True},
    "main_sr_g": {"kappa_region": True, "sr_region": True, "kappa_g": True, "sr_g": True},
    "linear_lapse": {"kappa_region": True, "sr_region": True, "lapse": True},       # the HONZA-114 lapse model (linear log g)
    "re_lapse": {"kappa_region": True, "sr_region": True, "kappa_g": True, "re": True, "gh": 5, "lapse": True},
    # HONZA-132 (Sol #7, Astra #6): the BIC-best model, player effects + lapse, at 11 x 11 Gauss-Hermite nodes (5 x 5
    # moved the variance components); and a time-varying criterion and noise (second-half shifts of the intercept,
    # the log g slope and log sigma_r), alone and with the lapse
    "re_lapse_gh11": {"kappa_region": True, "sr_region": True, "kappa_g": True, "re": True, "gh": 11, "lapse": True},
    "re_gh11": {"kappa_region": True, "sr_region": True, "kappa_g": True, "re": True, "gh": 11},
    # HONZA-134 (Opus round-5 review): game-state terms of the criterion (|score|, late and close, two outs, RISP,
    # runner on first), on the main model and on the player-effects + lapse model
    "main_state": {"kappa_region": True, "sr_region": True, "kappa_g": True, "kappa_state": True},
    "re_lapse_state_gh11": {"kappa_region": True, "sr_region": True, "kappa_g": True, "re": True, "gh": 11, "lapse": True,
                            "kappa_state": True},
    "main_time": {"kappa_region": True, "sr_region": True, "kappa_g": True, "kappa_time": True, "sr_time": True},
    "main_time_lapse": {"kappa_region": True, "sr_region": True, "kappa_g": True, "kappa_time": True, "sr_time": True,
                        "lapse": True},
}


def bins_table(e, pch, p1):
    """HONZA-122 (Sol, Astra reviews): the overturn check compares the observed overturn rate of a bin with the rate
    among the MODEL's challenges, sum(p1)/sum(pch). The old prediction averaged p1/pch over the pitches actually
    challenged, which under the deterministic truth is pi or 1 - pi on every pitch: it checked the geometry, not the
    selection model."""
    e = e.copy(); e["pch"] = pch; e["p1"] = p1; e["q"] = p1 / np.clip(pch, 1e-300, None)
    e["wbin"] = pd.cut(e.w_fit, [-8, -1, 0, 0.5, 1, 1.5, 2, 3, 4, 8])
    e["inn"] = pd.cut(e.inning, [0, 3, 6, 8, 30], labels=["1-3", "4-6", "7-8", "9+"])
    e["gbin"] = pd.qcut(e.g, 3, labels=["low g", "mid g", "high g"])
    rows, st = [], dict(chi2_challenge=0.0, bins_challenge=0, chi2_overturn=0.0, bins_overturn=0)
    for key in ("wbin", "left", "inn", "gbin", "reg"):
        for lv, d in e.groupby(key, observed=True):
            c = d[d.ch]
            rows.append(dict(by=key, level=str(lv), n=len(d), ch_obs=int(d.ch.sum()), ch_pred=float(d.pch.sum()), n_ch=len(c),
                             over_obs=float(c.ch_over.mean()) if len(c) else np.nan,
                             over_pred=float(d.p1.sum() / d.pch.sum()) if d.pch.sum() > 0 else np.nan))
            v = (d.pch * (1 - d.pch)).sum()
            if v > 0:
                st["chi2_challenge"] += (d.ch.sum() - d.pch.sum()) ** 2 / v; st["bins_challenge"] += 1
            if len(c) >= 20:
                q = d.p1.sum() / d.pch.sum()
                st["chi2_overturn"] += (c.ch_over.sum() - len(c) * q) ** 2 / (len(c) * q * (1 - q)); st["bins_overturn"] += 1
    return pd.DataFrame(rows), st


def side_frames(m, data, th, spec):
    """row-aligned frames (lik.prepare order) with w under the fitted truth offsets."""
    P = lik.unpack(np.asarray(th), spec); out = {}
    for side in ("off", "def"):
        e = m[lik.eligible_mask(m, side)]
        e = e.iloc[np.argsort(~e.ch.values, kind="stable")].copy()
        mu = lik.truth_mu(th, spec, e.edge.values == "vert", data[side]["reg"])
        dc = e.dist_in.values - mu; e["w_fit"] = np.where(e.orig_strike.values, -dc, dc)
        e["reg"] = np.array(lik.REGION_NAMES)[data[side]["reg"]]   # HONZA-130 (Fable review): inside/outside, not arm/glove
        out[side] = e
    return out


def evaluate(m, data, th, spec, name):
    fr = side_frames(m, data, th, spec); stats, tabs = {}, []
    for side in ("off", "def"):
        post = lik.re_posterior(data, th, side, spec) if spec.get("re") else None
        pch, p1 = lik.predict(data, th, spec, side)
        tab, st = bins_table(fr[side], pch, p1)
        tabs.append(tab.assign(side=side, variant=name))
        c = fr[side].ch.values
        st.update(ch_obs=int(c.sum()), ch_pred=float(pch.sum()), over_obs=float(fr[side].ch_over.values[c].mean()),
                  over_pred=float(p1.sum() / pch.sum()), over_pred_given_observed_challenges=float((p1[c] / pch[c]).mean()),
                  logscore_overturn=float(np.mean(np.where(
                      fr[side].ch_over.values[c], np.log(p1[c] / pch[c]), np.log(1 - p1[c] / pch[c])))))
        if post is not None:
            # HONZA-122: one posterior file per random-effects variant (re_lapse used to overwrite re's)
            post.to_parquet(f"data/re_{side}.parquet" if name == "re" else f"data/{name}_{side}.parquet")
            sr0 = float(np.exp(lik.unpack(np.asarray(th), spec)[side]["lsr"]))
            big = post[post.n_ch >= 15]
            st.update(re_players=len(post), re_players_15ch=len(big),
                      sr_player_quantiles_15ch=dict(zip(["p5", "p25", "p50", "p75", "p95"],
                                                        np.percentile(sr0 * np.exp(big.b), [5, 25, 50, 75, 95]).round(3).tolist())),
                      kappa_shift_quantiles_15ch=dict(zip(["p5", "p50", "p95"], np.percentile(big.a, [5, 50, 95]).round(3).tolist())))
        stats[side] = st
    return stats, pd.concat(tabs)


if __name__ == "__main__":
    only = sys.argv[1:] or list(VARIANTS)
    m = pd.read_parquet("data/called_2026_w.parquet")
    data = lik.prepare(m, players=True)
    n_obs = sum(data[s]["n"] for s in data)
    base = lik.base_theta()
    summary = json.load(open("results/perception_variants.json")) if os.path.exists("results/perception_variants.json") else {}
    summary = {k: v for k, v in summary.items() if k in VARIANTS}       # drop entries of retired variants (HONZA-88)
    cal_path = "results/perception_calibration.csv"
    cal = pd.read_csv(cal_path) if os.path.exists(cal_path) else pd.DataFrame()
    if len(cal):
        cal = cal[cal.variant.isin(list(VARIANTS))]
    for name in only:
        spec = lik.spec_of(name); t = time.time()
        base = lik.base_theta()
        if os.path.exists("data/th_normal.npy") and spec.get("det"):
            base = np.load("data/th_normal.npy")
        if (name.startswith("main_") or name in ("re_lapse", "linear_lapse", "re_lapse_state_gh11")) and os.path.exists("data/th_kappa_sr_region_g.npy"):
            # warm start: the common leading block of every side from the main fit (or the player-effects fit for
            # re_lapse), the extensions at their init values
            src = ("re" if name == "re_lapse" and os.path.exists("data/th_re.npy")
                   else "re_lapse_gh11" if name == "re_lapse_state_gh11"       # HONZA-134: + the state terms
                   else "kappa_sr_region" if name == "linear_lapse" else "kappa_sr_region_g")
            t0 = np.load(f"data/th_{src}.npy"); nt = lik.n_truth(spec); ns0 = lik.n_side(lik.spec_of(src)); ns = lik.n_side(spec)
            th0 = lik.init(spec); th0[:nt] = t0[:nt]
            for j in range(2):
                th0[nt + j * ns: nt + j * ns + ns0] = t0[nt + j * ns0: nt + (j + 1) * ns0]
            r, nll = lik.fit(data, spec, th0=th0, verbose=True)
        elif name.endswith("_gh11") and os.path.exists(f"data/th_{name[:-5]}.npy"):
            # HONZA-132: the same parameters at finer quadrature, warm-started from the 5 x 5 fit
            r, nll = lik.fit(data, spec, th0=np.load(f"data/th_{name[:-5]}.npy"), verbose=True)
        elif name == "re" and os.path.exists("data/th_kappa_sr_region_g.npy"):
            # side block of re = [beta, lsr, kreg, region, kg, re]: the main fit's block, then the player-effect params
            t0 = np.load("data/th_kappa_sr_region_g.npy"); nt = lik.n_truth(spec); ns = lik.n_side(dict(spec, re=False))
            th0 = np.r_[t0[:nt], t0[nt:nt + ns], np.log(0.3), np.log(0.2), 0.0, t0[nt + ns:nt + 2 * ns], np.log(0.3), np.log(0.2), 0.0]
            r, nll = lik.fit(data, spec, th0=th0, verbose=True)
        else:
            r, nll = lik.fit(data, spec, base=base, verbose=True)     # init() takes the last 14 entries (beta, log sr)
        np.save(f"data/th_{name}.npy", r.x)
        k = len(r.x)
        out = dict(spec=spec, nll=float(r.fun), k=k, aic=float(2 * r.fun + 2 * k), bic=float(2 * r.fun + k * np.log(n_obs)),
                   converged=bool(r.success), grad_max=float(np.abs(r.jac).max()), params=lik.readable(r.x, spec))
        if spec.get("re") and getattr(r, "hess_fd", None) is not None:
            np.save(f"data/cov_{name}.npy", np.linalg.inv(r.hess_fd))      # HONZA-132: FD Hessian of the exact gradient
        if not spec.get("re"):
            cov = np.linalg.inv(lik.hessian(nll, r.x)); np.save(f"data/cov_{name}.npy", cov)
            i = lik.n_truth(spec)
            for side in ("off", "def"):
                se = float(np.sqrt(cov[i + 6, i + 6])); sr = out["params"][f"{side}_sr"]
                out[f"{side}_sr_ci95"] = [sr * np.exp(-1.96 * se), sr * np.exp(1.96 * se)]
                i += lik.n_side(spec)
        out["calibration"], tab = evaluate(m, data, r.x, spec, name)
        out["seconds"] = round(time.time() - t, 1)
        summary[name] = out
        json.dump(summary, open("results/perception_variants.json", "w"), indent=1, default=float)
        cal = pd.concat([cal[cal.variant != name] if len(cal) else cal, tab]); cal.to_csv(cal_path, index=False)
        print(name, json.dumps({k: out.get(k) for k in ("nll", "k", "aic", "bic", "off_sr_ci95", "def_sr_ci95")}, default=float), flush=True)
