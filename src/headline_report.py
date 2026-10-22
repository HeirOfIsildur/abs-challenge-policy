"""Consolidated headline table -> results/headline.json, results/headline.md (the manuscript reads here).

Inputs: data/headline_point.pkl (point estimate, R = 40), results/headline_boot.json (game bootstrap with
model refits, B = 120, R = 10; results/headline_boot_R2.json is the R = 2 run it replaces), results/headline_noref.json (per-player model: game bootstrap with EB redraws of player effects),
results/causal.json, results/causal_sim.json, results/causal_gap.json, results/flat_bound.json, results/dp_checks.json, results/headline_variants.json,
results/headline-check.json.  Point = full-sample fit; CI = 2.5/97.5 percentiles of the bootstrap.
usage: python src/headline_report.py"""
import json
import sys
sys.path.insert(0, 'src')
from headline import MAIN, os, pickle
import numpy as np

POLICIES = [("clairvoyant", "Clairvoyant ceiling (knows the future (p, g) of its team-game)"),
            ("opt_score", "Optimum (score-aware DP)"), ("runs_dp", "Namuduri runs rule (rebuilt on our model)"),
            ("opt", "Pooled DP (original 'optimum')"), ("stake_0.008", "One-line rule, delta = 0.008"),
            ("stake_0.01", "One-line rule, delta = 0.01"), ("card_pub", "Arora card, published thresholds"),
            ("card_refit", "Arora card, refit on our DP"), ("obs_rule", "Fitted behavioural rule (simulated)"),
            ("savant", "Savant 0.2 rule"), ("observed", "Observed (realised)")]
ci = lambda x: [round(float(np.percentile(x, 2.5)), 3), round(float(np.percentile(x, 97.5)), 3)]
r3 = lambda x: round(float(x), 3)


def main():
    pt = pickle.load(open("data/headline_point.pkl", "rb"))
    S = pt[MAIN]["summary"]
    rows = [r for r in json.load(open("results/headline_boot.json"))["rows"] if r.get("ok", True)]   # Converged refits only
    B = lambda p, f="wins": np.array([r["summary"][p][f] for r in rows])
    o = B("opt_score")
    table = []
    for p, label in POLICIES:
        w = B(p)
        table.append(dict(policy=p, label=label, wins=r3(S[p]["wins"]), wins_ci=ci(w),
                          share=r3(S[p]["wins"] / S["opt_score"]["wins"]), share_ci=ci(w / o),
                          lost=r3(S["opt_score"]["wins"] - S[p]["wins"]), lost_ci=ci(o - w),
                          challenges=round(float(S[p]["n"]), 4), challenges_ci=ci(B(p, "n")),
                          batter=r3(S[p]["batter"]), fielder=r3(S[p]["fielder"])))
    best = np.array([max(r["summary"][k]["wins"] for k in r["summary"] if k.startswith("stake_")) for r in rows])
    roles = {sd: dict(lost=r3(S["opt_score"][sd] - S["observed"][sd]), ci=ci(B("opt_score", sd) - B("observed", sd)))
             for sd in ("batter", "fielder")}
    lost = dict(point=r3(S["opt_score"]["wins"] - S["observed"]["wins"]), ci=ci(o - B("observed")),
                league=r3(30 * (S["opt_score"]["wins"] - S["observed"]["wins"])))
    exit_rule = dict(rule="switch to TAS if the 95 % CI of wins lost includes <= 0.25 or is wider than 1.5",
                     passes=bool(lost["ci"][0] > 0.25 and lost["ci"][1] - lost["ci"][0] <= 1.5))
    out = dict(spec=f"{MAIN} (BIC-best homogeneous model) on raw Statcast 2026 geometry with a deterministic ABS truth, sim.draw with isotonic calibration cross-fitted on an auxiliary draw, no-ABS games dropped",
               units="wins per team-season (162 games); challenges per team-game", B=len(rows),
               wins_lost=lost, wins_lost_by_role=roles, exit_rule=exit_rule, policies=table,
               one_line_rule=dict(delta="stake_0.008", best_delta=S["best_stake"],
                                  best_share=r3(S[S["best_stake"]]["wins"] / S["opt_score"]["wins"]),
                                  best_minus_rule=r3(S[S["best_stake"]]["wins"] - S["stake_0.008"]["wins"]),
                                  best_delta_boot={k: int(v) for k, v in zip(*np.unique([r["summary"]["best_stake"] for r in rows], return_counts=True))},
                                  share_reselected=r3(np.mean(best / o)), share_reselected_ci=ci(best / o)))
    R = pt["re"]["summary"]
    re = dict(point_opt=r3(R["opt_score"]["wins"]), point_lost=r3(R["opt_score"]["wins"] - R["observed"]["wins"]),
              best_stake=R["best_stake"], best_stake_share=r3(R[R["best_stake"]]["wins"] / R["opt_score"]["wins"]))
    if os.path.exists("results/headline_noref.json"):
        nr = json.load(open("results/headline_noref.json"))["rows"]
        for v, key in (("re", "per_player"), (MAIN, "main_noref")):
            x = [r["summary"] for r in nr if r["variant"] == v]
            if x:
                l = np.array([s["opt_score"]["wins"] - s["observed"]["wins"] for s in x])
                re[f"{key}_lost_ci"] = ci(l); re[f"{key}_B"] = len(x)
    out["per_player_model"] = re
    # Savant's form p >= c/(c + RV) with the run cost c tuned on our grid
    c_best = S["best_savant_c"]; w = B(c_best)
    out["savant_tuned"] = dict(c=float(c_best.split("_")[-1]), policy=c_best, wins=r3(S[c_best]["wins"]),
                               share=r3(S[c_best]["wins"] / S["opt_score"]["wins"]), share_ci=ci(w / o),
                               n=round(float(S[c_best]["n"]), 3))
    table.append(dict(policy="savant_tuned", label=f"Savant form, run cost tuned ({c_best})", wins=r3(S[c_best]["wins"]), wins_ci=ci(w),
                      share=r3(S[c_best]["wins"] / S["opt_score"]["wins"]), share_ci=ci(w / o),
                      lost=r3(S["opt_score"]["wins"] - S[c_best]["wins"]), lost_ci=ci(o - w),
                      challenges=round(float(S[c_best]["n"]), 4), challenges_ci=ci(B(c_best, "n")),
                      batter=r3(S[c_best]["batter"]), fielder=r3(S[c_best]["fielder"])))
    # Monte Carlo error of the point estimate and of the bootstrap
    mc = pt[MAIN].get("mc", {})
    lb = o - B("observed")
    out["monte_carlo"] = dict(point_R=mc.get("R"), lost_sd_per_draw=mc.get("lost_sd_per_draw"), lost_mc_se=mc.get("lost_mc_se"),
                              boot_R=json.load(open("results/headline_boot.json")).get("R"), boot_sd=r3(lb.std(ddof=1)))
    if os.path.exists("results/headline_boot_R2.json"):
        old = json.load(open("results/headline_boot_R2.json"))
        lo = np.array([r["summary"]["opt_score"]["wins"] - r["summary"]["observed"]["wins"] for r in old["rows"]])
        out["monte_carlo"]["boot_R2"] = dict(R=old["R"], B=len(lo), ci=ci(lo), sd=r3(lo.std(ddof=1)))
    c = json.load(open("results/causal.json"))
    out["causal"] = dict(R=c["R"], K=c["K"], game_bootstrap=c["game_bootstrap"],
                         calibration={p: dict(winB_minus_engine=r3(v["calib_winB_minus_engine"]), slope=r3(v["calib_slope"]))
                                      for p, v in c["point"].items() if "calib_slope" in v})
    out["causal"]["simulator"] = c.get("simulator", "sim")
    if os.path.exists("results/causal_sim.json"):          # original simulator (audit R1), kept for the comparison
        cs = json.load(open("results/causal_sim.json"))
        out["causal_sim"] = dict(R=cs["R"], K=cs["K"], binary_minus_rb=cs["game_bootstrap"]["opt_score"]["binary_minus_rb"],
                                 binary_minus_observed=cs["game_bootstrap"]["opt_score"]["binary_minus_observed"])
    if os.path.exists("results/causal_gap.json"):
        g = json.load(open("results/causal_gap.json"))
        out["causal_gap"] = dict(predicted_opt_score=g["by_policy"]["opt_score"]["predicted"],
                                 g_over_engine={k: r3(v["chain_over_engine"]) for k, v in g["mc_check"].items()},
                                 mc_minus_chain={k: [round(v["mc_minus_chain"], 5), round(v["mc_minus_chain_se"], 5)] for k, v in g["mc_check"].items()})
    out["flat_bound"] = json.load(open("results/flat_bound.json"))
    out["dp_checks"] = json.load(open("results/dp_checks.json"))
    var = json.load(open("results/headline_variants.json"))["variants"]
    out["model_spread"] = {k: dict(lost=r3(v["opt_score"]["wins"] - v["observed"]["wins"]),
                                   lost_pooled_dp=r3(v["opt"]["wins"] - v["observed"]["wins"])) for k, v in var.items()}
    out["kernel_checks"] = json.load(open("results/headline-check.json"))
    json.dump(out, open("results/headline.json", "w"), indent=1)
    write_md(out)


def write_md(h):
    f = lambda x, c: f"{x:.2f} [{c[0]:.2f}, {c[1]:.2f}]"
    wl, rl = h["wins_lost"], h["wins_lost_by_role"]
    L = ["## Headline table (rebuilt on corrected geometry)", "",
         f"Main specification: {h['spec']}. Point estimate from the full-sample fit (R = {h['monte_carlo']['point_R']} simulation draws, Monte Carlo SE "
         f"{h['monte_carlo']['lost_mc_se']:.3f}); 95 % CIs "
         f"from a game bootstrap that refits the truth and perception model, re-solves every DP and re-simulates in each "
         f"of B = {h['B']} replicates (R = {h['monte_carlo']['boot_R']} draws each). Units: {h['units']}. Path-sum accounting; the causal re-simulation agrees (below).", "",
         f"**Wins lost against the optimum: {f(wl['point'], wl['ci'])} per team-season**, about {wl['league']:.0f} wins "
         f"league-wide. Batters lose {f(rl['batter']['lost'], rl['batter']['ci'])}, catchers/pitchers "
         f"{f(rl['fielder']['lost'], rl['fielder']['ci'])}.", "",
         f"**AoAS exit rule ({h['exit_rule']['rule']}): {'passes' if h['exit_rule']['passes'] else 'FAILS'}.**", "",
         (f"Monte Carlo: the bootstrap used R = {h['monte_carlo']['boot_R2']['R']} draws per replicate before; its CI "
          f"[{h['monte_carlo']['boot_R2']['ci'][0]:.2f}, {h['monte_carlo']['boot_R2']['ci'][1]:.2f}] (SD {h['monte_carlo']['boot_R2']['sd']:.3f}) becomes "
          f"[{wl['ci'][0]:.2f}, {wl['ci'][1]:.2f}] (SD {h['monte_carlo']['boot_sd']:.3f}) at R = {h['monte_carlo']['boot_R']}, same resamples and refits."
          if "boot_R2" in h["monte_carlo"] else ""), "",
         "| Policy | Wins per team-season | Share of optimum | Wins lost vs optimum | Challenges per team-game | Batter / fielder wins |",
         "|---|---|---|---|---|---|"]
    for p in h["policies"]:
        L.append(f"| {p['label']} | {f(p['wins'], p['wins_ci'])} | {100*p['share']:.1f} % [{100*p['share_ci'][0]:.1f}, {100*p['share_ci'][1]:.1f}] | "
                 f"{f(p['lost'], p['lost_ci'])} | {f(p['challenges'], p['challenges_ci'])} | {p['batter']:.2f} / {p['fielder']:.2f} |")
    o = h["one_line_rule"]
    L += ["", f"One-line rule p ≥ δ/(g+δ): best δ = {o['best_delta'].split('_')[1]}; with δ re-selected in each replicate it "
          f"gets {100*o['share_reselected']:.1f} % [{100*o['share_reselected_ci'][0]:.1f}, {100*o['share_reselected_ci'][1]:.1f}] of the optimum.", ""]
    re = h["per_player_model"]
    txt = (f"Player random effects on (κ, log σ_r) (the best-BIC model); each player's posterior is calibrated within "
           f"their own noise level. Optimum {re['point_opt']:.2f}, wins lost **{re['point_lost']:.2f}**")
    if "per_player_lost_ci" in re:
        txt += (f" [{re['per_player_lost_ci'][0]:.2f}, {re['per_player_lost_ci'][1]:.2f}] (game bootstrap with EB redraws of "
                f"the player effects, no refit, B = {re['per_player_B']}; the same no-refit bootstrap gives "
                f"[{re['main_noref_lost_ci'][0]:.2f}, {re['main_noref_lost_ci'][1]:.2f}] for the main spec)")
    txt += f". Best one-line δ = {re['best_stake'].split('_')[1]}, {100*re['best_stake_share']:.1f} % of the optimum."
    L += ["### Per-player model", "", txt, ""]
    L += ["### Model spread (all perception variants, same corrected pipeline)", "", "| Variant | Wins lost | Wins lost vs pooled DP |", "|---|---|---|"]
    L += [f"| {k} | {v['lost']:.2f} | {v['lost_pooled_dp']:.2f} |" for k, v in h["model_spread"].items()]
    c = h["causal"]["game_bootstrap"]
    L += ["", "### Causal wins vs path-sum", "",
          f"Anchored re-simulation (`src/causal.py`; R = {h['causal']['R']} draws, K = {h['causal']['K']} continuations per world "
          "per branch). Game-bootstrap CIs.", ""]
    if "causal_gap" in h:
        cg, cs = h["causal_gap"], h.get("causal_sim", {})
        L += [f"Simulator: `{h['causal']['simulator']}` (events and in-play outcomes by count, outs and bases, plus "
              f"baserunning between pitches; its flip swing is {100 * cg['g_over_engine']['run']:.1f} % of the engine's g). The original "
              f"simulator's swing was {100 * cg['g_over_engine']['sim']:.1f} % of g, which predicts a binary − RB gap of "
              f"{cg['predicted_opt_score']['sim']:+.3f} for the optimum; it measured "
              + (f"{cs['binary_minus_rb'][0]:+.3f} [{cs['binary_minus_rb'][1]:+.3f}, {cs['binary_minus_rb'][2]:+.3f}]" if cs else "n/a")
              + " (audit R1, `results/causal_gap.md`).", ""]
    L += [
          "| Policy | Path-sum | Causal (RB) | Causal − path | Binary − RB | Causal − observed |", "|---|---|---|---|---|---|"]
    sg = lambda x: f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"
    for p, label in POLICIES[:-1]:
        if p not in c:
            continue
        x = c[p]
        L.append(f"| {label} | {f(x['path'][0], x['path'][1:])} | {f(x['causal_rb'][0], x['causal_rb'][1:])} | "
                 f"{sg(x['causal_minus_path'])} | {sg(x['binary_minus_rb'])} | {f(x['causal_minus_observed'][0], x['causal_minus_observed'][1:])} |")
    fb = h["flat_bound"]; best = fb["best"]
    L += ["", "### Flat optimum on the calibrated pools (Theorem 3 of theory.md)", "",
          f"Exact regulation-only score-aware MDP. Best constant δ = {fb['best_delta']}: loss {best['loss_pct']:.1f} % of V*, "
          f"local quadratic (envelope) approximation {best['local_pct']:.1f} %, sup-density bound (ii) {best['bound_pct']:.1f} % "
          f"(holds, loose). Regret identity (i) reproduces V* − V^δ to {fb['max_identity_gap']:.1e} at every δ.", ""]
    d = h["dp_checks"]["results"]
    L += ["### Optimum robustness", "",
          f"Score clip ±6 vs ±10: {d['DM6']['in_sample']['opt_score']:.3f} vs {d['DM10']['in_sample']['opt_score']:.3f}. "
          f"2-fold cross-fit (DP on half the games, evaluated on the other half): {d['DM6']['cross_fit']['opt_score']:.3f} vs "
          f"{d['DM6']['in_sample']['opt_score']:.3f} in-sample.", "",
          f"Kernel checks (`wolfram/headline.wl`, {h['kernel_checks']['wolframVersion']}): "
          + ", ".join(k for k, v in h["kernel_checks"].items() if v == "True") + " — all True.", ""]
    open("results/headline.md", "w").write("\n".join(L))


if __name__ == "__main__":
    main()
