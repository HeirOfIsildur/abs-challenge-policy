"""HONZA-132: consolidated headline -> results/headline.json, results/headline.md (the manuscript reads here).

Primary estimate: the cross-fitted accounting gap (results/crossfit.json: stakes for decisions and for scoring from WP
engines fitted on disjoint folds of games, S random splits) between the pooled-model benchmark under the primary
intervention ("bench": change only the decision rule) and the observed challenges, main model; co-primary: the same
for the BIC-best model, player effects + lapse (11 x 11 nodes, joint posterior draws). Intervals: joint bootstrap over
games (results/joint_boot.json: Statcast games resampled with the called pitches, stakes cross-fitted, perception
refitted, every policy re-simulated), percentile / basic / normal; for the co-primary model a parametric bootstrap of
the hyperparameters on game resamples (results/joint_boot_re_lapse_gh11.json).
Also collected: the policy table, the intervention grid (results/intervention.json), the stakes validation
(results/stakes_validation.json), coverage (results/coverage_stream.json), the mechanism (results/mechanism_boot.json),
the causal re-simulation (results/causal.json), the full-sample (not cross-fitted) point (data/headline_point.pkl).
usage: python src/headline_report.py"""
import json, os, pickle, sys
import numpy as np
sys.path.insert(0, "src")
from headline import MAIN, CO_PRIMARY

POLICIES = [("bench", "Benchmark (pooled-model DP, primary intervention)"),
            ("clairvoyant", "Clairvoyant comparator (knows its future (p, g); not a bound)"),
            ("runs_dp", "Namuduri runs rule (rebuilt)"), ("opt", "DP without the score"),
            ("stake_0.008", "One-line rule, delta = 0.008"), ("stake_0.01", "One-line rule, delta = 0.01"),
            ("card_pub", "Arora card, published thresholds"), ("card_refit", "Arora card, median DP thresholds"),
            ("obs_rule", "Fitted behavioural rule (simulated)"), ("savant", "Savant 0.2-run rule"),
            ("observed", "Observed (realised)")]
J = lambda f: json.load(open(f"results/{f}")) if os.path.exists(f"results/{f}") else None
pct = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def boot_rows(name):
    b = J(name)
    return [r for r in b["rows"] if r["ok"]] if b else []


def main():
    cf = J("crossfit.json"); jb = J("joint_boot.json"); rows = boot_rows("joint_boot.json")
    P = cf[MAIN]; W = P["wins"]; N = P["n"]
    point = P["lost"]
    out = dict(spec=f"{MAIN}; pooled-model benchmark under the primary intervention (change only the decision rule)",
               units="wins per team-season (sum of WP increments; 161.7 games per team); challenges per team-game",
               estimand=("league-average accounting gap per team-season between the pooled-model benchmark and the "
                         "teams' behaviour, a season of games like 2026, each team deviating alone"),
               point=dict(lost=point, S=P["S"], R=P["R"], K=P["K"], sd_over_splits=P["lost_sd_over_splits"],
                          se_over_splits=P["lost_se"], source="results/crossfit.json"))
    if jb:
        out["interval"] = dict(B=jb["B"], B_failed=jb["B_failed"], R=jb["R"], **jb["lost"])
        out["exit_rule"] = dict(rule="switch to TAS if the 95 % CI of wins lost reaches 0.25 or is wider than 1.5",
                                passes=bool(jb["lost"]["percentile"][0] > 0.25 and jb["lost"]["percentile"][1] - jb["lost"]["percentile"][0] <= 1.5))
    # roles
    roles = {}
    for sd in ("batter", "fielder"):
        x = np.array([r[sd]["bench"] - r[sd]["observed"] for r in rows]) if rows and sd in rows[0] else None
        roles[sd] = dict(lost=P[sd]["bench"] - P[sd]["observed"], ci=pct(x) if x is not None else None)
    out["by_role"] = roles
    # policy table: cross-fitted point, joint-bootstrap percentile intervals
    tab = []
    for p, label in POLICIES:
        if p not in W:
            continue
        wb = np.array([r["wins"][p] for r in rows]) if rows else None
        sh = np.array([r["wins"][p] / r["wins"]["bench"] for r in rows]) if rows else None
        nb = np.array([r["n"][p] for r in rows]) if rows else None
        tab.append(dict(policy=p, label=label, wins=W[p], wins_ci=pct(wb) if rows else None, share=W[p] / W["bench"],
                        share_ci=pct(sh) if rows else None, n=N[p], n_ci=pct(nb) if rows else None,
                        batter=P["batter"][p], fielder=P["fielder"][p]))
    sav = P["rows"][0].get("best_savant_c")
    cs = max([k for k in W if k.startswith("savant_c_")], key=lambda k: W[k])
    out["savant_tuned"] = dict(policy=cs, c=float(cs.split("_")[-1]), wins=W[cs], share=W[cs] / W["bench"], n=N[cs],
                               share_ci=pct([r["wins"][cs] / r["wins"]["bench"] for r in rows]) if rows else None)
    st = max([k for k in W if k.startswith("stake_")], key=lambda k: W[k])
    out["one_line_rule"] = dict(delta=0.008, share=W["stake_0.008"] / W["bench"],
                                gap_closed=(W["stake_0.008"] - W["observed"]) / (W["bench"] - W["observed"]),
                                gap_closed_ci=jb["gap_closed_stake_0008"]["ci"] if jb else None,
                                best=st, best_share=W[st] / W["bench"],
                                best_by_split={k: int(v) for k, v in zip(*np.unique([r["best_stake"] for r in P["rows"]], return_counts=True))})
    if jb:
        out["freq_matched"] = jb["freq_matched"]
    out["policies"] = tab
    # co-primary
    co = cf.get(CO_PRIMARY)
    if co:
        rb = boot_rows(f"joint_boot_{CO_PRIMARY}.json"); jc = J(f"joint_boot_{CO_PRIMARY}.json")
        out["co_primary"] = dict(model=CO_PRIMARY, lost=co["lost"], S=co["S"], R=co["R"],
                                 share_rule=co["wins"]["stake_0.008"] / co["wins"]["bench"],
                                 lost_lapses_removed=co["wins"]["opt_score"] - co["wins"]["observed"],
                                 interval=dict(B=jc["B"], **jc["lost"]) if jc else None,
                                 method="parametric bootstrap of the hyperparameters on game resamples with cross-fitted stakes, "
                                        "player effects re-estimated (EB) and drawn jointly")
    # full-sample (not cross-fitted) point, for the cross-fitting effect
    if os.path.exists("data/headline_point.pkl"):
        hp = pickle.load(open("data/headline_point.pkl", "rb"))
        out["full_sample"] = {k: dict(lost=v["summary"]["bench"]["wins"] - v["summary"]["observed"]["wins"],
                                      mc_se=v["mc"]["lost_mc_se"], R=v["mc"]["R"],
                                      share_rule=v["summary"]["stake_0.008"]["wins"] / v["summary"]["bench"]["wins"])
                              for k, v in hp.items()}
    hv = J("headline_variants.json")
    if hv:                      # full-sample evaluation of every variant (supplement table), both lapse readings
        out["model_spread"] = {k: dict(lost=v["opt_score"]["wins"] - v["observed"]["wins"],
                                       lost_kept=v["bench"]["wins"] - v["observed"]["wins"] if "bench" in v else None,
                                       lost_pooled_dp=v["opt"]["wins"] - v["observed"]["wins"]) for k, v in hv["variants"].items()}
        out["variants_R"] = hv["R"]
    for key, f in (("intervention", "intervention.json"), ("stakes_validation", "stakes_validation.json"),
                   ("coverage", "coverage_stream.json"), ("mechanism", "mechanism_boot.json"), ("causal", "causal.json"),
                   ("block_boot", "block_boot.json")):
        x = J(f)
        if x is not None:
            if isinstance(x, dict):
                x = {k: v for k, v in x.items() if k not in ("rows", "seasons")}
                if key == "coverage":
                    x = {k: ({kk: vv for kk, vv in v.items() if kk != "seasons"} if isinstance(v, dict) else v)
                         for k, v in x.get("scenarios", {}).items()}
            out[key] = x
    json.dump(out, open("results/headline.json", "w"), indent=1, default=float)
    write_md(out)


def write_md(h):
    f = lambda x, c: f"{x:.2f} [{c[0]:.2f}, {c[1]:.2f}]" if c else f"{x:.2f}"
    iv = h.get("interval", {})
    L = ["## Headline (generated by src/headline_report.py from results/headline.json)", "",
         f"Estimand: {h['estimand']}. Main model, cross-fitted stakes (S = {h['point']['S']} splits, R = {h['point']['R']}).", "",
         f"**Wins lost: {h['point']['lost']:.2f}** per team-season; joint bootstrap (B = {iv.get('B')}, R = {iv.get('R')}): "
         f"percentile {iv.get('percentile')}, basic {iv.get('basic')}, normal {iv.get('normal')}.", ""]
    if "co_primary" in h:
        c = h["co_primary"]; ci_ = (c.get("interval") or {}).get("percentile")
        L += [f"Co-primary ({c['model']}): **{c['lost']:.2f}** {ci_ if ci_ else ''} (lapses removed: {c['lost_lapses_removed']:.2f}).", ""]
    L += ["| Policy | Wins | Share of benchmark | Challenges per team-game |", "|---|---|---|---|"]
    for p in h["policies"]:
        sc = p["share_ci"]
        L.append(f"| {p['label']} | {f(p['wins'], p['wins_ci'])} | {100*p['share']:.1f} %"
                 + (f" [{100*sc[0]:.1f}, {100*sc[1]:.1f}]" if sc else "") + f" | {p['n']:.2f} |")
    open("results/headline.md", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
