"""Every number quoted in the manuscript, read from results/ -> paper/generated/{numbers,tab_*}.tex.

The manuscript never types a result by hand: it uses the macros defined here (\\WinsLost, \\SrBatter, ...) and
\\input{}s the generated tables. Re-run after any change in results/ (``make -C paper``).
usage (from the repo root): python paper/make_numbers.py"""
import csv, json, math, os, re

R = "results"
OUT = "paper/generated"
J = lambda f: json.load(open(os.path.join(R, f)))
macros = {}


def m(name, value):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    assert name not in macros, f"duplicate macro {name}"
    macros[name] = value


# A real minus sign (text-mode hyphens read as "[-0.004, 0.000]"), and no "-0.00"
neg = lambda t: re.sub(r"(?<![\w])-(?=\d)", r"\\ensuremath{-}", re.sub(r"(?<![\w.])-(0(?:\.0+)?)(?![\d.])", r"\1", t))
f1, f2, f3 = (lambda x: neg(f"{x:.1f}")), (lambda x: neg(f"{x:.2f}")), (lambda x: neg(f"{x:.3f}"))
pct = lambda x, d=1: f"{100 * x:.{d}f}"
ci = lambda c, f=f2: neg(f"[{f(c[0])}, {f(c[1])}]")
thou = lambda n: f"{int(round(n)):,}".replace(",", "{,}")
PENDING = "\\textbf{[pending]}"
# the paper names the Mathematica version once, in short form ("15.0"); the full $Version stays in results/*.json
short_version = lambda v: ".".join(v.split()[0].split(".")[:2])


def geometry():
    """Geometry audit (results/geometry_audit.json)."""
    g = J("geometry_audit.json"); r = g["reproduction"]
    m("GeomRepro", f2(100 * r["statcast_raw"]["share"])); m("GeomMismatch", str(r["statcast_raw"]["mismatches"]))
    m("GeomReproOld", f1(100 * r["statcast_projected_old"]["share"])); m("GeomN", thou(r["statcast_raw"]["n"]))
    m("GeomReproFeed", f2(100 * r["feed_projected_mid"]["share"]))
    # The square-corner rule used in earlier versions, and the ball-radius sweep of the round-ball rule
    sq = r["statcast_raw_square_corners"]
    m("GeomReproSquare", f2(100 * sq["share"])); m("GeomMismatchSquare", str(sq["mismatches"]))
    m("GeomRelabelCalled", str(sq["relabelled_called"])); m("GeomRelabelChallenged", str(sq["relabelled_challenged"]))
    sw = r["statcast_raw_radius_sweep"]
    m("GeomRadiusSweep", ", ".join(f"{sw[k]} at {k}\\,in" for k in ("1.40", "1.43", "1.45", "1.47", "1.50")))
    m("GeomShiftOld", f2(g["feed_vs_statcast"]["dz_raw_in"]))
    m("GeomPlaneSD", f3(g["plane"]["statcast_2026"]["mid"]["dz_sd"]))


def headline():
    h = J("headline.json")
    wl = h["wins_lost"]
    m("WinsLost", f2(wl["point"])); m("WinsLostCI", ci(wl["ci"])); m("WinsLostLeague", f"{wl['league']:.0f}")
    m("WinsLostWidth", f2(wl["ci"][1] - wl["ci"][0])); m("BootB", str(h["B"]))
    for sd, nm in (("batter", "Batter"), ("fielder", "Fielder")):
        r = h["wins_lost_by_role"][sd]
        m(f"WinsLost{nm}", f2(r["lost"])); m(f"WinsLost{nm}CI", ci(r["ci"]))
    P = {p["policy"]: p for p in h["policies"]}
    names = {"opt_score": "Opt", "runs_dp": "Runs", "opt": "Pooled", "card_pub": "CardPub", "card_refit": "CardRefit",
             "obs_rule": "FitRule", "savant": "Savant", "observed": "Obs", "stake_0.008": "Rule", "stake_0.01": "RuleOld"}
    for p, nm in names.items():
        q = P[p]
        m(f"W{nm}", f2(q["wins"])); m(f"W{nm}CI", ci(q["wins_ci"]))
        m(f"Share{nm}", pct(q["share"])); m(f"Share{nm}CI", ci(q["share_ci"], lambda x: pct(x)))
        m(f"Lost{nm}", f2(q["lost"])); m(f"Lost{nm}CI", ci(q["lost_ci"]))
        m(f"N{nm}", f2(q["challenges"])); m(f"N{nm}CI", ci(q["challenges_ci"]))
    m("NMorePct", f"{100 * (P['opt_score']['challenges'] / P['observed']['challenges'] - 1):.0f}")
    # The mechanism. The one-line rule challenges no more often than teams yet closes most of the gap, so the
    # loss is mainly in targeting; the text says "no more than the teams'", which this assertion guards
    rl, ob, op = P["stake_0.008"], P["observed"], P["opt_score"]
    m("RuleGapClosedPct", f"{100 * (rl['wins'] - ob['wins']) / (op['wins'] - ob['wins']):.0f}")
    m("OptMinusRule", f2(op["wins"] - rl["wins"]))
    # The stated rule is delta = 0.008 (every table uses stake_0.008); the in-sample best of the grid at the
    # point estimate is reported next to it (0.007 after the zone-corner fix, 0.012 wins above 0.008)
    o = h["one_line_rule"]
    m("RuleDelta", o["delta"].split("_")[1]); m("RuleDeltaBest", o["best_delta"].split("_")[1])
    m("ShareRuleBest", pct(o["best_share"])); m("RuleBestMinusRule", f3(o["best_minus_rule"]))
    m("ShareRuleResel", pct(o["share_reselected"])); m("ShareRuleReselCI", ci(o["share_reselected_ci"], lambda x: pct(x)))
    m("RuleLossStream", f"{100 - 100 * P['stake_0.008']['share']:.0f}")
    pp = h["per_player_model"]
    m("PPOpt", f2(pp["point_opt"])); m("PPLost", f2(pp["point_lost"])); m("PPLostCI", ci(pp["per_player_lost_ci"]))
    m("PPB", str(pp["per_player_B"])); m("PPRuleShare", pct(pp["best_stake_share"]))
    m("MainNorefCI", ci(pp["main_noref_lost_ci"]))
    # model spread: variants the data do not reject (likelihood at least as good as the Gaussian baseline)
    pv = J("perception_variants.json")
    base = pv["normal"]["nll"]
    ok = [k for k in h["model_spread"] if k in pv and not pv[k]["spec"].get("re") and pv[k]["spec"].get("law", "normal") == "normal" and pv[k]["nll"] <= base + 1e-6]  # fixed-effects models
    ls = [h["model_spread"][k]["lost"] for k in ok]
    m("SpreadLo", f2(min(ls))); m("SpreadHi", f2(max(ls)))
    lr = [h["model_spread"][k]["lost"] for k in ("logistic", "t10", "t5", "t3")]
    m("SpreadRejLo", f2(min(lr))); m("SpreadRejHi", f2(max(lr)))
    m("SpreadEdge", f2(h["model_spread"]["sm_edge"]["lost"]))
    # The model-class spread for the abstract -- every evaluated model whose BIC is
    # at least as good as the main model's (homogeneous, lapse, player effects), main included
    MAINV = "kappa_sr_region_g"; bm = pv[MAINV]["bic"]
    lost_of = dict(h["model_spread"]); lost_of["main_lapse"] = dict(lost=J("lapse_wins.json")["lost"])
    lost_of[MAINV] = dict(lost=h["wins_lost"]["point"])
    good = [k for k in lost_of if k in pv and pv[k]["bic"] <= bm + 1e-6]
    la = [lost_of[k]["lost"] for k in good]
    m("SpreadAllLo", f2(min(la))); m("SpreadAllHi", f2(max(la))); m("SpreadAllModels", ", ".join(sorted(good)).replace("_", "\\_"))
    m("VariantR", str(J("headline_variants.json")["R"]))
    m("VariantMCSE", f3(h["monte_carlo"]["lost_sd_per_draw"] / math.sqrt(J("headline_variants.json")["R"])))
    # causal re-simulation vs path-sum
    c = h["causal"]["game_bootstrap"]
    d = c["opt_score"]["causal_minus_path"]
    m("CausalMinusPath", f3(d[0])); m("CausalMinusPathCI", ci(d[1:], f3))
    m("CausalLost", f2(c["opt_score"]["causal_minus_observed"][0])); m("CausalLostCI", ci(c["opt_score"]["causal_minus_observed"][1:]))
    m("BinaryLost", f2(c["opt_score"]["binary_minus_observed"][0]))
    m("BinaryLostCI", ci(c["opt_score"]["binary_minus_observed"][1:]))
    b = c["opt_score"]["binary_minus_rb"]
    m("BinaryMinusRB", f3(b[0])); m("BinaryMinusRBCI", ci(b[1:], f3))
    m("CausalMaxAbs", f3(max(abs(v["causal_minus_path"][0]) for v in c.values() if "causal_minus_path" in v)))
    m("CausalR", str(h["causal"]["R"])); m("CausalK", str(h["causal"]["K"]))
    m("SCausalOppEffect", f3(J("causal.json")["game_bootstrap"]["opt_score"]["causal_rb"][0] - J("causal.json")["game_bootstrap"]["opt_score"]["causal_rb_own"][0]))
    m("CausalBootN", thou(J("causal.json").get("boot_resamples", 2000)))     # Its own count, not \BootB
    # Monte Carlo error (audit R2) and the causal-gap diagnosis (audit R1, results/causal_gap.md)
    mc = h["monte_carlo"]
    m("BootR", str(mc["boot_R"])); m("PointR", str(mc["point_R"])); m("LostMCSE", f3(mc["lost_mc_se"]))
    m("BootSD", f3(mc["boot_sd"]))
    if "boot_R2" in mc:
        m("WinsLostCIRTwo", ci(mc["boot_R2"]["ci"])); m("BootSDRTwo", f3(mc["boot_R2"]["sd"])); m("BootRTwo", str(mc["boot_R2"]["R"]))
    if "causal_sim" in h:
        b = h["causal_sim"]["binary_minus_rb"]
        m("BinaryMinusRBSim", f2(b[0])); m("BinaryMinusRBSimAbs", f2(abs(b[0]))); m("BinaryMinusRBSimCI", ci(b[1:]))
        m("BinaryLostSim", f2(h["causal_sim"]["binary_minus_observed"][0]))
    if "causal_gap" in h:
        g = h["causal_gap"]
        m("SimGShare", pct(g["g_over_engine"]["sim"])); m("SimGShareRun", pct(g["g_over_engine"]["run"]))
        m("SimGShareRunGap", pct(1 - g["g_over_engine"]["run"]))
        m("GapPredSim", f3(g["predicted_opt_score"]["sim"])); m("GapPredRun", f3(g["predicted_opt_score"]["run"]))
        cgj = J("causal_gap.json")
        m("HomeFieldLostChange", f3(cgj["home_field"]["first_order_change_in_wins_lost"]))
        m("HomeFieldRelG", pct(abs(cgj["home_field"]["rel_g_opt_branches"])))
    # Theorem 3 on the calibrated pools
    fb = h["flat_bound"]["best"]
    m("FlatLoss", f1(fb["loss_pct"])); m("FlatLocal", f1(fb["local_pct"])); m("FlatBound", f1(fb["bound_pct"]))
    m("FlatIdentityGap", f"{h['flat_bound']['max_identity_gap']:.0e}".replace("e-", "\\times 10^{-") + "}")
    m("FlatBestDelta", str(h["flat_bound"]["best_delta"]))
    dc = h["dp_checks"]["results"]
    m("CrossFitIn", f2(dc["DM6"]["in_sample"]["opt_score"])); m("CrossFitOut", f2(dc["DM6"]["cross_fit"]["opt_score"]))
    m("ClipTen", f2(dc["DM10"]["in_sample"]["opt_score"])); m("DPCheckR", str(h["dp_checks"]["R"]))
    return h, P, pv


def policy_table(P):
    order = ["clairvoyant", "opt_score", "runs_dp", "opt", "stake_0.008", "stake_0.01", "card_pub", "card_refit", "observed",
             "obs_rule", "savant_tuned", "savant"]
    label = {"opt_score": "Optimum (score-aware DP)", "runs_dp": "Runs-objective rule \\citep{Namuduri2026runs}",
             "opt": "Pooled DP (no score in state)", "stake_0.008": "One-line rule, $\\delta=0.008$",
             "stake_0.01": "One-line rule, $\\delta=0.01$", "card_pub": "48-cell card, published \\citep{Arora2026use}",
             "card_refit": "48-cell card, refit on our DP", "observed": "\\textbf{Observed 2026 (realised)}",
             "obs_rule": "Fitted behavioural rule (simulated)", "savant": "Fixed 0.2-run rule \\citep{Savant2026abs}",
             "clairvoyant": "Clairvoyant ceiling (knows its game's future stakes and beliefs)",
             "savant_tuned": "Savant's form, run cost tuned"}
    L = ["\\begin{tabular}{lcccc}", "\\toprule",
         "Policy & Wins per team-season & Share of optimum (\\%) & Wins lost & Challenges per team-game \\\\", "\\midrule"]
    for p in order:
        q = P[p]
        lost = "---" if p in ("opt_score", "clairvoyant") else f"{f2(q['lost'])} {ci(q['lost_ci'])}"
        share = "100" if p == "opt_score" else f"{pct(q['share'])} {ci(q['share_ci'], lambda x: pct(x))}"
        L.append(f"{label[p]} & {f2(q['wins'])} {ci(q['wins_ci'])} & {share} & {lost} & {f2(q['challenges'])} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"{OUT}/tab_policies.tex", "w").write("\n".join(L) + "\n")


def perception(pv, h):
    MAIN = "kappa_sr_region_g"                 # BIC-best homogeneous model (src/headline.py MAIN)
    base, main = pv["normal"], pv[MAIN]
    m("SrBatter", f2(main["params"]["off_sr"])); m("SrCatcher", f2(main["params"]["def_sr"]))
    m("SrBaseBatter", f2(base["params"]["off_sr"])); m("SrBaseCatcher", f2(base["params"]["def_sr"]))
    # Sigma_r intervals from the B = 400 refits of headline_boot.json (the paper's bootstrap), not the older
    # B = 120 run of final_ci.py. Main spec theta: [logit pi | off: beta 6, log sr, kreg 3, sr region 3 | def: same], so
    # the bottom-edge log sigma_r of the batter is theta[7] and of the catcher theta[7 + 13]
    import numpy as np
    hb = J("headline_boot.json"); th = np.array([r["theta"] for r in hb["rows"] if r["ok"]])
    ns = (th.shape[1] - 1) // 2                # [logit pi | off block | def block], log sr at position 6 of a block
    assert th.shape[1] == 1 + 2 * ns == 33, th.shape
    for side, nm, j in (("off", "Batter", 7), ("def", "Catcher", 7 + ns)):
        lo, hi = np.exp(np.percentile(th[:, j], [2.5, 97.5])); pt_ = main["params"][f"{side}_sr"]
        m(f"Sr{nm}BootCI", ci([lo, hi])); m(f"Sr{nm}HalfPct", f1(100 * (hi - lo) / 2 / pt_))
    p = main["params"]
    # The main spec has a deterministic truth; the boundary offsets are those of the smooth truth model
    # (sm_region) refitted on the corrected geometry, where they collapse to ~0
    sr_ = pv["sm_region"]["params"]
    m("MuBottom", f2(-sr_["mu_bottom"])); m("MuTop", f2(sr_["mu_top"]))
    m("SmMax", f3(max(0.001, max(sr_[f"sm_{r}"] for r in ("bottom", "top", "arm", "glove")))))   # "at most": never print 0.000
    m("MuMaxAbs", f3(max(abs(sr_[f"mu_{r}"]) for r in ("bottom", "top", "arm", "glove"))))
    m("LRSmooth", f1(2 * (base["nll"] - pv["sm_region"]["nll"]))); m("BICSmoothMinusDet", thou(pv["sm_region"]["bic"] - base["bic"]))
    m("ContamPct", f2(100 * p["contam_pi"]))
    m("DLLEdge", thou(pv["sm_edge"]["nll"] - base["nll"]))
    m("DLLMain", thou(base["nll"] - main["nll"])); m("DLLRE", thou(base["nll"] - pv["re"]["nll"]))
    m("DLLREvsMain", thou(main["nll"] - pv["re"]["nll"]))
    m("BICMain", thou(main["bic"])); m("BICRE", thou(pv["re"]["bic"]))
    rej = [pv[k]["nll"] - base["nll"] for k in ("logistic", "t10", "t5", "t3")]
    m("DLLRejLo", thou(min(rej))); m("DLLRejHi", thou(max(rej)))
    # The same noise-law comparison on the main specification, and a lapse rate (the challenge curve tops out
    # below 1: P(challenge | w) = (1 - lambda) Phi(z)), with its wins lost (results/lapse_wins.json, R = 20)
    rejm = [pv[f"main_{k}"]["nll"] - main["nll"] for k in ("logistic", "t10", "t5", "t3")]
    m("DLLRejMainLo", thou(min(rejm))); m("DLLRejMainHi", thou(max(rejm)))
    la = pv["main_lapse"]; lw = J("lapse_wins.json")
    m("LapseDLL", thou(main["nll"] - la["nll"])); m("LapseDBIC", thou(main["bic"] - la["bic"])); m("LapseK", str(la["k"] - main["k"]))
    m("LapseBatter", f"{100 * la['params']['off_lapse']:.0f}"); m("LapseCatcher", f"{100 * la['params']['def_lapse']:.0f}")
    m("LapseSrBatter", f2(la["params"]["off_sr"])); m("LapseSrCatcher", f2(la["params"]["def_sr"]))
    m("LapseSrDropPct", f"{100 * (1 - la['params']['off_sr'] / main['params']['off_sr']):.0f}")
    m("LapseSrDropPctCatcher", f"{100 * (1 - la['params']['def_sr'] / main['params']['def_sr']):.0f}")
    m("LapseLost", f2(lw["lost"])); m("LapseR", str(lw["R"]))
    for side, nm in (("off", "Batter"), ("def", "Catcher")):
        c = main["calibration"][side]
        m(f"Over{nm}Pred", f1(100 * c["over_pred"])); m(f"Over{nm}Obs", f1(100 * c["over_obs"]))
    re_ = pv["re"]["params"]
    # On the data, how far the homogeneous main fit's batter sigma_r sits above the random-effects fit's
    m("HetDataBiasBatter", f1(100 * (p["off_sr"] / re_["off_sr"] - 1)))
    m("REKappaSDBatter", f2(re_["off_re"]["sd_kappa"])); m("REKappaSDCatcher", f2(re_["def_re"]["sd_kappa"]))
    m("RELogSrSDBatter", f2(re_["off_re"]["sd_logsr"])); m("RELogSrSDCatcher", f2(re_["def_re"]["sd_logsr"]))
    m("KappaTopBatter", f1(p["off_kappa_region_shift"]["top"]))
    m("SrTopCatcherPct", f"{100 * (p['def_sr_region_mult']['top'] - 1):.0f}")
    rows = [("sm_edge", "Smooth truth by edge, no mislabel"), ("sm_region", "Smooth truth by region + mislabel"),
            ("normal", "Baseline: deterministic truth + mislabel"), ("logistic", "Logistic $\\varepsilon$"),
            ("t10", "$t_{10}$ $\\varepsilon$"), ("t5", "$t_{5}$ $\\varepsilon$"), ("t3", "$t_{3}$ $\\varepsilon$"),
            ("sr_edge", "$\\sigma_r$ by edge"), ("sr_region", "$\\sigma_r$ by region"), ("sr_slope", "$\\sigma_r$ linear in $|w|$"),
            ("kappa_region", "$\\kappa$ by region"), ("kappa_sr_region", "$\\kappa$ and $\\sigma_r$ by region, $\\kappa$ linear in $\\log g$"),
            ("kappa_sr_region_slope", "Same + $\\sigma_r$ slope in $|w|$"), ("linear_lapse", "Same + lapse rate"),
            ("kappa_sr_region_g", "\\textbf{Same, $\\kappa$ piecewise linear in $\\log g$ (main)}"),
            ("main_glow", "Main + low-stake hinge in $\\kappa$"), ("main_sr_g", "Main + $\\sigma_r$ varying with $\\log g$"),
            ("main_lapse", "Main + lapse rate"), ("re", "Main + player random effects"),
            ("re_lapse", "Main + player random effects + lapse rate")]
    L = ["\\begin{tabular}{lrrrcc}", "\\toprule",
         "Model & $k$ & $\\Delta\\ell$ & BIC & $\\sigma_r$ batter / catcher (in) & Wins lost \\\\", "\\midrule"]
    for k, lab in rows:
        v = pv[k]
        dl = base["nll"] - v["nll"]
        lost = h["model_spread"].get(k, {}).get("lost") if k != "main_lapse" else J("lapse_wins.json")["lost"]
        if k not in pv:
            continue
        L.append(f"{lab} & {v['k']} & ${dl:+.0f}$ & {v['bic']:.0f} & {f2(v['params']['off_sr'])} / {f2(v['params']['def_sr'])}"
                 f" & {f2(lost) if lost is not None else '---'} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"{OUT}/tab_variants.tex", "w").write("\n".join(L) + "\n")
    vw = J("variant_wins.json")["calibration_test"]
    m("SlopeCalBatter", f2(vw["off"]["calibrated"]["slope_log_g"])); m("SlopeCalCatcher", f2(vw["def"]["calibrated"]["slope_log_g"]))
    os_ = J("opt_slope.json")["opt"]          # The optimum's own thresholds through the same regression
    m("OptSlopeBatter", f2(os_["off"]["slope_log_g"])); m("OptSlopeCatcher", f2(os_["def"]["slope_log_g"]))
    m("OptSlopeScoreBatter", f2(os_["off"]["slope_log_g_score_controls"])); m("OptSlopeScoreCatcher", f2(os_["def"]["slope_log_g_score_controls"]))
    m("SlopeFlatBatter", f2(vw["off"]["flat_prior"]["slope_log_g"])); m("SlopeFlatCatcher", f2(vw["def"]["flat_prior"]["slope_log_g"]))


def identification():
    ic = J("identification-check.json")
    m("WolframVersion", short_version(ic["wolframVersion"]))
    sv = lambda s: [float(x.split("`")[0]) for x in s.strip("{}").split(",")]
    b, c = sv(ic["I3_batter_sv_full"]), sv(ic["I3_catcher_sv_full"])
    m("IThreeBatterSV", "/".join(f"{x:.3f}" for x in b)); m("IThreeCatcherSV", "/".join(f"{x:.3f}" for x in c))
    m("IThreeRankFull", str(ic["I3_batter_rank_full"])); m("IThreeRankCh", str(ic["I3_batter_rank_challenge_only"]))
    idn = J("identification.json")
    m("RecovBiasBatter", f"{idn['correct']['bias_pct']['off_sr']:+.1f}"); m("RecovBiasCatcher", f"{idn['correct']['bias_pct']['def_sr']:+.1f}")
    m("RecovN", str(idn["correct"]["n"])); m("RecovConv", str(idn["correct"].get("converged", idn["correct"]["n"])))
    m("RecovTFiveBias", f"{idn['t5_noise']['bias_pct']['off_sr']:.0f}")
    m("RecovHetBias", f"{idn['player_heterogeneity']['bias_pct']['off_sr']:.0f}")
    if "coord_noise_0.3in" in idn:
        m("RecovCoordBias", f"{idn['coord_noise_0.3in']['bias_pct']['off_sr']:+.1f}")
    # Kernel check of the deterministic-truth identification (wolfram/identification_det.wl)
    dk = J("identification-det-check.json")
    m("DetRank", str(dk["D2_rank_batter"])); m("DetVersion", short_version(dk["wolframVersion"]))
    bias = [float(x) for x in dk["D4_rel_bias_pct_c03"].replace("{", "").replace("}", "").split(",")]
    m("DetCoordBiasCatcher", f1(bias[1])); m("DetCoordBiasBatter", f1(bias[3]))
    # sigma_r multiplier at which wins lost reaches 0.25 (linear interpolation of sr_breakeven.json).
    # Sr_breakeven.json is evaluated on the score-aware optimum (opt_score), like the headline.
    br = sorted((float(k), v["lost"]) for k, v in J("sr_breakeven.json").items())
    be = next(x0 + (x1 - x0) * (l0 - 0.25) / (l0 - l1) for (x0, l0), (x1, l1) in zip(br, br[1:]) if l0 >= 0.25 > l1)
    m("BreakEvenPct", f"{100 * (be - 1):.0f}")
    # What the break-even sigma_r costs in the profile log-likelihood (identification.json, baseline model),
    # quadratic interpolation through the three grid points nearest to it.
    def prof(side, mult):
        P = idn[f"profile_{side}"]; s0 = idn["truth"][f"{side}_sr"]
        pts = sorted(((x / s0, ll) for x, ll in P), key=lambda t: abs(t[0] - mult))[:3]
        return sum(l * math.prod((mult - x2) / (x1 - x2) for x2, _ in pts if x2 != x1) for x1, l in pts)
    llb, llc = prof("off", be), prof("def", be)
    m("BreakEvenLLBatter", f"{llb:.0f}"); m("BreakEvenLLCatcher", f"{llc:.0f}")
    m("ProfileTenBatter", f"{prof('off', 1.1):.0f}"); m("ProfileTenCatcher", f"{prof('def', 1.1):.0f}")
    o = J("oos.json")
    t = o["temporal_predict_test_with_train"]
    m("OOSCatcherExcess", f"{100 * (t['def']['ch_obs'] / t['def']['ch_pred'] - 1):.0f}")
    m("OOSTrainSrBatter", f2(o["fit_train_marjun"]["off_sr"])); m("OOSTrainSrCatcher", f2(o["fit_train_marjun"]["def_sr"]))
    m("OOSTestSrBatter", f2(o["fit_test_julsep"]["off_sr"])); m("OOSTestSrCatcher", f2(o["fit_test_julsep"]["def_sr"]))
    m("AAASrBatter", f2(o["aaa2025_refit"]["off_sr"])); m("AAASrCatcher", f2(o["aaa2025_refit"]["def_sr"]))
    m("SpringSrBatter", f2(o["spring2026_refit"]["off_sr"])); m("SpringSrCatcher", f2(o["spring2026_refit"]["def_sr"]))
    # How much noisier than MLB 2026 in the same model, range over the two sides (the main
    # model, which oos.py now fits to every sample)
    b0 = J("perception_variants.json")["kappa_sr_region_g"]["params"]
    for key, nm in (("aaa2025_refit", "AAA"), ("spring2026_refit", "Spring")):
        d_ = [100 * (o[key][f"{sd}_sr"] / b0[f"{sd}_sr"] - 1) for sd in ("off", "def")]
        m(f"{nm}SrPctLo", f"{min(d_):.0f}"); m(f"{nm}SrPctHi", f"{max(d_):.0f}")
    fc = J("feed_check.json")["kappa_sr_region_g"]
    m("FeedSrBatter", f2(fc["params"]["off_sr"])); m("FeedSrCatcher", f2(fc["params"]["def_sr"]))
    L = ["\\begin{tabular}{lrrcc}", "\\toprule",
         "Fit sample $\\to$ evaluation & \\multicolumn{2}{c}{$\\sigma_r$ (in)} & \\multicolumn{2}{c}{Overturn rate, model / obs. (\\%)} \\\\",
         " & batter & catcher & batter & catcher \\\\", "\\midrule"]

    def row(lab, fit, sc):
        L.append(f"{lab} & {f2(fit['off_sr'])} & {f2(fit['def_sr'])} & {f1(100 * sc['off']['over_pred'])} / {f1(100 * sc['off']['over_obs'])}"
                 f" & {f1(100 * sc['def']['over_pred'])} / {f1(100 * sc['def']['over_obs'])} \\\\")
    row("MLB 2026 Mar--Jun $\\to$ Jul--Sep", o["fit_train_marjun"], t)
    row("Triple-A 2025 (refit)", o["aaa2025_refit"], o["aaa2025_refit_score"])
    row("Spring training 2026 (refit)", o["spring2026_refit"], o["spring2026_refit_score"])
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"{OUT}/tab_oos.tex", "w").write("\n".join(L) + "\n")


def theory():
    p = J("proofs.json")
    m("ProofsVersion", short_version(p["wolframVersion"]))
    m("RTwoHome", f3(p["T2i_calibrated_home"]["R2"])); m("RTwoAway", f3(p["T2i_calibrated_away"]["R2"]))
    m("StylLoss", f1(100 * p["T3b_rel_loss_best_delta"])); m("StylBound", f1(100 * p["T3b_quadratic_bound {abs, rel}"][1]))
    m("StylCrude", f"{100 * p['T3b_crude_bound M(D-de)^2/(8de) {abs, rel}'][1]:.0f}")
    m("CrossTwo", str(p["T4_ret1>cons2_for_all_t>="])); m("CrossThree", str(p["T4_ret1>cons3_for_all_t>="]))
    m("CrossFive", str(p["T4_ret1>cons5_for_all_t>="]))
    t4 = p["T4_T3b_law_T60"]
    m("RetTwoOverConsTwo", f"{100 * (t4['ret2'] / t4['cons2'] - 1):.0f}")
    hc = J("headline-check.json")
    m("HeadlineChecks", "all \\texttt{True}" if all(v == "True" for k, v in hc.items() if k != "wolframVersion") else "\\textbf{NOT ALL TRUE}")


def drs():
    d = J("drs/summary.json")
    F = [d[k] for k in ("Test", "ODI", "T20")]
    for x, nm in zip(F, ("Test", "ODI", "Tt")):
        m(f"DRS{nm}Share", pct(x["per100"]["stake_share"][0])); m(f"DRS{nm}Lost", f3(x["per100"]["lost_vs_rule"][0]))
        m(f"DRS{nm}LostObs", f2(x["per100"]["lost_vs_observed"][0])); m(f"DRS{nm}LostObsCI", ci(x["per100"]["lost_vs_observed"][1]))
        m(f"DRS{nm}S", f2(x["model"]["s"])); m(f"DRS{nm}Move", f"{x['thresholds_lbw_pct']['move_c1_w0_to_w8']:.0f}")
        m(f"DRS{nm}Power", f2(x["sqrt_law"]["power"])); m(f"DRS{nm}LBW", f1(100 * x["model"]["lbw_error_rate"]))
    m("DRSMatches", thou(sum(x["matches"] for x in F)))
    m("DRSB", str(d["T20"]["bootstrap_reps"]))
    L = ["\\begin{tabular}{lccc}", "\\toprule", " & Test & ODI & T20 \\\\", "\\midrule"]
    row = lambda lab, fn: L.append(lab + " & " + " & ".join(fn(x) for x in F) + " \\\\")
    row("DRS matches", lambda x: thou(x["matches"]))
    row("Batting-side out decisions", lambda x: thou(x["decisions"]))
    row("Perception noise $\\hat s$ (margin SD)", lambda x: f2(x["model"]["s"]))
    row("Optimum, match points per 100 team-matches", lambda x: f"{f2(x['per100']['opt'][0])} {ci(x['per100']['opt'][1])}")
    row("One-line rule, share of optimum (\\%)", lambda x: pct(x["per100"]["stake_share"][0]))
    row("Lost against realised play", lambda x: f"{f2(x['per100']['lost_vs_observed'][0])} {ci(x['per100']['lost_vs_observed'][1])}")
    row("Lost by teams' fitted rule (simulated)", lambda x: f"{f3(x['per100']['lost_vs_rule'][0])} {ci(x['per100']['lost_vs_rule'][1], f2)}")
    row("Threshold move, 1 review left, top order $\\to$ tail (pp)", lambda x: f"{x['thresholds_lbw_pct']['move_c1_w0_to_w8']:.0f}")
    row("Power of $\\Delta_1$ in opportunities left", lambda x: f2(x["sqrt_law"]["power"]))
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"{OUT}/tab_drs.tex", "w").write("\n".join(L) + "\n")


def data_counts():
    roles = {r["role"]: r for r in csv.DictReader(open(os.path.join(R, "roles_observed.csv")))}
    for k, nm in (("batter", "Batter"), ("catcher", "Catcher"), ("pitcher", "Pitcher")):
        m(f"NCh{nm}", thou(int(roles[k]["n"]))); m(f"Overturn{nm}", f1(100 * float(roles[k]["overturn"])))
    f = os.path.join(R, "data_counts.json")
    d = json.load(open(f)) if os.path.exists(f) else {}
    m("ExtraGamesPct", f"{100 * d['games_extra'] / d['games_called']:.0f}")
    for k, key in (("NGames", "games"), ("NGamesABS", "games_abs"), ("NChallenges", "challenges"), ("NCalled", "called_pitches"), ("NPitches", "pitches"),
                   ("NOverturned", "overturned")):
        m(k, thou(d[key]) if key in d else PENDING)
    m("NGamesNoABS", str(d["games"] - d["games_abs"]) if "games_abs" in d else PENDING)


def thresholds():
    """Bar-2 threshold gaps and the sqrt-n fit on the score-aware DP (Manuscript build), if present."""
    keys = ("GapEarlyLateOne", "GapEarlyLateTwo", "GapOneTwo", "SqrtRTwo", "LinRTwo", "DeltaSqRTwoHome",
            "DeltaSqRTwoAway", "DeltaRatio", "GapMethod")
    f = os.path.join(R, "thresholds_score.json")
    vals = {}
    if os.path.exists(f):
        from thresholds_macros import macros as tm
        vals = tm(json.load(open(f)))
    for k in keys:
        m(k, vals.get(k, PENDING))
    for k in set(vals) - set(keys):
        m(k, vals[k])


def postseason():
    """Postseason out-of-sample check (results/postseason_preview.json) -> Post* macros, tab_post.tex and
    the supplement's tab_s_post*.tex."""
    d = J("postseason_preview.json")
    s = d["sample"]; po, rs = s["Postseason"], s["Regular season"]
    m("PostGames", str(po["games"])); m("PostCalled", thou(po["called"])); m("PostCh", str(po["challenges"]))
    m("PostChBatter", str(po["ch_batter"])); m("PostChCatcher", str(po["ch_catcher"] + po["ch_pitcher"]))
    m("PostPerTeamGame", f2(po["per_team_game"])); m("RSPerTeamGame", f2(rs["challenges"] / (2 * rs["games"])))   # 2.1755: the stored 2.175 rounds down
    for r, nm in (("Wild Card", "WC"), ("Division Series", "DS"), ("LCS", "LCS")):
        m(f"Post{nm}Games", str(s[r]["games"]))
    pt = d["perception_transfer"]
    tr, t2 = pt["transfer"], pt["transfer_second_half_fit"]
    reg = pt["regular_season_insample"]
    sr = pt["refit_sr_only"]
    for side, nm in (("off", "Batter"), ("def", "Catcher")):
        t = tr[side]
        m(f"PostChObs{nm}", str(t["ch_obs"])); m(f"PostChPred{nm}", f1(t["ch_pred"])); m(f"PostChZ{nm}", f"{t['ch_z']:+.1f}")
        m(f"PostSkill{nm}", f2(t["challenge_skill"])); m(f"PostSkillRS{nm}", f2(reg[side]["challenge_skill"]))
        m(f"PostChPredHalf{nm}", f1(t2[side]["ch_pred"])); m(f"PostChZHalf{nm}", f"{t2[side]['ch_z']:+.1f}")
        m(f"PostSr{nm}", f2(sr[side]["sr_playoff"])); m(f"PostSr{nm}CI", ci(sr[side]["sr_playoff_ci95"]))
        m(f"PostSrRS{nm}", f2(sr[side]["sr_regular"]))
        si = pt["refit_sr_and_intercept"][side]
        m(f"PostShift{nm}", f2(si["intercept_shift"])); m(f"PostShift{nm}CI", ci(si["intercept_shift_ci95"]))
        m(f"PostSrShift{nm}", f2(si["sr_playoff"])); m(f"PostSrShift{nm}CI", ci(si["sr_playoff_ci95"]))
        st = d["stakes_response"][side]
        m(f"PostSlope{nm}", f2(st["implied_slope_playoff_refit"])); m(f"PostSlope{nm}CI", ci(st["implied_slope_playoff_refit_ci95"]))
        m(f"PostSlopeRS{nm}", f2(st["implied_slope_regular"]))
    w = d["wins"]; ow = w["po"]["opt_minus_observed"]
    f4 = lambda x: f"{x:.4f}"
    m("PostLostWP", f4(ow["wp_per_team_game"])); m("PostLostWPCI", ci(ow["ci95"], f4))
    m("RSLostWP", f4(w["rs"]["opt_minus_observed"]["wp_per_team_game"]))
    m("PostShareObs", pct(ow["share_of_opt_captured"], 0)); m("PostShareObsCI", ci(ow["share_ci95"], lambda x: pct(x, 0)))
    m("PostB", str(ow["bootstrap_B"])); m("PostR", str(w["R"]))
    se = d["stakes_response"]["series"]
    m("PostNLLGame", f2(se["nll_game_wp"])); m("PostNLLSeries", f2(se["nll_series_wp"]))
    bl = se["by_leverage"]["L>=0.5"]
    m("PostHighLGames", str(bl["games"])); m("PostHighLZBatter", f"{bl['off']['ch_z']:+.1f}"); m("PostHighLZCatcher", f"{bl['def']['ch_z']:+.1f}")
    m("PostSeriesLostWP", f4(ow["series_wp_per_team_game"])); m("PostSeriesLostWPCI", ci(ow["series_wp_ci95"], f4))
    # main-text table: transfer with regular-season parameters held fixed
    sx = d["stakes_response"]
    L = ["\\begin{tabular}{lcc}", "\\toprule", " & Batters & Catchers and pitchers \\\\", "\\midrule"]
    row = lambda lab, fn: L.append(lab + " & " + " & ".join(fn(x) for x in ("off", "def")) + " \\\\")
    row("Challenges, observed / predicted ($z$)", lambda x: f"{tr[x]['ch_obs']} / {f1(tr[x]['ch_pred'])} (${tr[x]['ch_z']:+.1f}$)")
    row("\\quad predicted from a post-All-Star-break fit ($z$)", lambda x: f"{f1(t2[x]['ch_pred'])} (${t2[x]['ch_z']:+.1f}$)")
    row("Challenge log-score skill, postseason / regular season", lambda x: f"{f2(tr[x]['challenge_skill'])} / {f2(reg[x]['challenge_skill'])}")
    row("$\\sigma_r$ (in), postseason refit / regular season",
        lambda x: f"{f2(sr[x]['sr_playoff'])} {ci(sr[x]['sr_playoff_ci95'])} / {f2(sr[x]['sr_regular'])}")
    row("Threshold slope on $\\log g$, postseason refit / regular season",
        lambda x: f"{f2(sx[x]['implied_slope_playoff_refit'])} {ci(sx[x]['implied_slope_playoff_refit_ci95'])} / {f2(sx[x]['implied_slope_regular'])}")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"{OUT}/tab_post.tex", "w").write("\n".join(L) + "\n")
    # supplement: sample by round
    L = ["\\begin{tabular}{lrrrrrrr}", "\\toprule",
         " & Games & Called pitches & Challenges & Per team-game & Batter & Catcher, pitcher & Overturned (\\%) \\\\", "\\midrule"]
    for r in ("Wild Card", "Division Series", "LCS", "Postseason", "Regular season"):
        x = s[r]
        L.append(f"{r} & {thou(x['games'])} & {thou(x['called'])} & {thou(x['challenges'])} & {f2(x['challenges'] / (2 * x['games']))} & "
                 f"{thou(x['ch_batter'])} & {thou(x['ch_catcher'] + x['ch_pitcher'])} & {pct(x['overturn'])} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"{OUT}/tab_s_postsample.tex", "w").write("\n".join(L) + "\n")
    # supplement: learning over the season
    lr = d["learning"]
    L = ["\\begin{tabular}{lrcccc}", "\\toprule",
         "Period & Games & \\multicolumn{2}{c}{Challenges per 100 eligible pitches} & \\multicolumn{2}{c}{Observed / predicted ($z$)} \\\\",
         " & & batter & catcher & batter & catcher \\\\", "\\midrule"]
    for k, lab in (("first_half", "Before the All-Star break"), ("second_half", "After the All-Star break"), ("postseason", "Postseason")):
        x = lr[k]; mo = x["model"]
        L.append(f"{lab} & {thou(x['games'])} & {f2(mo['off']['ch_per_100_eligible'])} & {f2(mo['def']['ch_per_100_eligible'])} & "
                 f"{thou(mo['off']['ch_obs'])} / {f1(mo['off']['ch_pred'])} (${mo['off']['ch_z']:+.1f}$) & "
                 f"{thou(mo['def']['ch_obs'])} / {f1(mo['def']['ch_pred'])} (${mo['def']['ch_z']:+.1f}$) \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"{OUT}/tab_s_postlearn.tex", "w").write("\n".join(L) + "\n")
    # supplement: value in game WP per team-game (challenges per team-game in parentheses)
    lab = {"opt_score": "Optimum (score-aware DP)", "stake_0.008": "One-line rule, $\\delta=0.008$",
           "obs_rule": "Fitted behavioural rule (simulated)", "observed": "Observed (realised)"}
    L = ["\\begin{tabular}{lcc}", "\\toprule", "Policy & Regular season & Postseason \\\\", "\\midrule"]
    for p, nm in lab.items():
        a, b = dict(w["rs"][p]), w["po"][p]
        if p == "observed":                  # exact count ratio: the stored 2.175 (3 dp) would print as 2.17
            a["challenges_per_team_game"] = rs["challenges"] / (2 * rs["games"])
        L.append(f"{nm} & {f4(a['wp_per_team_game'])} ({f2(a['challenges_per_team_game'])}) & "
                 f"{f4(b['wp_per_team_game'])} ({f2(b['challenges_per_team_game'])}) \\\\")
    L.append(f"Optimum $-$ observed & {f4(w['rs']['opt_minus_observed']['wp_per_team_game'])} & "
             f"{f4(ow['wp_per_team_game'])} {ci(ow['ci95'], f4)} \\\\")
    L.append(f"Share of the optimum captured (\\%) & {pct(w['rs']['opt_minus_observed']['share_of_opt_captured'], 0)} & "
             f"{pct(ow['share_of_opt_captured'], 0)} {ci(ow['share_ci95'], lambda x: pct(x, 0))} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"{OUT}/tab_s_postvalue.tex", "w").write("\n".join(L) + "\n")


def causal_gap():
    """Simulator-versus-engine decomposition (results/causal_gap.json) -> SGap* macros, tab_s_gap*.tex."""
    g = J("causal_gap.json"); o = g["by_policy"]["opt_score"]; mc = g["mc_check"]
    m("SGapBranches", f"{o['branches_per_season']:.0f}"); m("SGapMeanG", f"{o['mean_g_branch_engine']:.4f}")
    m("SGapMoveRate", f1(100 * g["p_move_between_pitches"]))
    m("SGapHomeWP", f3(g["home_field"]["home_wp_first_pitch"]))
    for k, nm in (("sim", "Old"), ("run", "New")):
        m(f"SGapMC{nm}", f"{mc[k]['mc_minus_chain']:+.4f}"); m(f"SGapMC{nm}SE", f"{mc[k]['mc_minus_chain_se']:.4f}")
        m(f"SGapLevel{nm}", f"{mc[k]['home_win_mc_minus_engine']:+.4f}")
    meas, meas2 = o["measured_binary_minus_rb"], o["measured_binary_minus_rb_run"]
    m("SGapMeasOld", f3(meas[0])); m("SGapMeasOldCI", ci(meas[1:], f3))
    m("SGapMeasNew", f3(meas2[0])); m("SGapMeasNewCI", ci(meas2[1:], f3))
    pols = list(g["by_policy"])
    m("SGapAvgOld", f3(sum(g["by_policy"][k]["measured_binary_minus_rb"][0] for k in pols) / len(pols)))
    m("SGapAvgNew", f3(sum(g["by_policy"][k]["measured_binary_minus_rb_run"][0] for k in pols) / len(pols)))
    m("SGapNPolicies", str(len(pols)))
    pr = o["predicted"]
    share = {k: pct(mc[k]["chain_over_engine"]) for k in ("sim", "run")}
    rows = [("Events by count; in-play outcome by the plate appearance's starting outs and bases", "sim"),
            ("+ in-play outcome by count, outs and bases", "kc"),
            ("+ pitch events by outs and bases", "kcob"),
            ("+ baserunning between pitches (simulator used)", "run")]
    L = ["\\begin{tabular}{p{7.5cm}rr}", "\\toprule", "Simulator & $g$ / engine $g$ (\\%) & Predicted win count $-$ RB \\\\", "\\midrule"]
    for lab, k in rows:
        L.append(f"{lab} & {share.get(k, '---')} & ${pr[k]:+.3f}$ \\\\")
    L.append("WP engine & 100 & $0$ \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"{OUT}/tab_s_gap.tex", "w").write("\n".join(L) + "\n")
    L = ["\\begin{tabular}{lrrrrr}", "\\toprule", "Count & $n$ & \\multicolumn{4}{c}{$g$ / engine $g$ $-1$ (\\%)} \\\\",
         " & & first simulator & + contact by count & + events by outs, bases & + baserunning \\\\", "\\midrule"]
    for r in g["opt_score_by_count"]:
        L.append(f"{r['count']} & {thou(r['n'])} & " + " & ".join(f"${100 * r['rel_' + k]:+.1f}$" for k in ("sim", "kc", "kcob", "run")) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"{OUT}/tab_s_gapcount.tex", "w").write("\n".join(L) + "\n")


def review_fixes():
    """Bootstrap interval diagnostics, the nonparametric check of the
    challenge curve (results/isotonic_check.json) and the behavioural-rule gap (results/behaviour_gap.json)."""
    import random
    h = J("headline.json")
    rows = [r for r in J("headline_boot.json")["rows"] if r.get("ok")]
    lost = [r["summary"]["opt_score"]["wins"] - r["summary"]["observed"]["wins"] for r in rows]
    n = len(lost); srt = sorted(lost)
    q = lambda xs, a: sorted(xs)[min(len(xs) - 1, max(0, int(round(a * (len(xs) - 1)))))]
    rng = random.Random(106); ends = []
    for _ in range(2000):                     # Monte Carlo error of the percentile endpoints: resample the B replicates
        xs = [lost[rng.randrange(n)] for _ in range(n)]
        ends.append((q(xs, 0.025), q(xs, 0.975)))
    sd = lambda v: (sum((x - sum(v) / len(v)) ** 2 for x in v) / (len(v) - 1)) ** 0.5
    m("CIEndSE", f3(max(sd([e[0] for e in ends]), sd([e[1] for e in ends]))))
    pt, lo, hi = h["wins_lost"]["point"], q(srt, 0.025), q(srt, 0.975)
    s_ = sd(lost)
    basic, normal = (2 * pt - hi, 2 * pt - lo), (pt - 1.96 * s_, pt + 1.96 * s_)
    m("CIAltMaxDiff", f2(max(abs(round(a, 2) - round(b, 2)) for a, b in zip(basic + normal, (lo, hi, lo, hi)))))
    m("WinsLostCINormal", ci(normal))
    iso = J("isotonic_check.json")["panels"]
    m("IsoMaxGap", f"{100 * max(v['max_abs_gap'] for v in iso.values()):.0f}")
    m("IsoBandLo", f"{100 * min(v['share_grid_model_in_band'] for v in iso.values()):.0f}")
    m("IsoBandHi", f"{100 * max(v['share_grid_model_in_band'] for v in iso.values()):.0f}")
    m("IsoB", str(J("isotonic_check.json")["B"]))
    bg = J("behaviour_gap.json"); fm, fx = bg["fits"]["kappa_sr_region"], bg["fits"]["kappa_sr_region_g"]
    m("KGDLL", f"{fm['nll'] - fx['nll']:.0f}"); m("KGK", str(fx["k"] - fm["k"])); m("KGDBIC", f"{fm['bic'] - fx['bic']:.0f}")
    m("KGSrBatter", f2(fx["off_sr"])); m("KGSrCatcher", f2(fx["def_sr"]))
    ev = bg["evaluation"]; e0, e1 = ev["kappa_sr_region"], ev["kappa_sr_region_g"]
    m("KGOpt", f2(e1["opt_score"]["wins"])); m("KGOptMain", f2(e0["opt_score"]["wins"]))
    m("KGFitRule", f2(e1["obs_rule"]["wins"])); m("KGFitRuleMain", f2(e0["obs_rule"]["wins"])); m("KGR", str(bg["R"]))
    m("KGLost", f2(e1["opt_score"]["wins"] - e1["observed"]["wins"]))
    m("KGFitRuleN", f2(e1["obs_rule"]["n"]))
    m("KGFitRuleScaled", f2(e1["obs_rule"]["wins"] * e1["observed"]["n"] / e1["obs_rule"]["n"]))
    m("KGRemainder", f2(e1["observed"]["wins"] - e1["obs_rule"]["wins"] * e1["observed"]["n"] / e1["obs_rule"]["n"]))
    Q = bg["quintiles"]
    top = [r for r in Q if r["q"] == 4]
    m("BGTopChExcess", thou(sum(r["ch_obs"] - r["ch_pred"] for r in top)))
    m("BGTopValueExcess", f2(sum(r["v_obs"] - r["v_pred"] for r in top)))
    m("BGValueExpected", f2(bg["expected_total_value"])); m("BGValueObserved", f2(bg["observed_total_value"]))
    L = ["\\begin{tabular}{llrrrrrr}", "\\toprule",
         "Side & Stake quintile & Mean $g$ & \\multicolumn{2}{c}{Challenges} & \\multicolumn{2}{c}{Overturned} & Value obs.\\ $-$ exp. \\\\",
         " & & & obs. & exp. & obs. & exp. & (wins/team-season) \\\\", "\\midrule"]
    for side, lab in (("off", "Batters"), ("def", "Fielding side")):
        for r in sorted((r for r in Q if r["side"] == side), key=lambda r: r["q"]):
            L.append(f"{lab if r['q'] == 0 else ''} & {r['q'] + 1} & {r['g_mean']:.4f} & {thou(r['ch_obs'])} & {thou(r['ch_pred'])} & "
                     f"{thou(r['ov_obs'])} & {thou(r['ov_pred'])} & ${r['v_obs'] - r['v_pred']:+.3f}$ \\\\")
        L.append("\\addlinespace" if side == "off" else "\\bottomrule")
    L.append("\\end{tabular}")
    open(f"{OUT}/tab_s_behaviourgap.tex", "w").write("\n".join(L) + "\n")


def review_extras():
    """Sample flow, label agreement on all pitches, the clairvoyant ceiling, the tuned
    Savant constant, the cross-fitted constant, the joint and block bootstraps, the lapse re-test, the mechanism
    diagnostics and the first-divergence causal check. A missing result prints as pending."""
    def get(f):
        try:
            return J(f)
        except FileNotFoundError:
            return None
    sf = get("sample_flow.json")
    if sf:
        m("NMisjoin", thou(sf["misjoined_located"])); m("NMisjoinCh", str(sf["misjoined_challenges"]))
        m("NPosPitch", thou(sf["position_player_pitches"])); m("NPosPlayers", str(sf["position_players"]))
        m("NNonMono", str(sf["nonmonotone_cells"])); m("NNonMonoCh", str(sf["nonmonotone_challenged"]))
        m("NNoLoc", str(sf["no_location"])); m("NLocCh", thou(sf["located_challenges"]))
        m("NEvalOpp", thou(sf["evaluation_opportunities"])); m("NOutsideWindow", thou(sf["outside_fit_window"]))
        m("NLikBatter", thou(sf["likelihood"]["off"]["n"])); m("NLikCatcher", thou(sf["likelihood"]["def"]["n"]))
        m("NHalfRestart", str(sf["half_innings_restarted"]))
    if sf:
        rows = [("Called pitches in the feeds", sf["called_feed"], sf["challenges_feed"]),
                ("without a Statcast location", -sf["no_location"], None),
                ("mis-joined to another Statcast pitch", -sf["misjoined_located"], -sf["misjoined_challenges"]),
                ("Located called pitches", sf["located"], sf["located_challenges"]),
                ("thrown by position players (not opportunities)", -sf["position_player_pitches"], -sf["position_player_challenges"]),
                ("Policies' opportunity set", sf["evaluation_opportunities"], None),
                ("Likelihood sample, batters (challenge held, $|w|<8$\\,in)", sf["likelihood"]["off"]["n"], sf["likelihood"]["off"]["challenges"]),
                ("Likelihood sample, fielding side", sf["likelihood"]["def"]["n"], sf["likelihood"]["def"]["challenges"])]
        L = ["\\begin{tabular}{lrr}", "\\toprule", "Step & Called pitches & Challenges \\\\", "\\midrule"]
        for lab, a_, b_ in rows:
            fmt = lambda x: "" if x is None else (f"$-${thou(-x)}" if x < 0 else thou(x))
            L.append(f"{lab} & {fmt(a_)} & {fmt(b_)} \\\\")
        L += ["\\bottomrule", "\\end{tabular}"]
        open(f"{OUT}/tab_s_sampleflow.tex", "w").write("\n".join(L) + "\n")
    la = get("label_agreement.json")
    if la:
        m("LabelDifferPct", f2(100 * la["share_differ"])); m("LabelDifferN", thou(la["n_differ"]))
        m("LabelDifferUnchPct", f2(100 * la["unchallenged"]["share_differ"])); m("LabelDifferLikPct", f2(100 * la["likelihood_sample"]["share_differ"]))
        m("LabelDifferChPct", f2(100 * la["challenged"]["share_differ"])); m("LabelPitches", thou(la["pitches"]))
    h = J("headline.json"); P = {p["policy"]: p for p in h["policies"]}
    if "clairvoyant" in P:
        m("WClair", f2(P["clairvoyant"]["wins"])); m("WClairCI", ci(P["clairvoyant"]["wins_ci"]))
        m("ClairGap", f2(P["clairvoyant"]["wins"] - P["opt_score"]["wins"])); m("NClair", f2(P["clairvoyant"]["challenges"]))
    sv = h.get("savant_tuned")
    if sv:
        m("SavantTunedC", f"{sv['c']:g}"); m("ShareSavantTuned", pct(sv["share"])); m("ShareSavantTunedCI", ci(sv["share_ci"], lambda x: pct(x)))
    o = get("oos.json")
    if o and "frozen_share" in o.get("wins_lost_test_train_params", {}):
        t = o["wins_lost_test_train_params"]
        m("FrozenDelta", f"{t['frozen_delta']:g}"); m("FrozenShare", pct(t["frozen_share"])); m("FrozenGapClosed", f"{100 * t['frozen_gap_closed']:.0f}")
        m("OOSTestLost", f2(t["opt_minus_observed"])); m("OOSTestLostOwn", f2(o["wins_lost_test_test_params"]["opt_minus_observed"]))
    for mode, nm in (("joint", "Joint"), ("block", "Block")):
        b = get(f"{mode}_boot.json")
        if b:
            m(f"{nm}BootB", str(b["B"])); m(f"{nm}BootSD", f3(b["lost"]["sd"])); m(f"{nm}BootCI", ci(b["lost"]["ci"]))
            m(f"{nm}BootRatio", f2(b["sd_ratio_to_game_refit"])); m(f"{nm}BootR", str(b["R"]))
            m(f"{nm}BootShareRuleCI", ci(b["shares"]["stake_0.008"]["ci"], lambda x: pct(x)))
    pv = J("perception_variants.json")
    if "linear_lapse" in pv:
        a, l = pv["kappa_sr_region"], pv["linear_lapse"]
        m("LinLapseDBIC", thou(a["bic"] - l["bic"])); m("LinLapseDLL", thou(a["nll"] - l["nll"]))
    if "re_lapse" in pv:
        m("RELapseDBIC", thou(pv["re"]["bic"] - pv["re_lapse"]["bic"]))
    if "main_glow" in pv:
        mn = pv["kappa_sr_region_g"]
        m("GlowDBIC", thou(mn["bic"] - pv["main_glow"]["bic"])); m("SrGDBIC", thou(mn["bic"] - pv["main_sr_g"]["bic"]))
        m("GlowDBICAbs", thou(abs(mn["bic"] - pv["main_glow"]["bic"]))); m("SrGDBICAbs", thou(abs(mn["bic"] - pv["main_sr_g"]["bic"])))
        m("KLinDBIC", thou(pv["kappa_sr_region"]["bic"] - mn["bic"]))
    lw = get("lapse_wins.json")
    if lw and "opt_score_lapsed" in lw:
        m("LapseLostKept", f2(lw["opt_score_lapsed"]["wins"] - lw["observed"]["wins"])); m("LapseFitRule", f2(lw["obs_rule"]["wins"]))
    mc = get("mechanism.json")
    if mc:
        sc = mc["selected_calibration"]
        m("SelCalOpt", f1(100 * (sc["opt_score"]["ratio"] - 1))); m("SelCalRule", f1(100 * (sc["stake_0.008"]["ratio"] - 1)))
        Q = {(r["side"], r["q"]): r for r in mc["thresholds_by_stake_quintile"]}
        for side, nm in (("off", "Batter"), ("def", "Catcher")):
            m(f"PstarTeamLow{nm}", f2(Q[(side, 0)]["pstar_team"])); m(f"PstarTeamHigh{nm}", f2(Q[(side, 4)]["pstar_team"]))
            m(f"PstarOptLow{nm}", f2(Q[(side, 0)]["pstar_opt"])); m(f"PstarOptHigh{nm}", f2(Q[(side, 4)]["pstar_opt"]))
            m(f"KappaHigh{nm}", f1(Q[(side, 4)]["kappa_team_in"])); m(f"SRuleHigh{nm}", f1(Q[(side, 4)]["s_rule_in"]))
            m(f"KappaLow{nm}", f1(Q[(side, 0)]["kappa_team_in"])); m(f"SRuleLow{nm}", f1(Q[(side, 0)]["s_rule_in"]))
        L = ["\\begin{tabular}{llrrrrr}", "\\toprule",
             "Side & Stake quintile & Median $g$ & \\multicolumn{2}{c}{Confidence threshold $p^\\ast$} & \\multicolumn{2}{c}{Signal threshold (in)} \\\\",
             " & & & teams & DP benchmark & teams $\\kappa$ & one-line rule \\\\", "\\midrule"]
        for side, lab in (("off", "Batters"), ("def", "Fielding side")):
            for q in range(5):
                r = Q[(side, q)]
                L.append(f"{lab if q == 0 else ''} & {q + 1} & {r['g_median']:.4f} & {f2(r['pstar_team'])} & {f2(r['pstar_opt'])} & "
                         f"{f1(r['kappa_team_in'])} & {f1(r['s_rule_in'])} \\\\")
            L.append("\\addlinespace" if side == "off" else "\\bottomrule")
        L.append("\\end{tabular}")
        open(f"{OUT}/tab_s_mechanism.tex", "w").write("\n".join(L) + "\n")
        mf = [r for r in mc["model_free"] if r["band"] == "[0, 1)"]
        for side, nm in (("off", "Batter"), ("def", "Catcher")):
            rr = sorted((r for r in mf if r["side"] == side), key=lambda r: r["q"])
            m(f"MFRateLow{nm}", f"{100 * rr[0]['rate']:.0f}"); m(f"MFRateHigh{nm}", f"{100 * rr[-1]['rate']:.0f}")
    rq = get("re_quadrature.json")
    if rq:
        m("REQuadDLL", f1(rq["gh5"]["nll"] - rq["gh7"]["nll"]))
        m("REQuadSrPct", f1(100 * max(abs(rq["gh7"][k] / rq["gh5"][k] - 1) for k in ("off_sr", "def_sr"))))
        m("REQuadSdKFive", f2(rq["gh5"]["sd_kappa_off"])); m("REQuadSdKSeven", f2(rq["gh7"]["sd_kappa_off"]))
    c = get("causal.json")
    if c and "skip_branches_per_team_game" in c["point"]["opt_score"]:
        m("CausalSkipShare", f"{100 * c['point']['opt_score']['skip_branches_per_team_game'] / c['point']['opt_score']['branches_per_team_game']:.0f}")


def coverage_wp():
    """Coverage of the refit-bootstrap interval in a parametric simulation from the fitted
    model (results/coverage.json) and the extra spread from re-estimating the WP engine (results/wp_boot.json)."""
    c = J("coverage.json"); cv = c["coverage"]
    m("CovS", str(c["S"])); m("CovB", str(c["B"])); m("CovR", str(c["R"])); m("CovRPoint", str(c["R_point"]))
    m("CovGap", f2(c["truth"]["gap"])); m("CovNTrue", str(c["truth"]["n_seasons"]))
    for k, nm in (("percentile", ""), ("basic", "Basic"), ("normal", "Normal")):
        m(f"CovRate{nm}", f"{100 * cv[k]['rate']:.0f}"); m(f"CovRate{nm}CI", ci(cv[k]["ci"], lambda x: f"{100 * x:.1f}"))
        m(f"CovMissLow{nm}", str(cv[k]["miss_low"])); m(f"CovMissHigh{nm}", str(cv[k]["miss_high"]))
    m("CovWidth", f2(cv["percentile"]["mean_width"]))
    m("CovFailed", str(c["refits_failed"])); m("CovRefits", thou(c["S"] * (c["B"] + 1)))
    m("CovBias", f3(c["point_bias"])); m("CovPointSD", f3(c["point_sd_across_seasons"])); m("CovBootSD", f3(c["boot_sd_mean"]))
    w = J("wp_boot.json")
    m("WPBootB", str(w["B"])); m("WPBootR", str(w["R"])); m("WPBootSD", f3(w["dev"]["sd"]))
    m("WPBootDevCI", ci(w["dev"]["ci"], f3)); m("WPBootRatio", f2(w["sd_ratio"]))
    m("WinsLostCIWP", ci(w["combined"]["ci"])); m("WPBootSDComb", f3(w["combined"]["sd"])); m("WPBootRefitSD", f3(w["refit"]["sd"]))
    sh = w["shares"]["stake_0.008"]
    m("WPBootShareRuleCI", ci(sh["ci"], lambda x: pct(x)))


def main():
    os.makedirs(OUT, exist_ok=True)
    postseason()
    causal_gap()
    geometry()
    h, P, pv = headline()
    policy_table(P)
    perception(pv, h)
    identification()
    theory()
    drs()
    data_counts()
    thresholds()
    review_fixes()
    coverage_wp()
    review_extras()
    import supplement_numbers                      # S-macros and tab_s_*.tex for supplement.tex
    supplement_numbers.run(m, R, OUT)
    L =["% generated by paper/make_numbers.py from results/ -- do not edit"]
    L += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
    open(f"{OUT}/numbers.tex", "w").write("\n".join(L) + "\n")
    print(f"{len(macros)} macros; {sum(PENDING in v for v in macros.values())} pending")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
