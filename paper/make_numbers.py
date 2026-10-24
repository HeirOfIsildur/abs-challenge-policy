"""HONZA-77: every number quoted in the manuscript, read from results/ -> paper/generated/{numbers,tab_*}.tex.

The manuscript never types a result by hand: it uses the macros defined here (\\WinsLost, \\SrBatter, ...) and
\\input{}s the generated tables. Re-run after any change in results/ (``make -C paper``).
usage (from the repo root): python paper/make_numbers.py"""
import csv, json, math, os, re
import numpy as np

R = "results"
OUT = "paper/generated"
J = lambda f: json.load(open(os.path.join(R, f)))
macros = {}
DUPS = []
LATEX_CAPS = {"Prob", "E", "Phi", "Sref", "Mref", "Delta", "RequirePackage", "Theorem", "Section", "Table", "Figure",
              "Mathematica", "Rightarrow", "Leftarrow", "Pr", "S", "Lambda", "Gamma", "Omega", "Sigma", "Pi", "Psi",
              "Xi", "Upsilon", "Theta", "TeX", "LaTeX", "Bigl", "Bigm", "Bigr", "Big", "Bigg"}


def m(name, value):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    if name in macros:
        DUPS.append(name)              # HONZA-132: reported (and fatal) at the end, so all duplicates show at once
    macros[name] = value


# HONZA-106: a real minus sign (text-mode hyphens read as "[-0.004, 0.000]"), and no "-0.00"
neg = lambda t: re.sub(r"(?<![\w])-(?=\d)", r"\\ensuremath{-}", re.sub(r"(?<![\w.])-(0(?:\.0+)?)(?![\d.])", r"\1", t))
f1, f2, f3 = (lambda x: neg(f"{x:.1f}")), (lambda x: neg(f"{x:.2f}")), (lambda x: neg(f"{x:.3f}"))
pct = lambda x, d=1: f"{100 * x:.{d}f}"
ci = lambda c, f=f2: neg(f"[{f(c[0])}, {f(c[1])}]")
thou = lambda n: f"{int(round(n)):,}".replace(",", "{,}")
PENDING = "\\textbf{[pending]}"
# the paper names the Mathematica version once, in short form ("15.0"); the full $Version stays in results/*.json
short_version = lambda v: ".".join(v.split()[0].split(".")[:2])


def geometry():
    """HONZA-88 geometry audit (results/geometry_audit.json)."""
    g = J("geometry_audit.json"); r = g["reproduction"]
    # HONZA-133: the ruling agreement is read from the current count-joined frame (src/geometry_current.py);
    # geometry_audit.json keeps the historical plane / projection / square-corner comparisons on the older frame
    cur = J("geometry_current.json")
    m("GeomRepro", f2(100 * cur["share"])); m("GeomMismatch", str(cur["mismatches"]))
    m("GeomReproOld", f1(100 * r["statcast_projected_old"]["share"])); m("GeomN", thou(cur["n"]))
    m("GeomNOld", thou(r["statcast_raw"]["n"]))
    m("GeomReproFeed", f2(100 * r["feed_projected_mid"]["share"]))
    # HONZA-120: the square-corner rule used up to HONZA-114, and the ball-radius sweep of the round-ball rule
    sq = r["statcast_raw_square_corners"]
    m("GeomReproSquare", f2(100 * sq["share"])); m("GeomMismatchSquare", str(sq["mismatches"]))
    m("GeomRelabelCalled", str(sq["relabelled_called"])); m("GeomRelabelChallenged", str(sq["relabelled_challenged"]))
    sw = r["statcast_raw_radius_sweep"]
    m("GeomRadiusSweep", ", ".join(f"{sw[k]} at {k}\\,in" for k in ("1.40", "1.43", "1.45", "1.47", "1.50")))
    m("GeomShiftOld", f2(g["feed_vs_statcast"]["dz_raw_in"]))
    m("GeomPlaneSD", f3(g["plane"]["statcast_2026"]["mid"]["dz_sd"]))


def headline():
    """HONZA-132: the cross-fitted primary estimate (results/headline.json from src/headline_report.py) and its joint
    bootstrap; old macro names kept where the concept is unchanged (\\WinsLost, \\JointBootCI, \\WOpt = the benchmark)."""
    h = J("headline.json")
    pt, iv = h["point"], h["interval"]
    m("WinsLost", f2(pt["lost"])); m("WinsLostLeague", f"{30 * pt['lost']:.0f}")
    m("XFitS", str(pt["S"])); m("XFitR", str(pt["R"])); m("XFitK", str(pt["K"])); m("XFitSE", f3(pt["se_over_splits"]))
    m("JointBootCI", ci(iv["percentile"])); m("JointBootBasic", ci(iv["basic"])); m("JointBootNormal", ci(iv["normal"]))
    m("JointBootMean", f2(iv["mean"])); m("JointBootSD", f3(iv["sd"])); m("JointBootB", str(iv["B"])); m("JointBootR", str(iv["R"]))
    m("JointBootFailed", str(iv.get("B_failed", 0))); m("WinsLostCI", ci(iv["percentile"]))
    for sd, nm in (("batter", "Batter"), ("fielder", "Fielder")):
        r = h["by_role"][sd]
        m(f"WinsLost{nm}", f2(r["lost"])); m(f"WinsLost{nm}CI", ci(r["ci"]))
    P = {p["policy"]: p for p in h["policies"]}
    names = {"bench": "Opt", "runs_dp": "Runs", "opt": "Pooled", "card_pub": "CardPub", "card_refit": "CardRefit",
             "obs_rule": "FitRule", "savant": "Savant", "observed": "Obs", "stake_0.008": "Rule", "stake_0.01": "RuleOld",
             "clairvoyant": "Clair"}
    B = P["bench"]["wins"]
    for p, nm in names.items():
        q = P[p]
        m(f"W{nm}", f2(q["wins"])); m(f"W{nm}CI", ci(q["wins_ci"]))
        m(f"Share{nm}", pct(q["share"])); m(f"Share{nm}CI", ci(q["share_ci"], lambda x: pct(x)))
        m(f"N{nm}", f2(q["n"])); m(f"N{nm}CI", ci(q["n_ci"]))
        m(f"Lost{nm}", f2(B - q["wins"]))
    m("NMorePct", f"{100 * (P['bench']['n'] / P['observed']['n'] - 1):.0f}")
    o = h["one_line_rule"]
    m("RuleGapClosedPct", f"{100 * o['gap_closed']:.0f}"); m("RuleGapClosedCI", ci(o["gap_closed_ci"], lambda x: f"{100 * x:.0f}"))
    m("OptMinusRule", f2(B - P["stake_0.008"]["wins"])); m("RuleDelta", "0.008")
    m("RuleDeltaBest", o["best"].split("_")[1]); m("ShareRuleBest", pct(o["best_share"]))
    bs = o["best_by_split"]; m("RuleBestSplits", str(bs.get(o["best"], 0)))
    m("RuleLossStream", f"{100 - 100 * P['stake_0.008']['share']:.0f}")
    sv = h["savant_tuned"]
    m("SavantTunedC", f"{sv['c']:g}"); m("ShareSavantTuned", pct(sv["share"])); m("ShareSavantTunedCI", ci(sv["share_ci"], lambda x: pct(x)))
    fm = h.get("freq_matched")
    if fm:
        m("FMGapClosed", f"{100 * fm['gap_closed']['mean']:.0f}"); m("FMGapClosedCI", ci(fm["gap_closed"]["ci"], lambda x: f"{100 * x:.0f}"))
        m("FMDeltaCI", ci(fm["delta"], f3))
    co = h.get("co_primary")
    if co:
        m("CoLost", f2(co["lost"])); m("CoLostRemoved", f2(co["lost_lapses_removed"])); m("CoShareRule", pct(co["share_rule"]))
        if co.get("interval"):
            m("CoLostCI", ci(co["interval"]["percentile"])); m("CoB", str(co["interval"]["B"]))
    fs = h.get("full_sample", {})
    if "kappa_sr_region_g" in fs:
        m("WinsLostFull", f2(fs["kappa_sr_region_g"]["lost"])); m("LostMCSE", f3(fs["kappa_sr_region_g"]["mc_se"]))
        m("PointR", str(fs["kappa_sr_region_g"]["R"]))
    cf = J("crossfit.json")
    if "decomp" in cf:
        d = cf["decomp"]["mean_lost"]
        m("XDFullFull", f2(d["full_full"])); m("XDFullIn", f2(d["full_in"])); m("XDOutFull", f2(d["out_full"])); m("XDOutIn", f2(d["out_in"]))
    if "kappa_sr_region_g_K2" in cf:
        m("WinsLostKTwo", f2(cf["kappa_sr_region_g_K2"]["lost"]))
    bb = h.get("block_boot")
    if bb and "percentile" in bb.get("lost", {}):
        m("BlockBootCI", ci(bb["lost"]["percentile"])); m("BlockBootB", str(bb["B"]))
    return h, P


def policy_table(P):
    """HONZA-132 (Astra minor 12): the wide policy table split in two -- benchmark and observed play (tab_policies),
    and the rules a team could implement, with the share of the gap each recovers (tab_rules)."""
    h = J("headline.json")
    lab = {"bench": "Pooled-model benchmark (primary intervention)", "opt": "Earlier DP (no score, fixed opportunity counts)",
           "clairvoyant": "Clairvoyant comparator (knows its game's future $(p,g)$; not a bound)",
           "observed": "\\textbf{Observed 2026 (realised)}", "obs_rule": "Teams' fitted rule $s>\\kappa(x)$ (simulated)"}
    L = ["\\begin{tabular}{p{5.0cm}ccc}", "\\toprule", "Policy & Wins & Share of benchmark (\\%) & Challenges \\\\", "\\midrule"]
    for p in ("bench", "opt", "clairvoyant", "obs_rule", "observed"):
        q = P[p]
        share = "100" if p == "bench" else f"{pct(q['share'])} {ci(q['share_ci'], lambda x: pct(x))}"
        L.append(f"{lab[p]} & {f2(q['wins'])} {ci(q['wins_ci'])} & {share} & {f2(q['n'])} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"{OUT}/tab_s_oldpolicies.tex", "w").write("\n".join(L) + "\n")   # HONZA-134: the stakes-only estimator
    rows = h["policies"]; Pw = {p["policy"]: p for p in rows}
    obs, ben = Pw["observed"]["wins"], Pw["bench"]["wins"]
    sv = h["savant_tuned"]
    lab = {"stake_0.008": "One-line rule, $\\delta=0.008$", "stake_0.01": "One-line rule, $\\delta=0.01$",
           "runs_dp": "Runs-objective rule \\citep{Namuduri2026runs}", "card_pub": "48-cell card, published \\citep{Arora2026use}",
           "card_refit": "48-cell card, median DP thresholds", "savant": "Fixed 0.2-run rule \\citep{Savant2026abs}"}
    L = ["\\begin{tabular}{p{4.6cm}ccc}", "\\toprule", "Rule & Share of benchmark (\\%) & Gap recovered (\\%) & Challenges \\\\", "\\midrule"]
    for p in ("runs_dp", "stake_0.008", "stake_0.01", "card_refit", "card_pub", "savant"):
        q = Pw[p]
        rec = neg(f"{100 * (q['wins'] - obs) / (ben - obs):.0f}")
        L.append(f"{lab[p]} & {pct(q['share'])} {ci(q['share_ci'], lambda x: pct(x))} & {rec} & {f2(q['n'])} \\\\")
    L.append(f"Savant's form, run cost {sv['c']:g} & {pct(sv['share'])} {ci(sv['share_ci'], lambda x: pct(x))} & "
             f"{100 * (sv['wins'] - obs) / (ben - obs):.0f} & {f2(sv['n'])} \\\\")
    fm = h.get("freq_matched")
    if fm:
        L.append(f"One-line rule at the teams' challenge rate & --- & {100 * fm['gap_closed']['mean']:.0f} "
                 f"{ci(fm['gap_closed']['ci'], lambda x: f'{100 * x:.0f}')} & {f2(Pw['observed']['n'])} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"{OUT}/tab_s_oldrules.tex", "w").write("\n".join(L) + "\n")


def carried():
    """HONZA-132: macros the HONZA-130 headline() derived from other result files, read from those files directly."""
    def get(f):
        try:
            return J(f)
        except FileNotFoundError:
            return None
    fb = get("flat_bound.json")
    if fb:
        b = fb["best"]
        m("FlatLoss", f1(b["loss_pct"])); m("FlatLocal", f1(b["local_pct"])); m("FlatBound", f1(b["bound_pct"]))
        m("FlatIdentityGap", f"{fb['max_identity_gap']:.0e}".replace("e-", "\\times 10^{-") + "}")
        m("FlatBestDelta", str(fb["best_delta"]))
    cg = get("causal_gap.json")
    if cg:
        m("SimGShare", pct(cg["mc_check"]["sim"]["chain_over_engine"])); m("SimGShareRun", pct(cg["mc_check"]["run"]["chain_over_engine"]))
        m("SimGShareRunGap", pct(1 - cg["mc_check"]["run"]["chain_over_engine"]))
        m("GapPredSim", f3(cg["by_policy"]["opt_score"]["predicted"]["sim"])); m("GapPredRun", f3(cg["by_policy"]["opt_score"]["predicted"]["run"]))
        m("HomeFieldLostChange", f3(cg["home_field"]["first_order_change_in_wins_lost"]))
        m("HomeFieldRelG", pct(abs(cg["home_field"]["rel_g_opt_branches"])))
    c = get("causal.json")
    if c:
        gb = c["game_bootstrap"]["opt_score"]
        m("BinaryLost", f2(gb["binary_minus_observed"][0])); m("BinaryLostCI", ci(gb["binary_minus_observed"][1:]))
        m("BinaryMinusRB", f3(gb["binary_minus_rb"][0])); m("BinaryMinusRBCI", ci(gb["binary_minus_rb"][1:], f3))
        m("CausalBootN", thou(c.get("boot_resamples", 2000)))
        m("SCausalOppEffect", f3(gb["causal_rb"][0] - gb["causal_rb_own"][0]))
        m("CausalMinusPath", f3(gb["causal_minus_path"][0])); m("CausalMinusPathCI", ci(gb["causal_minus_path"][1:], f3))
    cs = get("causal_sim.json")
    if cs:
        b = cs["game_bootstrap"]["opt_score"]["binary_minus_rb"]
        m("BinaryMinusRBSimAbs", f2(abs(b[0])))
    h = J("headline.json")
    ms = h.get("model_spread")
    if ms:
        pv = J("perception_variants.json"); base = pv["normal"]["nll"]
        hom = [k for k in ms if k in pv and not pv[k]["spec"].get("re") and not pv[k]["spec"].get("lapse")
               and pv[k]["spec"].get("law", "normal") == "normal" and pv[k]["nll"] <= base + 1e-6]
        ls = [ms[k]["lost"] for k in hom]
        if ls:
            m("SpreadLo", f2(min(ls))); m("SpreadHi", f2(max(ls)))
        lr = [ms[k]["lost"] for k in ("main_logistic", "main_t10", "main_t5", "main_t3") if k in ms]
        if lr:
            m("SpreadRejLo", f2(min(lr))); m("SpreadRejHi", f2(max(lr)))
        b0 = pv["kappa_sr_region_g"]["bic"]
        good = [k for k in ms if k in pv and pv[k]["bic"] <= b0 + 1e-6 and k not in ("re", "re_lapse")]
        allv = [ms[k]["lost"] for k in good] + [ms[k]["lost_kept"] for k in good if ms[k].get("lost_kept") is not None]
        if allv:
            m("SpreadAllLo", f2(min(allv))); m("SpreadAllHi", f2(max(allv)))
        m("VariantR", str(h["variants_R"]))
    co = h.get("co_primary")
    if co:
        m("PPLost", f2(co["lost"]))


def honza132():
    """HONZA-132 (round-3 reviews, GPT-6.1-Sol and GPT-6-Astra): join audit, half-innings, stakes validation, the
    intervention grid, coverage under misspecification, the mechanism with uncertainty, the paired causal contrast."""
    def get(f):
        try:
            return J(f)
        except FileNotFoundError:
            return None
    ja = get("join_audit.json")
    if ja:
        cj = ja["count_join"]
        m("NRenumbered", thou(cj["renumbered"])); m("NUnmatched", str(cj["unmatched"]))
        m("RenumMaxDiff", f3(cj["renumbered_max_abs_diff_vs_feed_in"])); m("NCountIncons", str(ja["old_join"]["description_ok_count_inconsistent"]))
        m("NHalvesZero", str(ja["halves"]["without_opportunity_frame_rows"])); m("NHalvesAll", thou(ja["halves"]["statcast_halves"]))
    sv = get("stakes_validation.json")
    if sv:
        st = sv["statcast"]
        m("SVSharePct", f"{100 * st['share']:.1f}"); m("SVCorr", f3(st["corr"])); m("SVCorrCh", f3(st["challenged"]["corr"]))
        m("SVRatioPct", f"{100 * st['ratio_of_means']:.1f}"); m("SVRatioChPct", f"{100 * st['challenged']['ratio_of_means']:.1f}")
        e = sv["edge_test"]; oh = e["ours_held_out"]
        m("SVSlope", f2(oh["slope_mean"])); m("SVSlopeSE", f2(oh["se_typical"]))
        m("SVSlopeCI", ci([oh["slope_mean"] - 1.96 * oh["se_typical"], oh["slope_mean"] + 1.96 * oh["se_typical"]]))
        hi = e["ours_by_leverage_tercile"][2]
        m("SVSlopeHigh", f2(hi["slope"])); m("SVSlopeHighCI", ci(hi["ci"])); m("SVSlopeSC", f2(e["statcast"]["slope"]))
        m("SVN", thou(oh["splits"][0]["n"])); m("SVHalfCorr", f3(sv["half_engines"]["corr_mean"]))
        po = sv.get("policies")
        if po:
            m("SVLostSC", f2(po["statcast_everything"]["lost"])); m("SVLostOursSC", f2(po["ours_decide_statcast_score"]["lost"]))
            m("SVLostOursSame", f2(po["ours_same_rows"]["lost"]))
        L = ["\\begin{tabular}{rrrrr}", "\\toprule", "Leverage decile & Pitches & Mean $g$, ours & Mean $g$, Statcast & Correlation \\\\", "\\midrule"]
        for r in st["by_leverage_decile"]:
            L.append(f"{r['decile'] + 1} & {thou(r['n'])} & {r['mean_ours']:.4f} & {r['mean_statcast']:.4f} & {f3(r['corr'])} \\\\")
        L += ["\\bottomrule", "\\end{tabular}"]
        open(f"{OUT}/tab_s_svcells.tex", "w").write("\n".join(L) + "\n")
    iv = get("intervention.json")
    if iv:
        M = iv["models"]; m("IVR", str(iv["R"]))
        m("PrimLo", f2(iv["primary_range"][0])); m("PrimHi", f2(iv["primary_range"][1]))
        m("PIDHi", f2(max(c["lost"] for v in M.values() for c in v["cells"])))
        mh = [c for c in M["kappa_sr_region_g"]["cells"] if c["rho"] == 0.5]
        if mh: m("PIDMainHalf", f2(mh[0]["lost"]))
        if "main_time" in M: m("TimeLost", f2(M["main_time"]["primary"]))
        lab = {"kappa_sr_region_g": "Main", "main_lapse": "Main + lapse", "main_time": "Main, time-varying",
               "main_time_lapse": "Main, time-varying + lapse", "re_gh11": "Player effects",
               "re_lapse_gh11": "Player effects + lapse (primary)"}
        rhos = iv["rhos"]; lams = iv["lams"]
        L = ["\\begin{tabular}{l" + "c" * len(rhos) + "}", "\\toprule",
             "Model ($\\ell$) & " + " & ".join(f"$\\rho={r:g}$" for r in rhos) + " \\\\", "\\midrule"]
        for k, v in M.items():
            for lam in (lams if v["lapse"] else [0.0]):
                cells = {c["rho"]: c["lost"] for c in v["cells"] if c["lam"] == lam}
                nm = lab.get(k, k) + (f" ($\\ell={lam:g}$)" if v["lapse"] else "")
                L.append(nm + " & " + " & ".join(f2(cells[r]) for r in rhos) + " \\\\")
        L += ["\\bottomrule", "\\end{tabular}"]
        open(f"{OUT}/tab_s_pidall.tex", "w").write("\n".join(L) + "\n")     # HONZA-134: supplement; main text in honza134
    pv = J("perception_variants.json"); b0 = pv["kappa_sr_region_g"]["bic"]
    if "re_lapse_gh11" in pv:
        m("BICRELapse", thou(pv["re_lapse_gh11"]["bic"]))
        r = pv["re_lapse_gh11"]["params"]
        m("REKappaSDBatter", f2(r["off_re"]["sd_kappa"])); m("REKappaSDCatcher", f2(r["def_re"]["sd_kappa"]))
        m("RELogSrSDBatter", f2(r["off_re"]["sd_logsr"])); m("RELogSrSDCatcher", f2(r["def_re"]["sd_logsr"]))
    if "main_time" in pv:
        m("TimeDBICtxt", f"BIC better by {thou(b0 - pv['main_time']['bic'])}")
        m("TimeDBICnum", thou(b0 - pv["main_time"]["bic"])); m("TimeDBICLapse", thou(b0 - pv["main_time_lapse"]["bic"]))
    if "re_lapse_gh11" in pv and "re_lapse" in pv:
        a5, a11 = pv["re_lapse"], pv["re_lapse_gh11"]
        m("REGHDLL", f1(a5["nll"] - a11["nll"]))
        m("REGHSrPct", f1(100 * max(abs(a11["params"][k] / a5["params"][k] - 1) for k in ("off_sr", "def_sr"))))
        m("REGHSdKFive", f2(a5["params"]["off_re"]["sd_kappa"])); m("REGHSdKEleven", f2(a11["params"]["off_re"]["sd_kappa"]))
        m("DLLREvsMain", thou(pv["kappa_sr_region_g"]["nll"] - pv["re_gh11"]["nll"]))
    rc = get("re_calibration.json")
    if rc:
        a, b = rc["re_lapse_gh11_off"], rc["re_lapse_gh11_def"]
        m("RECalObsOff", thou(a["observed"])); m("RECalObsDef", thou(b["observed"]))
        m("RECalIntOff", thou(a["integrated"])); m("RECalIntDef", thou(b["integrated"]))
        m("RECalPlugOff", thou(a["plug_in"])); m("RECalPlugDef", thou(b["plug_in"]))
        m("RECalMainOff", thou(rc["kappa_sr_region_g_off"]["integrated"])); m("RECalMainDef", thou(rc["kappa_sr_region_g_def"]["integrated"]))
        m("RECalOffPct", f"{100 * (1 - a['integrated'] / a['observed']):.0f}")
    # model table (main text): the BIC-competitive set
    rows = [("kappa_sr_region_g", "\\textbf{Main}: $\\kappa$ piecewise linear in $\\log g$, $\\kappa,\\sigma_r$ by region"),
            ("main_lapse", "Main + lapse rate"), ("main_time", "Main, second-half shifts in $\\kappa$ and $\\sigma_r$"),
            ("main_time_lapse", "Same + lapse rate"), ("re_gh11", "Main + player effects ($11\\times11$ nodes)"),
            ("re_lapse_gh11", "Main + player effects + lapse (primary)"),
            ("kappa_sr_region", "$\\kappa$ linear in $\\log g$"), ("main_t10", "Main, $t_{10}$ noise"),
            ("main_logistic", "Main, logistic noise")]
    ivm = iv["models"] if iv else {}
    L = ["\\begin{tabular}{p{5.6cm}rrcc}", "\\toprule", "Model & $k$ & $\\Delta$BIC & $\\sigma_r$ batter / catcher (in) & Wins lost \\\\", "\\midrule"]
    for k, lab_ in rows:
        if k not in pv:
            continue
        v = pv[k]; lost = ivm.get(k, {}).get("primary")
        L.append(f"{lab_} & {v['k']} & ${v['bic'] - b0:+.0f}$ & {f2(v['params']['off_sr'])} / {f2(v['params']['def_sr'])} & "
                 f"{f2(lost) if lost is not None else '---'} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"{OUT}/tab_variants.tex", "w").write("\n".join(L) + "\n")
    cv = get("coverage_stream.json")
    if cv:
        Sc = cv["scenarios"]; c0 = Sc.get("kappa_sr_region_g")
        pl = cv["config"]["plan"]
        m("CovB", str(pl["kappa_sr_region_g"]["B"])); m("CovR", str(cv["config"]["R"])); m("CovTruthStreams", str(cv["config"]["n_truth"]))
        m("CovBMis", str(pl["main_lapse"]["B"])); m("CovSMis", str(pl["main_lapse"]["S"]))
        if c0 and c0["S"]:
            m("CovS", str(c0["S"])); m("CovMainRate", f"{100 * c0['coverage']['percentile']['rate']:.0f}")
            m("CovMainRateCI", ci(c0["coverage"]["percentile"]["ci"], lambda x: f"{100 * x:.1f}"))
            m("CovMainBasic", f"{100 * c0['coverage']['basic']['rate']:.0f}"); m("CovMainNormal", f"{100 * c0['coverage']['normal']['rate']:.0f}")
            m("CovMainBias", f3(c0["point_bias"])); m("CovGap", f2(c0["truth"]["gap"]))
            m("CovPointSD", f3(c0["point_sd_across_seasons"])); m("CovBootSD", f3(c0["boot_sd_mean"]))
            m("CovMainMissLow", str(c0["coverage"]["percentile"]["miss_low"])); m("CovMainMissHigh", str(c0["coverage"]["percentile"]["miss_high"]))
            m("CovMainBiasAbs", f2(abs(c0["point_bias"])))
        for k, nm in (("main_lapse", "Lapse"), ("re_lapse_gh11", "RE")):
            c = Sc.get(k)
            if c and c["S"]:
                m(f"Cov{nm}Rate", f"{100 * c['coverage']['percentile']['rate']:.0f}"); m(f"Cov{nm}Bias", f2(c["point_bias"]))
                m(f"Cov{nm}Normal", f"{100 * c['coverage']['normal']['rate']:.0f}")
                m(f"Cov{nm}Gap", f2(c["truth"]["gap"])); m(f"Cov{nm}S", str(c["S"]))
    mb = get("mechanism_boot.json")
    if mb and mb.get("B", 0) >= 2:
        m("MechB", str(mb["B"]))
        D = mb["decomp"]
        for k, nm in (("intercept_only", "Int"), ("stake_linear_only", "Stake"), ("both_linear", "Both"), ("stake_scale_only", "StakeScale")):
            m(f"Dec{nm}Gap", f"{100 * D[k]['point']:.0f}"); m(f"Dec{nm}GapCI", ci(D[k]["ci"], lambda x: f"{100 * x:.0f}"))
            m(f"Dec{nm}N", f2(D[k]["n"]))
        m("DecStakeBeatsPct", f"{100 * D['stake_minus_intercept']['share_positive']:.0f}")
        S_ = mb["slopes"]
        for side, nm in (("off", "Batter"), ("def", "Catcher")):
            for who, w in (("teams", "Team"), ("opt", "Opt")):
                x = S_[f"{side}_all_full_{who}"]
                # normal interval point +- 1.96 SD: the teams' slope runs through belief maps drawn afresh in every
                # replicate, so the replicates scatter around a centre that need not equal the point's draw
                m(f"Slope{w}{nm}", f2(x["point"])); m(f"Slope{w}{nm}CI", ci([x["point"] - 1.96 * x["sd"], x["point"] + 1.96 * x["sd"]]))
                m(f"Slope{w}{nm}FE", f2(S_[f"{side}_all_count_fe_{who}"]["point"]))
    c = get("causal.json")
    if c and "paired_rb" in c["full_vs_behaviour"]["opt_score"]:
        fv = c["full_vs_behaviour"]["opt_score"]
        m("PairRB", f2(fv["paired_rb"][0])); m("PairRBCI", ci(fv["paired_rb"][1:])); m("PairWin", f2(fv["paired_win"][0]))
        m("PairPath", f2(fv["paired_path"][0])); m("FullRB", f2(fv["rb"][0])); m("FullRBCI", ci(fv["rb"][1:]))
        m("FullWin", f2(fv["win"][0])); m("FullWinCI", ci(fv["win"][1:])); m("FullPathObs", f2(fv["path_minus_observed"][0]))
        m("FullPathRule", f2(fv["path_minus_obs_rule"][0]))
        m("WinMinusRBAbs", f2(abs(np.mean([fv["paired_win_minus_rb"][0], fv["win_minus_rb"][0]]))))
        m("CausalR", str(c["R"])); m("CausalK", str(c["K"]))


PRIMARY = "re_lapse_gh11"                      # HONZA-134: the BIC-best model of the decision set is the primary model
COMPARATOR = "kappa_sr_region_g"               # the homogeneous main model, now the comparator


def honza134():
    """HONZA-134: the full policy cross-fit (results/policy_crossfit.json, src/policy_crossfit.py) is the headline
    estimator, for the primary model (player effects + lapse) and the comparator (homogeneous main model); refit
    bootstraps of the whole procedure (results/pcf_boot_<model>.json); the coverage study on the 2026 stream
    (results/coverage_pcf.json); the state-dependent criterion rows."""
    def get(f):
        try:
            return J(f)
        except FileNotFoundError:
            return None
    pc = get("policy_crossfit.json") or {}
    pt = lambda k: pc.get(k + "_grid") or pc.get(k)
    for key, nm in ((PRIMARY, "Gap"), (COMPARATOR, "GapMain")):
        v = pt(key)
        if not v:
            continue
        m(nm, f2(v["lost"])); m(f"{nm}S", str(v["S"])); m(f"{nm}R", str(v["R"])); m(f"{nm}K", str(v["K"]))
        m(f"{nm}SplitSD", f3(v["lost_sd_over_splits"])); m(f"{nm}SplitSE", f3(v["lost_se_splits"])); m(f"{nm}MCSE", f3(v["lost_mc_se_within"]))
        m(f"{nm}Bench", f2(v["bench"])); m(f"{nm}Obs", f2(v["observed"])); m(f"{nm}NBench", f2(v["n_bench"]))
        m(f"{nm}NObs", f2(v["n_observed"])); m(f"{nm}NMorePct", f"{100 * (v['n_bench'] / v['n_observed'] - 1):.0f}")
        m(f"{nm}League", f"{30 * v['lost']:.0f}"); m(f"{nm}ObsShare", pct(v["observed"] / v["bench"]))
        ru = v["rules"]
        for r_, rn in (("stake_0.008", "Rule"), ("stake_frozen", "Frozen"), ("savant", "Savant"), ("obs_rule", "FitRule")):
            m(f"{nm}{rn}Share", pct(ru[r_]["share"])); m(f"{nm}{rn}GapPct", neg(f"{100 * ru[r_]['gap_closed']:.0f}"))
            m(f"{nm}{rn}N", f2(ru[r_]["n"])); m(f"{nm}{rn}Wins", f2(ru[r_]["wins"]))
        best = max((k for k in ru if k.startswith("stake_0") ), key=lambda k: ru[k]["wins"])
        m(f"{nm}BestDelta", best.split("_")[1]); m(f"{nm}BestShare", pct(ru[best]["share"]))
        m(f"{nm}DTrainLo", f"{min(v['d_train']):.4f}"); m(f"{nm}DTrainHi", f"{max(v['d_train']):.4f}")
        G = {(g["rho"], g["lam"]): g["lost"] for g in v["grid"]}
        if (0.5, 0.0) in G: m(f"{nm}RhoHalf", f2(G[(0.5, 0.0)]))
        if (0.25, 0.0) in G: m(f"{nm}RhoQuarter", f2(G[(0.25, 0.0)]))
        if (1.0, 1.0) in G: m(f"{nm}LamOne", f2(G[(1.0, 1.0)]))
        if (1.0, 0.5) in G: m(f"{nm}LamHalf", f2(G[(1.0, 0.5)]))
        if len(G) > 1: m(f"{nm}GridHi", f2(max(G.values())))
        b = get(f"pcf_boot_{key}.json")
        if b:
            L_ = b["lost"]
            # HONZA-134 (coverage study, round 1): the index's interval is the normal one (point +- 1.96 bootstrap SD)
            m(f"{nm}CI", ci(L_["normal"])); m(f"{nm}Pct", ci(L_["percentile"])); m(f"{nm}Basic", ci(L_["basic"])); m(f"{nm}Normal", ci(L_["normal"]))
            m(f"{nm}BootMean", f2(L_["mean"])); m(f"{nm}BootSD", f3(L_["sd"])); m(f"{nm}B", str(b["B"]))
            m(f"{nm}BFail", str(b["B"] - b["B_ok"])); m(f"{nm}BootBias", f3(L_["bias"]))
            m(f"{nm}RuleGapCI", ci(b["gap_closed"]["stake_0.008"]["ci"], lambda x: f"{100 * x:.0f}"))
            m(f"{nm}FrozenGapCI", ci(b["gap_closed"]["stake_frozen"]["ci"], lambda x: f"{100 * x:.0f}"))
            m(f"{nm}RuleShareCI", ci(b["share"]["stake_0.008"]["ci"], lambda x: pct(x)))
            m(f"{nm}SavantShareCI", ci(b["share"]["savant"]["ci"], lambda x: pct(x)))
            m(f"{nm}BenchCI", ci(b["bench"])); m(f"{nm}ObsCI", ci(b["observed"]))
            m(f"{nm}BootMinutes", f"{b['secs_median'] / 60:.0f}")
            m(f"{nm}Refits", thou(5 * b["B"])); m(f"{nm}BootHours", thou(b["B"] * b["secs_median"] / 3600))
            if b.get("lost_ok_only"):
                d_ = max(abs(a - c) for a, c in zip(b["lost_ok_only"]["percentile"], L_["percentile"]))
                m(f"{nm}BFailDiff", f3(d_))
    # HONZA-134 (round-1 referees): competitor and re-tuned rules in the same cross-fit (<model>_ext runs), the estimator's
    # own choices (exact-count cells, K, training-only noise quintiles)
    for key, nm in ((PRIMARY, "Gap"), (COMPARATOR, "GapMain")):
        v = pt(key)
        if v and "stake_0.01" in v["rules"]:
            m(f"{nm}RuleTenGapPct", f"{100 * v['rules']['stake_0.01']['gap_closed']:.0f}")
        e = pc.get(key + "_ext")
        if e:
            for r_, rn in (("runs_dp", "Runs"), ("card_pub", "CardPub"), ("card_refit", "CardRefit"), ("savant_tuned", "SavantTuned"),
                           ("retune_intercept", "RetuneInt"), ("retune_slope", "RetuneSlope"), ("retune_both", "RetuneBoth")):
                if r_ in e["rules"]:
                    m(f"{nm}{rn}GapPct", neg(f"{100 * e['rules'][r_]['gap_closed']:.0f}")); m(f"{nm}{rn}N", f2(e["rules"][r_]["n"]))
                    m(f"{nm}{rn}Share", pct(e["rules"][r_]["share"]))
            m(f"{nm}ExtS", str(e["S"]))
        for suf, mn in (("_calfine", "CalFine"), ("_trainq5", "TrainQFive"), ("_K2", "KTwo"), ("_K10", "KTen"),
                        ("_naux8", "NauxEight"), ("_calfine_naux8", "CalFineNauxEight")):
            if pc.get(key + suf):
                m(f"{nm}{mn}", f2(pc[key + suf]["lost"]))
    qd = get("quadrature_primary.json")
    if qd:
        m("QuadDiff", f1(abs(qd["diff_11_vs_31"])))
    sds = [pt(k_)["lost_sd_over_splits"] for k_ in ("kappa_sr_region_g", "main_lapse", "main_time", "main_time_lapse", "main_state",
                                                  "re_gh11", "re_lapse_gh11", "re_lapse_state_gh11") if pt(k_)]
    if sds:
        m("SplitSDLo", f3(min(sds))); m("SplitSDHi", f3(max(sds)))
    rp = get("recovery_primary.json")
    if rp:
        P_ = rp["params"]; m("RecPN", str(rp["replicates"]))
        for sd, nm in (("off", "Batter"), ("def", "Catcher")):
            m(f"RecPLapse{nm}", neg(f"{100 * P_[f'{sd}_lapse']['bias']:+.1f}")); m(f"RecPSr{nm}", neg(f"{100 * P_[f'{sd}_sr']['rel_bias']:+.1f}"))
            m(f"RecPSdK{nm}", neg(f"{100 * P_[f'{sd}_sd_kappa']['rel_bias']:+.0f}"))
            m(f"RecPLapseSD{nm}", f1(100 * P_[f"{sd}_lapse"]["sd"]))
    # the decision models at least as good as the comparator by BIC, all through the full policy cross-fit
    pv = J("perception_variants.json"); b0 = pv[COMPARATOR]["bic"]
    bic = {k: pv[k]["bic"] for k in pv}
    good = [k for k in ("kappa_sr_region_g", "main_lapse", "main_time", "main_time_lapse", "main_state", "re_gh11",
                        "re_lapse_gh11", "re_lapse_state_gh11") if pt(k) and k in bic and bic[k] <= b0 + 1e-9]
    if good:
        vals = [pt(k)["lost"] for k in good]
        m("PCFLo", f2(min(vals))); m("PCFHi", f2(max(vals))); m("PCFNModels", str(len(good)))
    for k, nm in (("main_state", "GapState"), ("re_lapse_state_gh11", "GapREState"), ("main_lapse", "GapLapse"),
                  ("main_time", "GapTime"), ("re_gh11", "GapRE")):
        if pt(k):
            m(nm, f2(pt(k)["lost"]))
    if "main_state" in pv:
        m("StateDBIC", thou(b0 - pv["main_state"]["bic"]))
        st = pv["main_state"]["params"]
        m("StateTwoOutsBatter", f2(st["off_kappa_state"]["two_outs"])); m("StateTwoOutsCatcher", f2(st["def_kappa_state"]["two_outs"]))
        m("StateSrBatter", f2(st["off_sr"])); m("StateSrCatcher", f2(st["def_sr"]))
    if "re_lapse_state_gh11" in pv:
        r0, r1 = pv[PRIMARY], pv["re_lapse_state_gh11"]
        m("REStateDBIC", thou(r0["bic"] - r1["bic"]))
        m("REStateSrBatter", f2(r1["params"]["off_sr"])); m("REStateSrCatcher", f2(r1["params"]["def_sr"]))
    if PRIMARY in pv:
        pp = pv[PRIMARY]["params"]; m("PK", str(pv[PRIMARY]["k"]))
        m("PSrBatter", f2(pp["off_sr"])); m("PSrCatcher", f2(pp["def_sr"]))
        m("PLapseBatter", f"{100 * pp['off_lapse']:.0f}"); m("PLapseCatcher", f"{100 * pp['def_lapse']:.0f}")
        m("PDBIC", thou(b0 - pv[PRIMARY]["bic"]))
    # model table (main text): every model at least as good as the comparator by BIC, cross-fitted gap
    rows = [(PRIMARY, "\\textbf{Primary}: player effects ($11\\times11$ nodes) + lapse"),
            ("re_lapse_state_gh11", "Primary + game-state terms in $\\kappa$"),
            ("re_gh11", "Player effects, no lapse"),
            (COMPARATOR, "Comparator (homogeneous): $\\kappa$ piecewise linear in $\\log g$, $\\kappa,\\sigma_r$ by region"),
            ("main_state", "Comparator + game-state terms in $\\kappa$"),
            ("main_lapse", "Comparator + lapse rate"), ("main_time", "Comparator, second-half shifts in $\\kappa$ and $\\sigma_r$"),
            ("main_time_lapse", "Same + lapse rate")]
    L = ["\\begin{tabular}{p{5.6cm}rrcc}", "\\toprule",
         "Model & $k$ & $\\Delta$BIC & $\\sigma_r$ batter / catcher (in) & Gap \\\\", "\\midrule"]
    for k, lab_ in rows:
        if k not in pv:
            continue
        v = pv[k]; lost = pt(k)["lost"] if pt(k) else None
        L.append(f"{lab_} & {v['k']} & ${v['bic'] - b0:+.0f}$ & {f2(v['params']['off_sr'])} / {f2(v['params']['def_sr'])} & "
                 f"{f2(lost) if lost is not None else '---'} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"{OUT}/tab_variants.tex", "w").write("\n".join(L) + "\n")
    # sensitivity grid (main text): primary and comparator, full policy cross-fit
    rhos = [1.0, 0.75, 0.5, 0.25]
    L = ["\\begin{tabular}{l" + "c" * len(rhos) + "}", "\\toprule",
         "Model ($\\ell$) & " + " & ".join(f"$\\rho={r:g}$" for r in rhos) + " \\\\", "\\midrule"]
    for key, lab_, lams in ((PRIMARY, "Primary", [0.0, 0.5, 1.0]), (COMPARATOR, "Comparator", [0.0])):
        v = pt(key)
        if not v or len(v["grid"]) < 2:
            continue
        G = {(g["rho"], g["lam"]): g["lost"] for g in v["grid"]}
        for lam in lams:
            nm_ = lab_ + (f" ($\\ell={lam:g}$)" if key == PRIMARY else "")
            L.append(nm_ + " & " + " & ".join(f2(G[(r, lam)]) if (r, lam) in G else "---" for r in rhos) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"{OUT}/tab_pid.tex", "w").write("\n".join(L) + "\n")
    # benchmark and observed play (main text), both models, full policy cross-fit
    v = pt(PRIMARY); b = get(f"pcf_boot_{PRIMARY}.json"); vm = pt(COMPARATOR); bm = get(f"pcf_boot_{COMPARATOR}.json")
    if v and vm:
        def cell(vv, bb, key, wins):
            return f2(wins) + (" " + ci(bb[key]) if bb and key in bb else "")
        L = ["\\begin{tabular}{p{3.9cm}cccc}", "\\toprule",
             " & \\multicolumn{2}{c}{Primary} & \\multicolumn{2}{c}{Comparator} \\\\",
             "Policy & Wins & Challenges & Wins & Challenges \\\\", "\\midrule"]
        L.append("Benchmark (primary intervention) & " + cell(v, b, "bench", v["bench"]) + f" & {f2(v['n_bench'])} & "
                 + cell(vm, bm, "bench", vm["bench"]) + f" & {f2(vm['n_bench'])} \\\\")
        L.append("Teams' fitted rule $s>\\kappa(x)$, simulated & " + f2(v["rules"]["obs_rule"]["wins"]) + f" & {f2(v['rules']['obs_rule']['n'])} & "
                 + f2(vm["rules"]["obs_rule"]["wins"]) + f" & {f2(vm['rules']['obs_rule']['n'])} \\\\")
        L.append("\\textbf{Observed 2026 (realised)} & " + cell(v, b, "observed", v["observed"]) + f" & {f2(v['n_observed'])} & "
                 + cell(vm, bm, "observed", vm["observed"]) + f" & {f2(vm['n_observed'])} \\\\")
        L.append("\\midrule")
        L.append("\\textbf{Accounting index} & " + f2(v["lost"]) + (" " + ci(b["lost"]["normal"]) if b else "") + " & & "
                 + f2(vm["lost"]) + (" " + ci(bm["lost"]["normal"]) if bm else "") + " & \\\\")
        L += ["\\bottomrule", "\\end{tabular}"]
        open(f"{OUT}/tab_policies.tex", "w").write("\n".join(L) + "\n")
    # rules (main text), primary model, full policy cross-fit, bootstrap intervals
    v = pt(PRIMARY); b = get(f"pcf_boot_{PRIMARY}.json"); vm = pt(COMPARATOR); bm = get(f"pcf_boot_{COMPARATOR}.json")
    if v:
        lab = {"stake_0.008": "One-line rule, $\\delta=0.008$", "stake_frozen": "One-line rule, $\\delta$ chosen on the training folds",
               "stake_0.01": "One-line rule, $\\delta=0.01$", "savant": "Fixed 0.2-run rule \\citep{Savant2026abs}",
               "obs_rule": "Teams' fitted rule $s>\\kappa(x)$, simulated"}
        L = ["\\begin{tabular}{p{4.2cm}cccc}", "\\toprule",
             " & \\multicolumn{2}{c}{Primary} & \\multicolumn{2}{c}{Comparator} \\\\",
             "Rule & Gap (\\%) & Chall. & Gap (\\%) & Chall. \\\\", "\\midrule"]
        ve, vme = pc.get(PRIMARY + "_ext"), pc.get(COMPARATOR + "_ext")
        lab.update({"runs_dp": "Runs-objective rule \\citep{Namuduri2026runs}", "card_pub": "48-cell card, published \\citep{Arora2026use}",
                    "card_refit": "48-cell card, training-DP thresholds", "savant_tuned": "Savant's form, run cost tuned on training folds",
                    "retune_intercept": "Teams' rule, intercept re-tuned", "retune_slope": "Teams' rule, slope on $\\log g$ re-tuned",
                    "retune_both": "Teams' rule, intercept and slope re-tuned"})
        order = ["stake_0.008", "stake_frozen", "stake_0.01"] + (["runs_dp", "card_pub", "card_refit", "savant_tuned"] if ve else []) \
            + ["savant"] + (["retune_intercept", "retune_slope", "retune_both"] if ve else []) + ["obs_rule"]
        for r_ in order:
            cells = []
            for vv, bb in (((ve if r_ not in v["rules"] else v), b), ((vme if (vm and r_ not in vm["rules"]) else vm), bm)):
                q = vv["rules"].get(r_) if vv else None
                if q is None:
                    cells += ["---", "---"]; continue
                c_ = ""
                if bb and r_ in bb["gap_closed"]:
                    c_ = " " + ci(bb["gap_closed"][r_]["ci"], lambda x: f"{100 * x:.0f}")
                cells += [neg(f"{100 * q['gap_closed']:.0f}") + c_, f2(q["n"])]
            L.append(f"{lab[r_]} & " + " & ".join(cells) + " \\\\")
        L.append(f"Observed 2026 & 0 & {f2(v['n_observed'])} & 0 & {f2(vm['n_observed']) if vm else '---'} \\\\")
        L.append(f"Benchmark & 100 & {f2(v['n_bench'])} & 100 & {f2(vm['n_bench']) if vm else '---'} \\\\")
        L += ["\\bottomrule", "\\end{tabular}"]
        open(f"{OUT}/tab_rules.tex", "w").write("\n".join(L) + "\n")
    rf = get("extra_checks.json") or {}
    if "dp_monotonicity" in rf:
        d = rf["dp_monotonicity"]; av = lambda k: np.mean([d[h][k] for h in ("home", "away")])
        m("DPRiseEntPct", f"{100 * av('share_entries_rise_in_k'):.0f}"); m("DPRiseOppPct", f"{100 * av('share_opportunities_rise_in_k'):.0f}")
        m("DPConcEntPct", f1(100 * av("share_entries_delta1_lt_delta2"))); m("DPConcOppPct", f1(100 * av("share_opportunities_delta1_lt_delta2")))
    if "tail" in rf:
        for sd, nm in (("off", "Batter"), ("def", "Catcher")):
            t = rf["tail"][sd]; bb = {x["bin"]: x for x in t["by_bin"]}
            m(f"TailThree{nm}", thou(t["n_beyond_3"])); m(f"TailFour{nm}", thou(t["n_beyond_4"]))
            m(f"TailRateThreeFour{nm}", f"{100 * bb['(3, 4]']['rate']:.0f}"); m(f"TailRateFourFive{nm}", f"{100 * bb['(4, 5]']['rate']:.0f}")
            m(f"WrongWithinTwo{nm}", f"{100 * t['share_wrong_within_2']:.0f}")
    if "lapse_far_misses" in rf:
        for sd, nm in (("off", "Batter"), ("def", "Catcher")):
            L_ = rf["lapse_far_misses"][sd]; c, se = L_["logit"]["coef"], L_["logit"]["se"]
            m(f"FarN{nm}", thou(L_["n"])); m(f"FarRate{nm}", f"{100 * L_['challenge_rate']:.0f}")
            m(f"FarLogG{nm}", f2(c["log_g"])); m(f"FarLogGSE{nm}", f2(se["log_g"]))
            m(f"FarOneLeft{nm}", f2(c["one_left"])); m(f"FarOneLeftSE{nm}", f2(se["one_left"]))
            m(f"FarRateOne{nm}", f"{100 * L_['rate_one_left']:.0f}"); m(f"FarRateTwo{nm}", f"{100 * L_['rate_two_left']:.0f}")
    if "edge_checks" in rf:
        e = rf["edge_checks"]; sdf = e["standardised_differences"]["final_call"]
        m("EdgeStrikesSD", f2(abs(sdf["strikes"]))); m("EdgeOtherSDMax", f2(max(abs(v) for k, v in sdf.items() if k not in ("strikes",))))
        o = e["original_call_slope"]; m("EdgeOrigSlope", f2(o["slope_mean"]))
        m("EdgeOrigSlopeCI", ci([o["slope_mean"] - 1.96 * o["se_typical"], o["slope_mean"] + 1.96 * o["se_typical"]]))
    sbias = get("scoring_bias.json")
    if sbias:
        m("ScoreRatioAll", f3(sbias["mean"]["ratio_all"])); m("ScoreRatioCh", f3(sbias["mean"]["ratio_challenged"]))
        m("ScoreDecLo", f3(min(sbias["ratio_by_decile"]))); m("ScoreDecHi", f3(max(sbias["ratio_by_decile"])))
    svj = get("stakes_validation.json")
    if svj:
        cs = [x["corr"] for x in svj["statcast"]["by_leverage_decile"]]
        m("SVDecCorrLo", f2(min(cs))); m("SVDecCorrHi", f2(max(cs)))
    for key, nm in ((PRIMARY, "Gap"), (COMPARATOR, "GapMain")):
        v = pt(key); b = get(f"pcf_boot_{key}.json")
        if v and b:
            alg = v["lost_sd_over_splits"] ** 2 + v["lost_mc_se_within"] ** 2
            m(f"{nm}BootSDAlg", f3(np.sqrt(max(b["lost"]["sd"] ** 2 - alg, 0)))); m(f"{nm}AlgPct", f"{100 * alg / b['lost']['sd'] ** 2:.0f}")
            m(f"{nm}SimRuleGap", f2(v["bench"] - v["rules"]["obs_rule"]["wins"]))
            m(f"{nm}FitRuleCI", ci(b["obs_rule_gap"]["ci"], lambda x: f"{100 * x:.0f}"))
            if b.get("freq_matched") and b["freq_matched"].get("point") is not None:
                m(f"{nm}FMGapPct", f"{100 * b['freq_matched']['point']:.0f}"); m(f"{nm}FMGapCI", ci(b["freq_matched"]["ci"], lambda x: f"{100 * x:.0f}"))
            if b.get("sim_rule_gap"):
                m(f"{nm}SimRuleGapCI", ci(b["sim_rule_gap"]["ci"]))
    for key, nm in (("main_lapse", "Lapse"), ("main_time_lapse", "TimeLapse")):
        v = pt(key)
        if v:
            m(f"{nm}SplitsNotOK", str(sum(not r["ok"] for r in v["rows"]))); m(f"{nm}Splits", str(len(v["rows"])))
    hs = get("honest-scoring-check.json")
    if hs:
        m("SDistinctShare", f"{100 * hs['distinct_share_n2425']:.1f}\\%")
    # HONZA-134 (Fable / Opus round 1): one table of every estimator and reading of the index, both models
    h_ = get("headline.json") or {}; fs = h_.get("full_sample", {}); co = h_.get("co_primary", {}); cvj = get("coverage_pcf.json") or {}
    cr = cvj.get("results", {})
    def cell(x):
        return f2(x) if x is not None else "---"
    vP, vC = pt(PRIMARY), pt(COMPARATOR)
    GP = {(g["rho"], g["lam"]): g["lost"] for g in vP["grid"]} if vP else {}
    GC = {(g["rho"], g["lam"]): g["lost"] for g in vC["grid"]} if vC else {}
    cfC = (get("crossfit.json") or {}).get(COMPARATOR, {})
    learn = lambda k: (cr[k]["gap"] - cr[k]["gapK"]) if k in cr and cr[k].get("gapK") is not None else None
    rows = [("In sample: everything fitted and scored on all games", fs.get(COMPARATOR, {}).get("lost"), fs.get(PRIMARY, {}).get("lost")),
            ("Stakes cross-fitted only", cfC.get("lost"), co.get("lost")),
            ("\\textbf{Full policy cross-fit (the index)}", vC["lost"] if vC else None, vP["lost"] if vP else None),
            ("\\quad against the teams' fitted rule, simulated", (vC["bench"] - vC["rules"]["obs_rule"]["wins"]) if vC else None,
             (vP["bench"] - vP["rules"]["obs_rule"]["wins"]) if vP else None),
            ("\\quad every lapse removable ($\\ell=1$)", None, GP.get((1.0, 1.0))),
            ("\\quad half the noise variance removable ($\\rho=0.5$)", GC.get((0.5, 0.0)), GP.get((0.5, 0.0))),
            ("Oracle minus learned benchmark (coverage study)", learn("kappa_sr_region_g|kappa_sr_region_g"), learn("re_lapse_gh11|re_lapse_gh11"))]
    L = ["\\begin{tabular}{p{7.4cm}cc}", "\\toprule", "Estimator or reading & Comparator & Primary \\\\", "\\midrule"]
    L += [f"{a} & {cell(b)} & {cell(c)} \\\\" for a, b, c in rows]
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"{OUT}/tab_recon.tex", "w").write("\n".join(L) + "\n")
    cv = get("coverage_pcf.json")
    if cv:
        for key, nm in (("kappa_sr_region_g|kappa_sr_region_g", "CPMain"), ("main_lapse|kappa_sr_region_g", "CPLapse"),
                        ("re_lapse_gh11|kappa_sr_region_g", "CPRE"), ("re_lapse_gh11|re_lapse_gh11", "CPPrim"),
                        ("re_gh11|re_lapse_gh11", "CPMis")):
            c = cv["results"].get(key)
            if not c:
                continue
            m(f"{nm}S", str(c["S"])); m(f"{nm}B", str(c["B"])); m(f"{nm}Gap", f2(c["gap"]))
            for it, inm in (("percentile", "Pct"), ("basic", "Basic"), ("normal", "Normal")):
                m(f"{nm}{inm}", f"{100 * c['coverage'][it]['rate']:.0f}")
            m(f"{nm}PctCI", ci(c["coverage"]["percentile"]["ci"], lambda x: f"{100 * x:.0f}"))
            m(f"{nm}Bias", f3(c["point_bias"])); m(f"{nm}BiasSE", f3(c["point_bias_se"]))
            m(f"{nm}MissLow", str(c["coverage"]["percentile"]["miss_low"])); m(f"{nm}MissHigh", str(c["coverage"]["percentile"]["miss_high"]))
            m(f"{nm}PointSD", f3(c["point_sd_across_seasons"])); m(f"{nm}BootSD", f3(c["boot_sd_mean"]))
            if c.get("gapK") is not None:
                m(f"{nm}GapK", f2(c["gapK"])); m(f"{nm}BiasK", f3(c["point_bias_K"])); m(f"{nm}Learn", f2(c["gap"] - c["gapK"]))
                for it, inm in (("percentile", "Pct"), ("basic", "Basic"), ("normal", "Normal")):
                    m(f"{nm}K{inm}", f"{100 * c['coverageK'][it]['rate']:.0f}")
                m(f"{nm}KPctCI", ci(c["coverageK"]["percentile"]["ci"], lambda x: f"{100 * x:.0f}"))


def perception(pv, h):
    MAIN = "kappa_sr_region_g"                 # HONZA-122: BIC-best homogeneous model (src/headline.py MAIN)
    base, main = pv["normal"], pv[MAIN]
    m("SrBatter", f2(main["params"]["off_sr"])); m("SrCatcher", f2(main["params"]["def_sr"]))
    m("SrBaseBatter", f2(base["params"]["off_sr"])); m("SrBaseCatcher", f2(base["params"]["def_sr"]))
    # HONZA-114: sigma_r intervals from the B = 400 refits of headline_boot.json (the paper's bootstrap), not the older
    # B = 120 run of final_ci.py. Main spec theta: [logit pi | off: beta 6, log sr, kreg 3, sr region 3 | def: same], so
    # the bottom-edge log sigma_r of the batter is theta[7] and of the catcher theta[7 + 13]
    import numpy as np
    # HONZA-132: sigma_r intervals (bottom edge) from the joint bootstrap's refits
    srs = np.array([r["sr"] for r in J("joint_boot.json")["rows"] if r.get("ok")])
    for side, nm, j in (("off", "Batter", 0), ("def", "Catcher", 1)):
        lo, hi = np.percentile(srs[:, j], [2.5, 97.5]); pt_ = main["params"][f"{side}_sr"]
        m(f"Sr{nm}BootCI", ci([lo, hi])); m(f"Sr{nm}HalfPct", f1(100 * (hi - lo) / 2 / pt_))
    p = main["params"]
    # HONZA-88: the main spec has a deterministic truth; the boundary offsets are those of the smooth truth model
    # (sm_region) refitted on the corrected geometry, where they collapse to ~0
    sr_ = pv["sm_region"]["params"]
    m("MuBottom", f2(-sr_["mu_bottom"])); m("MuTop", f2(sr_["mu_top"]))
    m("SmMax", f3(max(0.001, max(sr_[f"sm_{r}"] for r in ("bottom", "top", "arm", "glove")))))   # "at most": never print 0.000
    m("MuMaxAbs", f3(max(abs(sr_[f"mu_{r}"]) for r in ("bottom", "top", "arm", "glove"))))
    m("LRSmooth", f1(2 * (base["nll"] - pv["sm_region"]["nll"]))); m("BICSmoothMinusDet", thou(pv["sm_region"]["bic"] - base["bic"]))
    m("ContamPct", f2(100 * p["contam_pi"]))
    m("DLLEdge", thou(pv["sm_edge"]["nll"] - base["nll"]))
    m("DLLMain", thou(base["nll"] - main["nll"])); m("DLLRE", thou(base["nll"] - pv["re"]["nll"]))
    m("BICMain", thou(main["bic"])); m("BICRE", thou(pv["re"]["bic"]))
    rej = [pv[k]["nll"] - base["nll"] for k in ("logistic", "t10", "t5", "t3")]
    m("DLLRejLo", thou(min(rej))); m("DLLRejHi", thou(max(rej)))
    # HONZA-114: the same noise-law comparison on the main specification, and a lapse rate (the challenge curve tops out
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
    # HONZA-114: on the data, how far the homogeneous main fit's batter sigma_r sits above the random-effects fit's
    m("HetDataBiasBatter", f1(100 * (p["off_sr"] / re_["off_sr"] - 1)))
    m("KappaTopBatter", f1(p["off_kappa_region_shift"]["top"]))
    m("SrTopCatcherPct", f"{100 * (p['def_sr_region_mult']['top'] - 1):.0f}")
    # HONZA-132: the main-text model table (tab_variants) is written by honza132()
    vw = J("variant_wins.json")["calibration_test"]
    m("SlopeCalBatter", f2(vw["off"]["calibrated"]["slope_log_g"])); m("SlopeCalCatcher", f2(vw["def"]["calibrated"]["slope_log_g"]))
    os_ = J("opt_slope.json")["opt"]          # HONZA-114: the optimum's own thresholds through the same regression
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
    # HONZA-88 kernel check of the deterministic-truth identification (wolfram/identification_det.wl)
    dk = J("identification-det-check.json")
    m("DetRank", str(dk["D2_rank_batter"])); m("DetVersion", short_version(dk["wolframVersion"]))
    bias = [float(x) for x in dk["D4_rel_bias_pct_c03"].replace("{", "").replace("}", "").split(",")]
    m("DetCoordBiasCatcher", f1(bias[1])); m("DetCoordBiasBatter", f1(bias[3]))
    # sigma_r multiplier at which wins lost reaches 0.25 (linear interpolation of sr_breakeven.json).
    # HONZA-88: sr_breakeven.json is evaluated on the score-aware optimum (opt_score), like the headline.
    br = sorted((float(k), v["lost"]) for k, v in J("sr_breakeven.json").items())
    be = next(x0 + (x1 - x0) * (l0 - 0.25) / (l0 - l1) for (x0, l0), (x1, l1) in zip(br, br[1:]) if l0 >= 0.25 > l1)
    m("BreakEvenPct", f"{100 * (be - 1):.0f}")
    # HONZA-91: what the break-even sigma_r costs in the profile log-likelihood (identification.json, baseline model),
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
    # HONZA-114: how much noisier than MLB 2026 in the same model, range over the two sides (HONZA-122: the main
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
    """Bar-2 threshold gaps and the sqrt-n fit on the score-aware DP (HONZA-77 refresh), if present."""
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
    """HONZA-87 postseason out-of-sample check (results/postseason_preview.json) -> Post* macros, tab_post.tex and
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
    lab = {"opt_score": "Transported benchmark (regular-season DP)", "stake_0.008": "One-line rule, $\\delta=0.008$",
           "obs_rule": "Fitted behavioural rule (simulated)", "observed": "Observed (realised)"}
    L = ["\\begin{tabular}{lcc}", "\\toprule", "Policy & Regular season & Postseason \\\\", "\\midrule"]
    for p, nm in lab.items():
        a, b = dict(w["rs"][p]), w["po"][p]
        if p == "observed":                  # exact count ratio: the stored 2.175 (3 dp) would print as 2.17
            a["challenges_per_team_game"] = rs["challenges"] / (2 * rs["games"])
        L.append(f"{nm} & {f4(a['wp_per_team_game'])} ({f2(a['challenges_per_team_game'])}) & "
                 f"{f4(b['wp_per_team_game'])} ({f2(b['challenges_per_team_game'])}) \\\\")
    L.append(f"Benchmark $-$ observed & {f4(w['rs']['opt_minus_observed']['wp_per_team_game'])} & "
             f"{f4(ow['wp_per_team_game'])} {ci(ow['ci95'], f4)} \\\\")
    L.append(f"Share of the benchmark captured (\\%) & {pct(w['rs']['opt_minus_observed']['share_of_opt_captured'], 0)} & "
             f"{pct(ow['share_of_opt_captured'], 0)} {ci(ow['share_ci95'], lambda x: pct(x, 0))} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"{OUT}/tab_s_postvalue.tex", "w").write("\n".join(L) + "\n")


def causal_gap():
    """HONZA-100 simulator-versus-engine decomposition (results/causal_gap.json) -> SGap* macros, tab_s_gap*.tex."""
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
    """HONZA-106 (AoAS referee report of HONZA-103): bootstrap interval diagnostics, the nonparametric check of the
    challenge curve (results/isotonic_check.json) and the behavioural-rule gap (results/behaviour_gap.json)."""
    import random
    # HONZA-132: interval diagnostics of the joint bootstrap (the engine-fixed bootstrap is no longer run)
    lost = [r["lost"] for r in J("joint_boot.json")["rows"] if r.get("ok")]
    n = len(lost)
    q = lambda xs, a: sorted(xs)[min(len(xs) - 1, max(0, int(round(a * (len(xs) - 1)))))]
    rng = random.Random(106); ends = []
    for _ in range(2000):                     # Monte Carlo error of the percentile endpoints: resample the B replicates
        xs = [lost[rng.randrange(n)] for _ in range(n)]
        ends.append((q(xs, 0.025), q(xs, 0.975)))
    sd = lambda v: (sum((x - sum(v) / len(v)) ** 2 for x in v) / (len(v) - 1)) ** 0.5
    m("CIEndSE", f3(max(sd([e[0] for e in ends]), sd([e[1] for e in ends]))))
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


def honza122():
    """HONZA-122 (five-referee review): sample flow, label agreement on all pitches, the clairvoyant ceiling, the tuned
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
                ("without a Statcast location", -sf["no_location"], -sf["no_location_challenges"] if "no_location_challenges" in sf else None),
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
    if "clairvoyant" in P:                     # HONZA-132: WClair, SavantTuned* now from headline()
        m("ClairGap", f2(P["clairvoyant"]["wins"] - P["bench"]["wins"]))
    o = get("oos.json")
    if o and "frozen_share" in o.get("wins_lost_test_train_params", {}):
        t = o["wins_lost_test_train_params"]
        m("FrozenDelta", f"{t['frozen_delta']:g}"); m("FrozenShare", pct(t["frozen_share"])); m("FrozenGapClosed", f"{100 * t['frozen_gap_closed']:.0f}")
    fz = o.get("frozen_crossfit", {}).get("with_train_engine")      # HONZA-130 (Opus, Sol, Astra): everything frozen
    if fz:
        m("FrozenAllDelta", f"{fz['frozen_delta']:g}"); m("FrozenDPShare", pct(fz["frozen_dp_share_of_retro"]))
        m("FrozenRuleShareRetro", pct(fz["frozen_rule_share_of_retro"])); m("FrozenRuleShareDP", pct(fz["frozen_rule_share_of_frozen_dp"]))
        m("FrozenLost", f2(fz["wins_lost_vs_frozen_dp"])); m("FrozenRuleGapClosed", f"{100 * fz['frozen_rule_gap_closed_vs_frozen_dp']:.0f}")
        m("FrozenTrainGames", thou(fz["train_games"])); m("FrozenTestGames", thou(fz["test_games"]))
        m("FrozenR", str(fz["R"]))
        for key, nm in (("no_engine", "NoEng"), ("no_engine_world_testfit", "WorldTest")):
            z = o["frozen_crossfit"].get(key)
            if z:
                m(f"FrozenLost{nm}", f2(z["wins_lost_vs_frozen_dp"])); m(f"FrozenDPShare{nm}", pct(z["frozen_dp_share_of_retro"]))
                m(f"FrozenRuleShareRetro{nm}", pct(z["frozen_rule_share_of_retro"]))
        m("OOSTestLost", f2(t["opt_minus_observed"])); m("OOSTestLostOwn", f2(o["wins_lost_test_test_params"]["opt_minus_observed"]))
        m("OOSTrainLost", f2(o["wins_lost_train_train_params"]["opt_minus_observed"]))   # HONZA-130 (Opus, Sonnet)
    # HONZA-132: the joint and block bootstrap macros come from headline()
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


def referee130():
    """HONZA-130 (five-referee review): rule-class decomposition, piecewise slopes, cross-fitted stakes, engine
    shrinkage, months, moot calls, random-effects quadrature and calibration, state-dependent lapse."""
    if not os.path.exists(os.path.join(R, "referee130.json")):
        return
    d = J("referee130.json")
    # HONZA-132: the decomposition is the cross-fitted point of results/mechanism_boot.json when present
    mbp = J("mechanism_boot.json").get("point") if os.path.exists(os.path.join(R, "mechanism_boot.json")) else None
    dec = mbp["full"]["decomp"] if mbp and mbp.get("full") else d["decomp"]
    D = dec["rows"]; o = lambda k: D[k]["out_of_sample"]
    for k, nm in (("fitted_rule", "Fit"), ("intercept_only", "Int"), ("stake_linear_only", "Stake"), ("both_linear", "Both"),
                  ("stake_scale_only", "StakeScale"), ("both_scale", "BothScale")):
        m(f"Dec{nm}Share", pct(o(k)["share_opt"]))
        if not (mbp and nm in ("Int", "Stake", "Both", "StakeScale")):        # those come from honza132()
            m(f"Dec{nm}Gap", f"{100 * o(k)['gap_closed']:.0f}"); m(f"Dec{nm}N", f2(o(k)["n"]))
    m("DecR", str(dec["R"])); m("DecRTune", str(dec["R_tune"]))
    for side, nm in (("off", "Batter"), ("def", "Catcher")):
        S = d["slopes"]["sides"][side]
        m(f"SlopeHighTeam{nm}", f2(S["high"]["teams_logit_full"])); m(f"SlopeHighOpt{nm}", f2(S["high"]["opt_logit_full"]))
        m(f"SlopeLowTeam{nm}", f2(S["low"]["teams_logit_full"])); m(f"SlopeLowOpt{nm}", f2(S["low"]["opt_logit_full"]))
        m(f"KappaSlopeHighTeam{nm}", f2(S["high"]["teams_kappa_ols_full"])); m(f"KappaSlopeHighOpt{nm}", f2(S["high"]["opt_signal_threshold_ols_full"]))
    c = d["crossfit_stakes"]
    m("CurseBias", f2(c["bias_mean"])); m("CurseBiasSD", f2(c["bias_sd_over_splits"]))
    m("CurseLostSame", f2(c["lost_same_mean"])); m("CurseLostCross", f2(c["lost_cross_mean"]))
    E = d["wp_shrink"]["engines"]
    ls = [E[k]["lost"] for k in E]
    m("ShrinkLostLo", f2(min(ls))); m("ShrinkLostHi", f2(max(ls)))
    M = d["months"]
    m("MonthLostLo", f2(min(v["lost_per_team_season"] for v in M["months"].values())))
    m("MonthLostHi", f2(max(v["lost_per_team_season"] for v in M["months"].values())))
    m("MonthTrend", f2(M["linear_trend_per_month"]))
    m("PreASBLost", f2(M["first_half_pre_asb"]["lost_per_team_season"])); m("PostASBLost", f2(M["second_half_post_asb"]["lost_per_team_season"]))
    mo = d["moot"]["moot_same_pitch"]
    m("MootN", str(mo["n"])); m("MootStakePct", f"{100 * mo['share_of_total_stake']:.2f}")
    g9 = d["re_quad"]["gh9"]
    m("REGHNineKappaSD", f2(g9["off_re"]["sd_kappa"])); m("REGHNineSr", f2(g9["off_sr"]))
    ct = d["re_quad"]["calibration_table"]["re"]
    m("RECalDefPct", f"{100 * (1 - ct['def_ch_pred'] / ct['def_ch_obs']):.1f}")
    L = d["lapse_state"]["main_lapse_low"]
    m("LapseLowBatter", f"{100 * L['off']['lapse_low_stake']:.0f}"); m("LapseHighBatter", f"{100 * L['off']['lapse_high_stake']:.0f}")
    m("LapseLowZ", f1(L["off"]["z"])); m("LapseStateDBIC", f"{L['delta_bic_vs_main_lapse']:.0f}")
    pc = d["pclip"]
    m("PClipOne", f"{pc['counts_per_draw']['p1']:.0f}"); m("PClipBelowPi", thou(round(pc["counts_per_draw"]["p_lt_pi"])))
    # tables for the supplement's further checks
    lab = {"fitted_rule": "Teams' fitted rule $s>\\kappa(x)$", "intercept_only": "Intercept re-tuned (frequency)",
           "stake_linear_only": "Slope on $\\log g$ re-tuned (targeting)", "both_linear": "Intercept and slope re-tuned",
           "stake_scale_only": "Stake terms scaled", "both_scale": "Intercept and stake terms scaled"}
    T = ["\\begin{tabular}{lcccc}", "\\toprule",
         "Rule & Wins & Share of optimum (\\%) & Gap closed (\\%) & Challenges per team-game \\\\", "\\midrule"]
    for k in ("fitted_rule", "intercept_only", "stake_linear_only", "both_linear", "stake_scale_only", "both_scale"):
        q = o(k)
        T.append(f"{lab[k]} & {f2(q['wins'])} & {pct(q['share_opt'])} & {100 * q['gap_closed']:.0f} & {f2(q['n'])} \\\\")
    T += ["\\bottomrule", "\\end{tabular}"]
    open(f"{OUT}/tab_s_decomp.tex", "w").write("\n".join(T) + "\n")
    T = ["\\begin{tabular}{lcccc}", "\\toprule", "Period & Games & Wins lost (SE) & Observed share of optimum (\\%) & One-line rule share (\\%) \\\\", "\\midrule"]
    rows = list(M["months"].items()) + [("Before the All-Star break", M["first_half_pre_asb"]), ("After it", M["second_half_post_asb"]), ("Season", M["season"])]
    for k, v in rows:
        T.append(f"{k} & {thou(v['games'])} & {f2(v['lost_per_team_season'])} ({f2(v['boot_se'])}) & {pct(v['observed_share_opt'])} & {pct(v['rule_share_opt'])} \\\\")
    T += ["\\bottomrule", "\\end{tabular}"]
    open(f"{OUT}/tab_s_months.tex", "w").write("\n".join(T) + "\n")
    lab = {"main_pooled_w20": "Pooled over counts, weight 20 (main)", "pooled_w0": "No shrinkage", "pooled_w100": "Pooled over counts, weight 100",
           "zero_zero_w20": "Toward the 0--0 table, weight 20"}
    T = ["\\begin{tabular}{lcccc}", "\\toprule", "Engine & Wins lost & One-line rule share (\\%) & Non-monotone pitches & Non-monotone cells \\\\", "\\midrule"]
    for k, v in E.items():
        nm = v["nonmonotone"]
        T.append(f"{lab.get(k, k)} & {f2(v['lost'])} & {pct(v['share_0008'])} & {nm.get('rows', nm.get('n_rows', '')) if isinstance(nm, dict) else nm} & {nm.get('distinct_state_cells', '') if isinstance(nm, dict) else ''} \\\\")
    T += ["\\bottomrule", "\\end{tabular}"]
    open(f"{OUT}/tab_s_shrink.tex", "w").write("\n".join(T) + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    postseason()
    causal_gap()
    geometry()
    h, P = headline()
    pv = J("perception_variants.json")
    referee130()
    policy_table(P)
    perception(pv, h)
    identification()
    theory()
    drs()
    data_counts()
    thresholds()
    review_fixes()
    honza122()
    honza132()
    honza134()                                     # HONZA-134: writes tab_variants / tab_pid / tab_rules over honza132's
    carried()
    import supplement_numbers                      # HONZA-92: S-macros and tab_s_*.tex for supplement.tex
    supplement_numbers.run(m, R, OUT)
    # HONZA-132: every macro the manuscripts use that no result has defined yet prints as pending (make check fails on it)
    import re as _re
    defined = set(macros)
    for tex in ("paper/main.tex", "paper/supplement.tex"):
        used = set(_re.findall(r"\\([A-Z][A-Za-z]+)", open(tex).read()))
        for name in sorted(used - defined - LATEX_CAPS):
            m(name, PENDING); defined.add(name)
    L =["% generated by paper/make_numbers.py from results/ -- do not edit"]
    L += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
    open(f"{OUT}/numbers.tex", "w").write("\n".join(L) + "\n")
    print(f"{len(macros)} macros; {sum(PENDING in v for v in macros.values())} pending")
    assert not DUPS, f"duplicate macros: {sorted(set(DUPS))}"


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
