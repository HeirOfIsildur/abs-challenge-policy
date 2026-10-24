"""HONZA-92: numbers and tables of the Supplementary Material, read from results/ (including the
HONZA-89). Called by make_numbers.py, which owns the macro table: every macro here starts with S and every table is
written to paper/generated/tab_s_*.tex. Nothing is typed by hand."""
import json, math, os, re
from fractions import Fraction


def run(m, R, OUT):
    J = lambda f: json.load(open(os.path.join(R, f)))
    A = lambda f: json.load(open(os.path.join("results", f)))
    from make_numbers import neg                    # HONZA-106: real minus signs, no "-0.00"
    f1, f2, f3 = (lambda x: neg(f"{x:.1f}")), (lambda x: neg(f"{x:.2f}")), (lambda x: neg(f"{x:.3f}"))
    f4 = lambda x: neg(f"{x:.4f}")
    pct = lambda x, d=1: f"{100 * x:.{d}f}"
    ci = lambda c, f=f2: neg(f"[{f(c[0])}, {f(c[1])}]")
    thou = lambda n: f"{int(round(n)):,}".replace(",", "{,}")

    def sci(x):
        """|x| as a LaTeX power of ten: 4.3e-29 -> $4\\times10^{-29}$; exact zero -> $0$."""
        x = abs(float(x))
        if x == 0:
            return "$0$"
        e = math.floor(math.log10(x)); c = x / 10 ** e
        return f"${c:.0f}\\times10^{{{e}}}$"

    def table(name, cols, header, rows):
        L = [f"\\begin{{tabular}}{{{cols}}}", "\\toprule"] + header + ["\\midrule"] + rows + ["\\bottomrule", "\\end{tabular}"]
        open(f"{OUT}/tab_s_{name}.tex", "w").write("\n".join(L) + "\n")

    def wl_num(s):
        """a Wolfram InputForm number string (precision marks, *^ exponents) -> float."""
        return float(re.sub(r"`[0-9.]*", "", str(s)).replace("*^", "e"))

    # ------------------------------------------------------------------ S1 proofs (results/proofs.json)
    p = J("proofs.json")
    v = p["wolframVersion"]
    m("SWolframShort", ".".join(v.split(".")[:2]))
    cd = [Fraction(x.strip()) for x in p["T1d_c_dependent_law_Delta(1),Delta(2) (p=1/10 at c=1, p=9/10 at c=2, g=1)"].strip("{}").split(",")]
    m("SCdepOne", f3(float(cd[0]))); m("SCdepTwo", f2(float(cd[1])))
    laws = p["T1c_exact_random_laws"]
    laws_ok = all(all(vv is True for kk, vv in x.items() if kk not in ("nLaws", "arithmetic")) for x in laws.values())
    m("SRandomLaws", str(sum(x["nLaws"] for x in laws.values())))
    last = lambda key: p[key][-1][1]
    m("SAlphaOne", f4(last("T2d_alpha1_Delta/sqrt(2 K1 t)"))); m("SAlphaTwo", f3(last("T2d_alpha2_Delta/(3 K2 t)^(1/3)")))
    m("SAtom", f4(last("T2d_atom_Delta/t"))); m("SAtomLimit", f4(float(Fraction(p["T2d_atom_limit E[g; p=1]"]))))
    m("STLast", f"10^{{{round(math.log10(p['T2d_alpha1_Delta/sqrt(2 K1 t)'][-1][0]))}}}")
    m("STimeVarying", f4(last("T2h_time_varying_Delta^2/sum(g_s^2)")))
    m("SKc", ", ".join(f3(x) for x in p["T2e_k_c c=1..6"]))
    m("SUniformKc", ", ".join(f4(x) for x in p["T2e_uniform_Delta_t(c)/sqrt(t) c=1..4"][-1][1]))
    m("SConsBound", f"{p['T2f_uniform_cons_Delta at t=1e6 (sup pg = 1)']:.6f}")
    m("SStylBestDelta", f4(p["T3b_best_delta"])); m("SStylLoss", f1(100 * p["T3b_rel_loss_best_delta"]))
    m("SStylBound", f1(100 * p["T3b_quadratic_bound {abs, rel}"][1]))
    m("SStylCrude", f"{100 * p['T3b_crude_bound M(D-de)^2/(8de) {abs, rel}'][1]:.0f}")
    ri = p["T3b_regret_identity {Vstar-Vdelta, sum occupation-weighted gaps}"]
    m("SStylIdentityGap", sci(ri[0] - ri[1]))
    m("SStylDeltaMax", f3(p["T3b_Delta_range_on_path"][1]))
    lv = dict((round(d, 4), l) for d, l in p["T3b_rel_loss_vs_delta"])
    m("SStylLossVsDelta", ", ".join(f"{pct(lv[d], 0)}\\,\\% at $\\delta={d}$" for d in (0.01, 0.02, 0.03, 0.05) if d in lv))
    rows = [f"{r['t']:,} & {f2(r['cons1'])} & {f2(r['NBA'])} & {f2(r['ret1'])} & {f2(r['cons2'])} & {f2(r['ret2'])} \\\\".replace(",", "{,}", 1)
            for r in p["T4_uniform_p_g1"] if r["t"] in (1, 2, 5, 10, 50, 100, 1000, 10000)]
    m("SRuleOrderAll", "every" if all(r["ordering cons1<=NBA<=min(ret1,cons2), cons2<=ret2"] for r in p["T4_uniform_p_g1"]) else "\\textbf{NOT every}")
    table("rules", "rrrrrr", ["$t$ & consumption, 1 & NBA cap & retention, 1 & consumption, 2 & retention, 2 \\\\"], rows)
    t4 = p["T4_T3b_law_T60"]
    for k, nm in (("cons1", "ConsOne"), ("NBA", "NBA"), ("ret1", "RetOne"), ("cons2", "ConsTwo"), ("ret2", "RetTwo")):
        m(f"STSixty{nm}", f3(t4[k]))
    m("STSixtyGain", f"{100 * (t4['ret2'] / t4['cons2'] - 1):.0f}")

    # ------------------------------------------------------------------ S2 symbolic verification
    chk = {"check": J("check.json"), "headline": J("headline-check.json"), "ident": J("identification-check.json"),
           "det": J("identification-det-check.json"), "sup": J("supplement-check.json"), "drs": J("drs/check.json"), "drsaudit": J("drs/audit-check.json")}
    FAIL = "\\textbf{fails}"
    allt = lambda d, ks: "True" if all(str(d[k]) == "True" for k in ks) else FAIL
    zero = lambda x: "$0$ (exact)" if str(x) == "0" else FAIL
    small = lambda x: f"max.\\ error {sci(wl_num(x))}"
    H, C, D, I, Q = chk["headline"], chk["check"], chk["det"], chk["ident"], chk["drs"]
    m("SCtrExDelta", f'{float(chk["sup"]["S6a_Delta_2000"]):.3f}')
    S = [  # (script, what is checked, outcome)
        ("check.wl", "Cutoff: challenge iff $p\\ge\\Delta/(g+\\Delta)$ (no counterexample)", "True" if C["A_equiv_counterexample"] == "False" else FAIL),
        ("check.wl", "Cutoff increasing in $\\Delta$, decreasing in $g$", "True" if C["B_signs"] == [True, True] else FAIL),
        ("identification\\_det.wl", "D1: deterministic truth is the $\\sigma_m\\to0$ limit of the smooth truth model", small(D["D1_limit_maxabs"])),
        ("identification\\_det.wl", "D2: two-point Jacobian determinant $\\phi_1\\phi_2(w_1-w_2)/\\sigma_r^3$", zero(D["D2_det_minus_closed_form"])),
        ("identification\\_det.wl", "D2: rank of the challenge-curve Jacobian on the data grid, batter\\,/\\,catcher",
         f"{D['D2_rank_batter']}\\,/\\,{D['D2_rank_catcher']}"),
        ("identification\\_det.wl", "D3: cross information of $\\pi$ with $(\\sigma_r,\\kappa)$",
         zero(D["D3_cross_pi_sr"]) if D["D3_cross_pi_sr"] == D["D3_cross_pi_k"] else FAIL),
        ("identification\\_det.wl", "D4: tracking noise $c$ gives $\\Phi\\{(w-\\kappa)/(\\sigma_r^2+c^2)^{1/2}\\}$",
         small(D["D4_integral_minus_closed_form_maxabs"])),
        ("check.wl", "Gaussian convolution of tracking and perception noise (symbolic and 30-digit)",
         allt(C, ["C_conv_symbolic_density"]) + "; " + small(C["C_conv_maxabs"])),
        ("identification.wl", "I1, I3: in the superseded smooth-truth model only $(\\sigma_m^2+\\sigma_r^2)^{1/2}$ enters the challenge curve",
         zero(I["I1_sr_dPdsm_minus_sm_dPdsr"]) + f"; rank {I['I3_batter_rank_challenge_only']} of 3"),
        ("proofs.wl", "T1a: Lemma~\\ref{lem:phi} for any law (ForAll over the reals)",
         allt(p, ["T1a_phi_nonincreasing", "T1a_psi_nondecreasing", "T1a_phi_slope_in_[-1,0]"])),
        ("proofs.wl", "T1b: induction step of Theorem~\\Mref{thm:mono}",
         allt(p, ["T1b_concavity_step", "T1b_nonneg_step", "T1b_monotone_in_t_step", "T1b_concavity_step_c1"])),
        ("proofs.wl", f"T1c: both monotonicities in exact DPs on {sum(x['nLaws'] for x in laws.values())} random laws", "True" if laws_ok else FAIL),
        ("proofs.wl", "T1d: concavity fails for a law that depends on $c$", allt(p, ["T1d_concavity_fails"])),
        ("proofs.wl", "T2a: $d\\varphi(d)\\to g^2h/2$ and $d^\\alpha\\varphi(d)\\to g^{1+\\alpha}/(1+\\alpha)$",
         "True" if p["T2a_lim_d_phi_generic_density"] == "(g^2*h)/2" and p["T2a_lim_d^a_phi"] == "g^(1 + aa)/(1 + aa)" else FAIL),
        ("proofs.wl", "T2b: square-increment identity~\\eqref{eq:sqinc}", zero(p["T2b_square_increment"])),
        ("proofs.wl", "T2d: exact constants $K_1=9/8$, $K_2=83/144$",
         "True" if p["T2d_K1=lim d phi"] == p["T2d_E[h g^2]/2"] and p["T2d_K2=lim d^2 phi"] == p["T2d_E[g^3]/3"] else FAIL),
        ("proofs.wl", "T2e: $k_2=(\\sqrt5-1)/2$; SA drift decreasing with root $k_c^2$",
         allt(p, ["T2e_k2_equals_GoldenRatio-1", "T2e_drift_root_check"])),
        ("proofs.wl", "T2f: consumption step $x\\le s,\\ Y\\le s\\Rightarrow x+\\E(Y-x)^+\\le s$", allt(p, ["T2f_consumption_bound_step"])),
        ("proofs.wl", "T3a: per-state regret on both sides of $D$; crude bound",
         "True" if p["T3a_regret_closed_form_both_sides"] == "{True, True}" and p["T3a_g^2/((g+D)(g+de)^2) <= 1/(4 de)"] == "True" else FAIL),
        ("proofs.wl", "T3b: regret identity in the stylized game", f"gap {sci(ri[0] - ri[1])}"),
        ("supplement.wl", "Substitution and domination bounds in the proof of Theorem~\\ref{thm:sqrtS}", allt(chk["sup"], [k for k in chk["sup"] if k.startswith(("S1", "S2"))])),
        ("supplement.wl", "Bound (ii) of Theorem~\\ref{thm:flatS}; crude bound (iii)", allt(chk["sup"], [k for k in chk["sup"] if k.startswith(("S3", "S5"))])),
        ("supplement.wl", "Mean-value step for the rate $t^{1/(1+\\alpha)}$", allt(chk["sup"], [k for k in chk["sup"] if k.startswith("S4")])),
        ("supplement.wl", "Theorem~\\ref{thm:rules}(c): bound $\\varphi(L)\\ge g_0/2\\cdot\\Prob(\\cdot)$; counterexample without $g_0$",
         allt(chk["sup"], [k for k in chk["sup"] if k.startswith("S6") and k != "S6a_Delta_2000"])),
        ("headline.wl", "Per-state regret, both sides; $r'(D)=0$", allt(H, ["T1_regret_D_gt_delta", "T1_regret_D_lt_delta", "T1_envelope_r'(D)=0"])),
        ("headline.wl", "Causal decomposition and RB difference (Section~\\ref{sec:causal})", allt(H, ["T2_causal_decomposition", "T2_rb_difference", "T2_rb_difference_signed"])),
        ("headline.wl", "Runs rule and one-line rule are cutoff rules in $p$", allt(H, ["T3_runs_rule", "T3_stake_rule"])),
        ("drs\\_check.wl", "DRS posterior and marginal; inversion of the review rule", allt(Q, ["A_posterior", "A_marginal", "B_sstar", "B_p_increasing_in_S"])),
        ("drs\\_check.wl", "DRS orthant correlation; bivariate normal CDF of the code", allt(Q, ["C_corr"]) + "; " + small(Q["C_bvn_maxabs"])),
        ("drs\\_check.wl", "DRS review rule with a retained umpire's call", allt(Q, ["D_general"])),
        ("drs\\_audit.wl", "DRS three-way likelihood and posterior loss probability at the Test fit",
         f"max.\\ error {sci(max(abs(float(chk['drsaudit'][k])) for k in ('success_diff', 'uc_diff', 'lost_diff', 'l_diff')))}"),
    ]
    iv = J("intervention-check.json")                      # HONZA-132: the intervention model (wolfram/intervention.wl)
    S += [("intervention.wl", "Behavioural noise $\\sqrt\\rho\\,\\sigma_r\\varepsilon_1+\\sqrt{1-\\rho}\\,\\sigma_r\\varepsilon_2\\sim N(0,\\sigma_r^2)$; "
                                  "$\\partial\\Prob(\\text{challenge}\\mid w)/\\partial\\rho=0$",
           "True" if iv["convolution_is_N0sr"] == "True" and iv["dPchallenge_drho"] == "0" else "False"),
          ("intervention.wl", f"Value of the calibrated cutoff decision falls with the signal noise ({iv['grid_points']} grid points)",
           "True" if iv["value_decreasing_in_noise_on_grid"] else "False")]
    table("checks", "lp{5.9cm}l", ["Script & Claim & Outcome \\\\"], [f"\\texttt{{{a}}} & {b} & {c} \\\\" for a, b, c in S])
    m("SNChecks", str(len(S))); m("SChecksFailed", str(sum(FAIL in c for _, _, c in S)))

    # ------------------------------------------------------------------ S3 geometry and data audit
    g = J("geometry_audit.json")
    lab = {"statcast_2024_one_day": "Statcast 2024 (one day)", "statcast_2025_one_day": "Statcast 2025 (one day)",
           "statcast_2026": "Statcast 2026 regular season", "statcast_post2026": "Statcast 2026 postseason"}
    rows = []
    for k, nm in lab.items():
        q = g["plane"][k]
        rows.append(f"{nm} & {thou(q['n'])} & ${q['front']['dz_mean']:+.3f}$ ({f3(q['front']['dz_sd'])}) & "
                    f"${q['mid']['dz_mean']:+.3f}$ ({f3(q['mid']['dz_sd'])}) & {f2(q['ystar_ft_quantiles_5_25_50_75_95'][2])} \\\\")
    fv = g["feed_vs_statcast"]
    rows.append(f"Stats API feed 2026, challenged & {thou(fv['n'])} & --- & --- & {f2(fv['feed_ystar_ft_quantiles_5_25_50_75_95'][2])} \\\\")
    table("plane", "lrccc", ["Source & Pitches & $\\Delta z$ at front (SD) & $\\Delta z$ at midpoint (SD) & Median $y^*$ (ft) \\\\"], rows)
    m("SFeedDz", f2(fv["dz_raw_in"])); m("SFeedDzSD", f2(fv["dz_raw_sd"])); m("SFeedDzMid", f3(fv["dz_feed_projected_to_mid_in"]))
    m("SFeedTrajIdentical", pct(fv["trajectory_identical_share"])); m("SFeedSzIdentical", pct(fv["sz_identical_share"], 2))
    rp = g["reproduction"]
    rl = [("statcast_raw", "Statcast 2026 as published, ball radius included"), ("statcast_raw_no_ball_radius", "Statcast as published, no ball radius"),
          ("statcast_projected_old", "Statcast projected to the midpoint a second time"),
          ("statcast_projected_old_no_ball_radius", "Projected a second time, no ball radius"),
          ("feed_raw_front", "Stats API feed as published (front of plate)"), ("feed_projected_mid", "Stats API feed projected to the midpoint")]
    table("repro", "lrr", ["Coordinates & Rulings reproduced (\\%) & Mismatches \\\\"],
          [f"{nm} & {pct(rp[k]['share'], 2)} & {thou(rp[k]['mismatches'])} \\\\" for k, nm in rl])
    sh = rp["statcast_projected_old_best_line_shift"]
    m("SOldShiftBottom", f2(sh["bottom_line_shift_in"])); m("SOldShiftTop", f2(sh["top_line_shift_in"]))
    rs = rp["statcast_raw_best_line_shift"]
    m("SRawShift", f2(max(abs(rs["bottom_line_shift_in"]), abs(rs["top_line_shift_in"]))))
    rows = []
    grp = [("edge4", "Edge", {"bottom": "Bottom", "top": "Top", "inside (lik 'arm')": "Inside (batter's side)", "outside (lik 'glove')": "Outside"}),
           ("edge_pitcher", "", {"pitcher arm side": "Pitcher's arm side", "pitcher glove side": "Pitcher's glove side"}),
           ("stand", "Batter", {"L": "Left-handed", "R": "Right-handed"}),
           ("role", "Challenger", {"batter": "Batter", "catcher": "Catcher", "pitcher": "Pitcher"}),
           ("month", "Month", {k: k for k in sorted(g["by"]["month"])})]
    for key, head, names in grp:
        for i, (k, nm) in enumerate(names.items()):
            q = g["by"][key][k]
            rows.append(f"{head if i == 0 else ''} & {nm} & {thou(q['n'])} & {pct(q['raw'], 2)} & {q['mismatches_raw']} & {pct(q['projected'])} \\\\")
        rows.append("\\addlinespace")
    table("bygroup", "llrrrr", ["& & Challenges & As published (\\%) & Mismatches & Projected twice (\\%) \\\\"], rows[:-1])
    of = g["other_feeds"]
    on = {"mlb2026feed": "MLB 2026 regular season", "aaa2025": "Triple-A 2025", "spring2026": "Spring training 2026", "post2026": "MLB 2026 postseason"}
    table("feeds", "lrrr", ["Feed sample & Challenges & Front of plate (\\%) & Projected to midpoint (\\%) \\\\"],
          [f"{nm} & {thou(of[k]['n'])} & {pct(of[k]['raw']['share'])} & {pct(of[k]['projected_to_mid']['share'], 2)} \\\\" for k, nm in on.items()])
    m("SPostStatcastRepro", pct(of["post2026_statcast"]["raw"]["share"], 0))
    zh = g["zone_vs_height"]
    m("SZoneBatters", str(zh["batters"])); m("SZoneTopSD", f3(zh["top_sd_in"])); m("SZoneBotSD", f3(zh["bot_sd_in"]))
    m("SZoneWithin", pct(zh["share_within_025in"])); m("SZoneMax", f3(zh["max_abs_in"]))
    m("SZoneTopMean", f3(zh["top_mean_in"])); m("SZoneBotMean", f3(zh["bot_mean_in"]))
    mm = g["mismatches"]; bc = mm["by_class"]
    m("SMisTie", str(bc["boundary tie (|d| <= 0.5 in, sources agree)"])); m("SMisTrack", str(bc["tracking disagreement (feed agrees with ruling)"]))
    m("SMisOther", str(bc["other"])); m("SMisN", str(mm["n"]))
    m("SMisTieBall", str(mm["boundary_ties_d_positive_ruled_ball"])); m("SMisTieStrike", str(mm["boundary_ties_d_negative_ruled_strike"]))
    m("SMisNearN", thou(mm["challenged_within_05in"])); m("SMisNearRate", pct(mm["mismatch_rate_within_05in"], 2))
    m("SMisFarRate", pct(mm["mismatch_rate_beyond_1in"], 2))
    m("SMisDisagree", str(mm["challenged_statcast_feed_disagree_05in"])); m("SMisFeedRight", str(mm["disagree_feed_right_statcast_wrong"]))
    m("SMisStatcastRight", str(mm["disagree_statcast_right_feed_wrong"]))
    fc = J("feed_counters.json")
    m("SFcTeamGames", thou(fc["team_games"])); m("SFcMatch", thou(fc["team_games_counts_match"]))
    m("SFcBadGame", ", ".join(str(x) for x in fc["games_counts_mismatch"]))
    m("SFcRemainingRule", pct(fc["feed_remaining_is_max0_2_minus_failed"], 0)); m("SFcMaxFailed", str(fc["max_used_failed"]))
    m("SFcGrantTeamGames", str(fc["team_games_with_grant"])); m("SFcGrants", str(fc["grants"])); m("SFcExtraInnings", str(fc["extra_innings"]))
    m("SFcDiffer", str(fc["ours_left_differs_from_feed"])); m("SFcDifferNoGrant", str(fc["ours_left_differs_from_feed_without_grant"]))
    m("SFcZeroLeft", str(fc["challenges_with_zero_left"])); m("SBeyondEight", str(fc["challenges_beyond_8in"])); m("SFcExtraCh", str(fc["extra_inning_challenges"]))
    dc = J("data_counts.json")
    m("SNoLocation", str(dc["challenges"] - dc["called_challenged"]))

    # ------------------------------------------------------------------ S4 perception model
    pv = J("perception_variants.json"); h = J("headline.json"); base = pv["normal"]; main = pv["kappa_sr_region_g"]   # HONZA-122 main
    names = [("sm_edge", "Smooth truth by edge, no mislabel"), ("sm_region", "Smooth truth by region + mislabel"),
             ("normal", "Baseline: deterministic truth + mislabel"), ("logistic", "Logistic $\\varepsilon$"),
             ("t10", "$t_{10}$ $\\varepsilon$"), ("t5", "$t_{5}$ $\\varepsilon$"), ("t3", "$t_{3}$ $\\varepsilon$"),
             ("sr_edge", "$\\sigma_r$ by edge"), ("sr_region", "$\\sigma_r$ by region"), ("sr_slope", "$\\sigma_r$ linear in $|w|$"),
             ("sr_region_slope", "$\\sigma_r$ by region + slope"), ("kappa_region", "$\\kappa$ by region"),
             ("kappa_sr_region", "$\\kappa$ and $\\sigma_r$ by region, $\\kappa$ linear in $\\log g$"), ("kappa_sr_region_slope", "Same + $\\sigma_r$ slope"),
             ("linear_lapse", "Same + lapse rate"),
             ("kappa_sr_region_g", "\\textbf{Same, $\\kappa$ piecewise linear in $\\log g$ (comparator)}"),
             ("main_glow", "Main + low-stake hinge in $\\kappa$"), ("main_sr_g", "Main + $\\sigma_r$ varying with $\\log g$"),
             ("main_logistic", "Main, logistic $\\varepsilon$"), ("main_t10", "Main, $t_{10}$ $\\varepsilon$"),
             ("main_t5", "Main, $t_{5}$ $\\varepsilon$"), ("main_t3", "Main, $t_{3}$ $\\varepsilon$"),
             ("main_lapse", "Main + lapse rate"), ("main_time", "Main, second-half shifts in $\\kappa$, $\\sigma_r$"),
             ("main_time_lapse", "Same + lapse rate"), ("re", "Main + player effects ($5\\times5$ nodes)"),
             ("re_gh11", "Main + player effects ($11\\times11$)"), ("re_lapse", "Main + player effects + lapse ($5\\times5$)"),
             ("re_lapse_gh11", "Main + player effects + lapse ($11\\times11$, primary)"),
             ("main_state", "Main + game-state terms in $\\kappa$"),
             ("re_lapse_state_gh11", "Primary + game-state terms in $\\kappa$")]
    lw = J("lapse_wins.json")
    # HONZA-134 (Opus round-1 minor 3): lapse models at the primary intervention (lapses kept, ell = 0) in both columns
    lapse_lost = lw["opt_score_lapsed"]["wins"] - lw["observed"]["wins"] if "opt_score_lapsed" in lw else lw["lost"]
    rows = []
    for k, nm in names:
        if k not in pv:
            continue
        q = pv[k]; ms = h["model_spread"].get(k, {}) if k != "main_lapse" else {"lost": lapse_lost}
        pi = q["params"].get("contam_pi")
        rows.append(f"{nm} & {q['k']} & ${base['nll'] - q['nll']:+.1f}$ & ${q['bic'] - main['bic']:+.0f}$ & "
                    f"{'---' if pi is None else pct(pi, 2)} & {f2(q['params']['off_sr'])} & {f2(q['params']['def_sr'])} & "
                    f"{f2(ms.get('lost_kept', ms.get('lost'))) if 'lost' in ms else '---'} & {f2(ms['lost_pooled_dp']) if 'lost_pooled_dp' in ms else '---'} \\\\")
    table("variants", "lrrrrrrrr", ["Model & $k$ & $\\Delta\\ell$ & $\\Delta$BIC & $\\pi$ (\\%) & \\multicolumn{2}{c}{$\\sigma_r$ (in)} & \\multicolumn{2}{c}{Wins lost} \\\\",
                                    "& & & & & batter & catcher & score DP & pooled DP \\\\"], rows)
    P = main["params"]; RE = pv["re_lapse_gh11"]["params"]          # HONZA-132: the co-primary model
    cov = ["intercept", "$\\log g$", "one challenge left", "inning$/9$", "two strikes", "three balls"]
    rows = [f"$\\kappa$: {c} & {f2(P['off_beta'][i])} & {f2(P['def_beta'][i])} & {f2(RE['off_beta'][i])} & {f2(RE['def_beta'][i])} \\\\" for i, c in enumerate(cov)]
    for kn in ("-5.0", "-4.0", "-3.2"):               # HONZA-122: hinges of the piecewise-linear criterion in log g
        rows.append(f"$\\kappa$: $(\\log g {'+' if True else ''}{-float(kn):g})_+$ & {f2(P['off_kappa_logg_hinges'][kn])} & {f2(P['def_kappa_logg_hinges'][kn])} & "
                    f"{f2(RE['off_kappa_logg_hinges'][kn])} & {f2(RE['def_kappa_logg_hinges'][kn])} \\\\")
    for r, nm in (("top", "top"), ("arm", "inside"), ("glove", "outside")):
        rows.append(f"$\\kappa$ shift, {nm} edge (in) & {f2(P['off_kappa_region_shift'][r])} & {f2(P['def_kappa_region_shift'][r])} & "
                    f"{f2(RE['off_kappa_region_shift'][r])} & {f2(RE['def_kappa_region_shift'][r])} \\\\")
    rows.append(f"$\\sigma_r$, bottom edge (in) & {f2(P['off_sr'])} & {f2(P['def_sr'])} & {f2(RE['off_sr'])} & {f2(RE['def_sr'])} \\\\")
    for r, nm in (("top", "top"), ("arm", "inside"), ("glove", "outside")):
        rows.append(f"$\\sigma_r$ multiplier, {nm} edge & {f2(P['off_sr_region_mult'][r])} & {f2(P['def_sr_region_mult'][r])} & "
                    f"{f2(RE['off_sr_region_mult'][r])} & {f2(RE['def_sr_region_mult'][r])} \\\\")
    rows.append(f"Mislabel rate $\\pi$ (\\%) & \\multicolumn{{2}}{{c}}{{{pct(P['contam_pi'], 2)}}} & \\multicolumn{{2}}{{c}}{{{pct(RE['contam_pi'], 2)}}} \\\\")
    rows.append(f"SD of player $\\kappa_i$ (in) & --- & --- & {f2(RE['off_re']['sd_kappa'])} & {f2(RE['def_re']['sd_kappa'])} \\\\")
    rows.append(f"SD of player $\\log\\sigma_{{r,i}}$ & --- & --- & {f2(RE['off_re']['sd_logsr'])} & {f2(RE['def_re']['sd_logsr'])} \\\\")
    rows.append(f"Correlation of player effects & --- & --- & {f2(RE['off_re']['rho'])} & {f2(RE['def_re']['rho'])} \\\\")
    rows.append(f"Lapse rate $\\lambda$ (\\%) & --- & --- & {pct(RE['off_lapse'], 1)} & {pct(RE['def_lapse'], 1)} \\\\")
    table("params", "lrrrr", ["& \\multicolumn{2}{c}{Comparator} & \\multicolumn{2}{c}{Player effects + lapse} \\\\",
                              "Parameter & batter & catcher & batter & catcher \\\\"], rows)
    cal = main["calibration"]
    for side, nm in (("off", "Batter"), ("def", "Catcher")):
        c = cal[side]
        m(f"SChiCh{nm}", f"{c['chi2_challenge']:.0f}"); m(f"SChiOv{nm}", f"{c['chi2_overturn']:.0f}"); m(f"SChiBins{nm}", str(c["bins_challenge"]))
        m(f"SChObs{nm}", thou(c["ch_obs"])); m(f"SChPred{nm}", thou(c["ch_pred"]))
    o = J("oos.json")
    rows = []
    for key, nm in (("temporal_insample_train", "Mar--Jun, in sample"), ("temporal_predict_test_with_train", "Jul--Sep, Mar--Jun fit"),
                    ("temporal_predict_test_with_test", "Jul--Sep, Jul--Sep fit"),
                    ("aaa2025_transport_mlb2026", "Triple-A 2025, MLB fit"), ("aaa2025_refit_score", "Triple-A 2025, refit"),
                    ("spring2026_transport_mlb2026", "Spring 2026, MLB fit"), ("spring2026_refit_score", "Spring 2026, refit")):
        q = o[key]
        rows.append(f"{nm} & {thou(q['off']['ch_obs'])} / {thou(q['off']['ch_pred'])} & {pct(q['off']['over_obs'])} / {pct(q['off']['over_pred'])} & "
                    f"{thou(q['def']['ch_obs'])} / {thou(q['def']['ch_pred'])} & {pct(q['def']['over_obs'])} / {pct(q['def']['over_pred'])} \\\\")
    table("oos", "lcccc", ["& \\multicolumn{2}{c}{Batter} & \\multicolumn{2}{c}{Catcher} \\\\",
                           "Sample and parameters & Challenges & Overturn (\\%) & Challenges & Overturn (\\%) \\\\"], rows)
    for key, nm in (("fit_train_marjun", "Train"), ("fit_test_julsep", "Test")):
        m(f"SOOS{nm}SrBatter", f2(o[key]["off_sr"])); m(f"SOOS{nm}SrCatcher", f2(o[key]["def_sr"])); m(f"SOOS{nm}Games", thou(o[key]["games"]))
    for key, nm in (("wins_lost_train_train_params", "TrainTrain"), ("wins_lost_test_train_params", "TestTrain"), ("wins_lost_test_test_params", "TestTest")):
        m(f"SOOSLost{nm}", f2(o[key]["opt_minus_observed"]))
    for key, nm in (("aaa2025_refit", "AAA"), ("spring2026_refit", "Spring")):
        m(f"S{nm}SrBatterCI", ci(o[key]["off_sr_ci95"])); m(f"S{nm}SrCatcherCI", ci(o[key]["def_sr_ci95"]))
    idn = J("identification.json")
    m("SRecovTruthBatter", f2(idn["truth"]["off_sr"])); m("SRecovTruthCatcher", f2(idn["truth"]["def_sr"]))
    rows = []
    for key, nm in (("correct", "Correctly specified"), ("coord_noise_0.3in", "0.3\\,in tracking noise"),
                    ("t5_noise", "$t_5$ perception noise"), ("player_heterogeneity", "Player random effects")):
        q = idn[key]
        rows.append(f"{nm} & {q['converged']}/{q['n']} & ${q['bias_pct']['off_sr']:+.1f}$ & ${q['bias_pct']['def_sr']:+.1f}$ & "
                    f"{f2(q['sd']['off_sr'])} & {f2(q['sd']['def_sr'])} & {pct(q['mean']['pi'], 2)} \\\\")
    table("recovery", "lcrrrrr", ["Data-generating process & Converged & \\multicolumn{2}{c}{Bias of $\\hat\\sigma_r$ (\\%)} & \\multicolumn{2}{c}{SD of $\\hat\\sigma_r$ (in)} & Mean $\\hat\\pi$ (\\%) \\\\",
                                  "& & batter & catcher & batter & catcher & \\\\"], rows)
    m("SRecovPiTruth", pct(idn["truth"]["pi"], 2))
    m("SREUsedBatter", f2(idn["re_sd_used"]["off"][0])); m("SREUsedLogBatter", f2(idn["re_sd_used"]["off"][1]))
    br = J("sr_breakeven.json")

    def prof(side, mult):
        s0 = idn["truth"][f"{side}_sr"]
        return dict((round(x / s0, 2), ll) for x, ll in idn[f"profile_{side}"]).get(round(mult, 2))
    rows = []
    for k in sorted(br, key=float):
        q = br[k]; x = float(k); pb, pc = prof("off", x), prof("def", x)
        rows.append(f"{x:.1f} & {f2(q['opt'])} & ${q['lost']:+.2f}$ & ${q['lost_pooled_dp']:+.2f}$ & "
                    f"{'---' if pb is None else f'{pb:.1f}'} & {'---' if pc is None else f'{pc:.1f}'} \\\\")
    table("breakeven", "rrrrrr", ["$\\sigma_r$ / estimate & Benchmark & \\multicolumn{2}{c}{Wins lost} & \\multicolumn{2}{c}{Profile $-\\Delta\\ell$ (baseline model)} \\\\",
                                  "& & score DP & pooled DP & batter & catcher \\\\"], rows)
    m("SObservedWins", f2(next(iter(br.values()))["observed"]))
    fcj = J("feed_check.json")
    m("SFeedMainSrBatter", f2(fcj["kappa_sr_region_g"]["params"]["off_sr"])); m("SFeedMainSrCatcher", f2(fcj["kappa_sr_region_g"]["params"]["def_sr"]))
    m("SFeedLost", f2(fcj["normal"]["wins"]["lost"])); m("SFeedLostMain", f2(fcj["kappa_sr_region_g"]["wins"]["lost"]))
    vw = J("variant_wins.json")["calibration_test"]
    for side, nm in (("off", "Batter"), ("def", "Catcher")):
        m(f"SThrCal{nm}", f2(vw[side]["calibrated"]["mean_p_threshold"])); m(f"SThrFlat{nm}", f2(vw[side]["flat_prior"]["mean_p_threshold"]))

    # ------------------------------------------------------------------ S5 policy evaluation
    c = J("causal.json"); gb = h["causal"]["game_bootstrap"]
    lab = {"opt_score": "Benchmark (score-aware DP)", "runs_dp": "Runs-objective rule", "opt": "Pooled DP", "stake_0.008": "One-line rule, $\\delta=0.008$",
           "stake_0.01": "One-line rule, $\\delta=0.01$", "card_pub": "48-cell card, published", "card_refit": "48-cell card, refit",
           "obs_rule": "Fitted behavioural rule", "savant": "Fixed 0.2-run rule"}
    rows = []
    for k, nm in lab.items():
        q = gb[k]
        rows.append(f"{nm} & {f2(q['path'][0])} & {f2(q['causal_rb'][0])} & {f2(q['causal_binary'][0])} & "
                    f"${q['causal_minus_path'][0]:+.3f}$ {ci(q['causal_minus_path'][1:], f3)} & ${q['binary_minus_rb'][0]:+.2f}$ {ci(q['binary_minus_rb'][1:])} \\\\")
    table("causal", "lrrrcc", ["Policy & Path sum & RB & Win count & RB $-$ path & Win count $-$ RB \\\\"], rows)
    pt = c["point"]["opt_score"]
    m("SCalibSlope", f3(pt["calib_slope"]))
    m("SLaterRecorded", f2(pt["later_recorded"])); m("SLaterNoflip", f2(pt["later_sim_noflip"])); m("SLaterFlip", f2(pt["later_sim_flip"]))
    m("SBranches", f2(pt["branches_per_team_game"]))
    fid = gb["opt_score"]["fidelity_noflip_minus_recorded"]
    m("SFidelity", f2(fid[0])); m("SFidelityCI", ci(fid[1:]))
    cw = [q["calib_winB_minus_engine"] for q in c["point"].values() if "calib_winB_minus_engine" in q]
    m("SCalibWinBLo", f3(min(cw))); m("SCalibWinBHi", f3(max(cw)))
    Pol = {q["policy"]: q for q in h["policies"]}
    order = ["bench", "runs_dp", "opt", "stake_0.008", "stake_0.01", "card_pub", "card_refit", "observed", "obs_rule", "savant"]
    lab["observed"] = "Observed 2026 (realised)"; lab["bench"] = "Benchmark"
    rows = [f"{lab[k]} & {f2(Pol[k]['batter'])} & {f2(Pol[k]['fielder'])} & {f2(Pol[k]['n'])} {ci(Pol[k]['n_ci'])} \\\\" for k in order]
    table("roles", "lrrc", ["Policy & Wins, batters & Wins, fielding side & Challenges per team-game \\\\"], rows)
    fb = J("flat_bound.json")
    rows = [f"{r['delta']:.3f} & {pct(r['vdelta'] / r['vstar'])} & {f1(r['loss_pct'])} & {f1(r['local_pct'])} & {f1(r['bound_pct'])} & {sci(r['identity_gap'])} \\\\"
            for r in fb["table"] if round(r["delta"], 3) in (0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.012, 0.015, 0.02)]
    table("flat", "rrrrrr", ["$\\delta$ & $V^\\delta/V^\\star$ (\\%) & Loss (\\%) & Local approx.\\ (\\%) & Bound (ii) (\\%) & Identity gap \\\\"], rows)
    m("SFlatR", str(fb["R"]))
    # HONZA-132: the engine-fixed bootstrap S-macros (SBoot*, SMC*) are gone; the joint bootstrap is in make_numbers
    cf = A("crossfit_calibration.json")
    d_opt = [r["cross_fit"]["opt_score"] - r["in_sample"]["opt_score"] for r in cf["rows"]]
    d_rule = [r["cross_fit"]["stake_0008"] - r["in_sample"]["stake_0008"] for r in cf["rows"]]
    m("SCrossFitOpt", neg(f"{sum(d_opt) / len(d_opt):+.3f}")); m("SCrossFitRule", neg(f"{sum(d_rule) / len(d_rule):+.3f}")); m("SCrossFitR", str(cf["R"]))
    wc = A("wp_calibration.json")
    m("SWPBrier", f4(wc["brier_engine"])); m("SWPBrierSC", f4(wc["brier_statcast_home_win_exp"])); m("SWPCorr", f4(wc["corr_engine_statcast"]))
    m("SWPHomeRate", pct(wc["home_win_rate"])); m("SWPHomeGap", pct(wc["home_field_gap_tied_states"]))
    z = [abs(r["z"]) for r in wc["by_inning_score"]]
    m("SWPCells", str(len(z))); m("SWPCellsBig", str(sum(x > 3 for x in z)))
    ind = A("independent.json")
    m("SIndObs", f2(ind["observed_wins_per_team_season"])); m("SIndSavant", f2(ind["savant_wins_per_team_season"]))
    m("SIndRule", f2(ind["stake_0.008_wins_per_team_season"])); m("SIndSavantRatio", f2(ind["savant_over_observed"]))

    # ------------------------------------------------------------------ S6 cricket DRS
    d = J("drs/summary.json"); rc = A("drs/recount.json"); cnt = rc["counts"]
    xs = [100 * d[k]["crossfit_delta"]["share"] for k in ("Test", "ODI", "T20") if "crossfit_delta" in d[k]]   # HONZA-122
    m("SDRSCrossShareLo", f"{min(xs):.0f}"); m("SDRSCrossShareHi", f"{max(xs):.0f}")
    F = ("Test", "ODI", "T20")
    rowf = lambda lab_, fn: f"{lab_} & " + " & ".join(fn(k) for k in F) + " \\\\"
    rows = [rowf("DRS matches", lambda k: thou(cnt[k]["matches"])),
            rowf("Batting-side out decisions", lambda k: thou(cnt[k]["bat_decisions"])),
            rowf("\\quad of which LBW", lambda k: thou(cnt[k]["bat_dec_lbw"])),
            rowf("Batting reviews", lambda k: thou(cnt[k]["bat_rev"])),
            rowf("\\quad overturned / umpire's call / lost", lambda k: f"{cnt[k]['bat_success']} / {cnt[k]['bat_uc']} / {cnt[k]['bat_lost']}"),
            rowf("Fielding reviews", lambda k: thou(cnt[k]["bowl_rev"])),
            rowf("\\quad overturned / umpire's call / lost", lambda k: f"{cnt[k]['bowl_success']} / {cnt[k]['bowl_uc']} / {cnt[k]['bowl_lost']}")]
    table("drsdata", "lrrr", [" & Test & ODI & T20 \\\\"], rows)
    m("SDRSNoneLeft", str(sum(rc["none_left"].values()))); m("SDRSReviews", thou(sum(rc["review_types"].values())))
    idd = J("drs/identification.json")
    rows = [rowf("Perception noise $\\hat s$ (margin SD)", lambda k: f2(d[k]["model"]["s"])),
            rowf("\\quad 95\\% profile-likelihood interval", lambda k: ci(d[k]["model"]["s_95"], f1)),
            rowf("Umpire's-call band $\\hat\\kappa$", lambda k: f2(d[k]["model"]["kappa"])),
            rowf("LBW out decisions that are wrong (\\%)", lambda k: pct(d[k]["model"]["lbw_error_rate"])),
            rowf("Caught out decisions that are wrong (\\%)", lambda k: pct(d[k]["model"]["caught_error_rate"], 2)),
            rowf("Constant-rate model: $P(\\text{kept}\\mid\\text{failed})$", lambda k: f2(d[k]["model"]["rho"])),
            rowf("Known truth: median $\\hat s$ (truth $=$ fitted $\\hat s$)", lambda k: f2(idd[k]["s_median"])),
            rowf("\\quad replicates with $\\hat s>1$", lambda k: f"{sum(r['s'] > 1 for r in idd[k]['reps'])}/{len(idd[k]['reps'])}"),
            rowf("\\quad coverage of the 95\\% LR set (\\%)", lambda k: pct(idd[k]["lr_cover_rate"], 0))]
    table("drsmodel", "lrrr", [" & Test & ODI & T20 \\\\"], rows)
    shares = [sum(r["s"] > 1 for r in idd[k]["reps"]) / len(idd[k]["reps"]) for k in F]
    m("SDRSOverOneLo", pct(min(shares), 0)); m("SDRSOverOneHi", pct(max(shares), 0)); m("SDRSRecovR", str(idd["Test"]["R"]))
    per = lambda k, key: d[k]["per100"][key]
    rows = [rowf("Benchmark", lambda k: f"{f2(per(k, 'opt')[0])} {ci(per(k, 'opt')[1])}"),
            rowf("One-line rule", lambda k: f"{f2(per(k, 'stake')[0])} ($\\delta={per(k, 'best_delta')}$)"),
            rowf("\\quad share of benchmark (\\%)", lambda k: f"{pct(per(k, 'stake_share')[0])} {ci([100 * x for x in per(k, 'stake_share')[1]], f1)}"),
            rowf("Teams' fitted rule (simulated)", lambda k: f2(per(k, "obs_rule")[0])),
            rowf("Observed (realised)", lambda k: f"{f2(per(k, 'observed')[0])} {ci(per(k, 'observed')[1])}"),
            rowf("Review iff $p\\ge0.5$", lambda k: f2(per(k, "conf50"))),
            rowf("Lost vs fitted rule", lambda k: f"{f3(per(k, 'lost_vs_rule')[0])} {ci(per(k, 'lost_vs_rule')[1], f3)}"),
            rowf("Lost vs realised", lambda k: f"{f2(per(k, 'lost_vs_observed')[0])} {ci(per(k, 'lost_vs_observed')[1])}"),
            rowf("Reviews per team-match: benchmark / rule / observed", lambda k: " / ".join(f2(d[k]["reviews_per_team_match"][x]) for x in ("opt", "stake", "observed")))]
    table("drsresults", "lccc", [" & Test & ODI & T20 \\\\"], rows)
    rows = []
    for k in F:
        t = d[k]["thresholds_lbw_pct"]
        rows.append(f"{k}, 1 left & " + " & ".join(f"{x:.1f}" for x in t["c1"]) + " \\\\")
        rows.append(f"{k}, 2 left & " + " & ".join(f"{x:.1f}" for x in t["c2"]) + " \\\\")
    table("drsthr", "l" + "r" * 10, ["Wickets down & " + " & ".join(str(i) for i in range(10)) + " \\\\"], rows)
    ck = J("drs/check.json")
    rows = [rowf("$R^2$ of $\\Delta_1$ on $\\sqrt N$", lambda k: f3(ck[f"E_{k}"]["R2_sqrt"])),
            rowf("$R^2$ of $\\Delta_1$ on $N$", lambda k: f3(ck[f"E_{k}"]["R2_linear"])),
            rowf("Power exponent [95\\% CI]", lambda k: f"{f2(ck[f'E_{k}']['power_exponent'])} {ci(ck[f'E_{k}']['power_exponent_CI'])}"),
            rowf("Expected out decisions left at the start, $N$", lambda k: f2(d[k]["sqrt_law"]["N"][0]))]
    table("drssqrt", "lrrr", [" & Test & ODI & T20 \\\\"], rows)
    sv = J("drs/sensitivity.json")

    def sens(k, key):
        q = d[k]["sensitivity"][key] if key in d[k]["sensitivity"] else sv[k][key]
        return f"{f2(q['opt'])} / {f3(q['lost_vs_rule'])} / {pct(q['stake_share'])}"
    rows = [rowf("Main (band model, $\\hat s$)", lambda k: sens(k, "main")),
            rowf("Upper end of the $s$ set", lambda k: sens(k, "s_hi") + f" ($s={d[k]['sensitivity']['s_hi']['s']}$)"),
            rowf("$s=2$", lambda k: sens(k, "s2.0") + f" ($-\\Delta\\ell={d[k]['sensitivity']['s2.0']['dnll']}$)"),
            rowf("Constant-rate umpire's call", lambda k: sens(k, "const")),
            rowf("Recalibrated WP engine", lambda k: sens(k, "recal")),
            rowf("Stakes $\\times1.5$", lambda k: sens(k, "g_x1.5")),
            rowf("Stakes $\\times0.67$", lambda k: sens(k, "g_x0.67"))]
    table("drssens", "lccc", [" & Test & ODI & T20 \\\\"], rows)
    for k, nm in zip(F, ("Test", "ODI", "Tt")):
        b = d[k]["bowl"]
        m(f"SDRSBowl{nm}Success", pct(b["success"], 0)); m(f"SDRSBowl{nm}UC", pct(b["uc"], 0))
