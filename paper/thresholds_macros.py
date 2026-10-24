"""Macros from results/thresholds_score.json (src/refresh_thresholds.py, HONZA-77); imported by make_numbers.py."""


def macros(d):
    g, bg = d["gaps_pp"], d["bootstrap"]["gaps_pp"]
    out = {}
    for key, nm in (("early_vs_late_c1_pp", "GapEarlyLateOne"), ("early_vs_late_c2_pp", "GapEarlyLateTwo"),
                    ("c1_vs_c2_mid_pp", "GapOneTwo")):
        for suf, tag in (("tied", ""), ("occ", "Occ"), ("lead-3", "Down"), ("lead+3", "Up")):
            out[nm + tag] = f"{g[f'{key}_{suf}']:.1f}"
        for suf, tag in (("tied", ""), ("occ", "Occ")):
            b = bg[f"{key}_{suf}"]
            out[nm + tag + "CI"] = f"[{b[1]:.1f}, {b[2]:.1f}]"
    t = d["thresholds"]
    for k, nm in (("early_c1_tied", "ThrEarlyTied"), ("mid5_c1_tied", "ThrMidTied"), ("late8_c1_tied", "ThrLateTied"),
                  ("late8_c1_occ", "ThrLateOcc")):
        out[nm] = f"{t[k]:.2f}"
    s = d["sqrt_n"]
    out["SqrtRTwo"] = f"{s['home']['occ_model']['sqrt']['R2']:.4f}"
    out["SqrtRTwoAway"] = f"{s['away']['occ_model']['sqrt']['R2']:.4f}"
    out["LinRTwo"] = f"{s['home']['occ_model']['linear']['R2']:.2f}"
    out["DeltaSqRTwoHome"] = f"{s['home']['occ_model']['sq_on_n']['R2']:.4f}"
    out["DeltaSqRTwoAway"] = f"{s['away']['occ_model']['sq_on_n']['R2']:.4f}"
    out["SqrtRTwoTied"] = f"{s['home']['tied_model']['sqrt']['R2']:.2f}--{s['away']['tied_model']['sqrt']['R2']:.2f}"
    out["SqrtCoef"] = f"{s['home']['occ_model']['sqrt']['b']:.4f}"
    r = [s[x]["D2_over_D1_first_pitch"] for x in ("home", "away")]
    lo, hi = f"{min(r):.2f}", f"{max(r):.2f}"
    out["DeltaRatio"] = lo if lo == hi else f"{lo}--{hi}"
    out["DeltaFirst"] = f"{s['home']['D1_first']:.3f}"
    out["GapMethod"] = f"game bootstrap with refits, $B={d['bootstrap']['B']}$"
    return out
