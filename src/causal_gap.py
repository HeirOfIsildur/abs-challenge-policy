"""(audit R1): where does the binary-vs-RB gap of the causal check come from?

causal.py's two estimators agree only if the WP engine (wp.py, which prices every flip as g) is the value function of
the pitch-level simulator that replays the game after the flip.  Both ignore the home team and use the same extra-
inning and walk-off rules, so a gap has to come from the PA dynamics.  The engine's PA kernel T[count, outs, bases] is
the empirical distribution of the PA's end (outs, bases, runs) from every 2026 pitch at that count.  The simulator built the PA pitch by pitch from
  ev[count] -> event (called / other strike / foul / in play / HBP / automatic ball), not by outs and bases,
  kip[outs, bases] -> in-play outcome, the same after 0-2 as after 3-0,
  and no baserunning between pitches (steals, wild pitches, pickoffs).
Each simplification is a Markov chain whose PA kernel and WP table are exact (no Monte Carlo): chain_T turns
(ev, kip, called-strike share, baserunning) into a T, wp.solve(T) gives its WP table and stakes.Vcount its g.  Steps:
  sim    ev[c],        kip[o, b]       (causal.py up to )
  kc     ev[c],        kip[c, o, b]    (in-play outcome depends on the count)
  kcob   ev[c, o, b],  kip[c, o, b]    (events also depend on outs and bases)
  run    kcob + between-pitch baserunning M[o, b] -> (o', b', runs)   (causal.py from )
  engine wp.py's T
  homefield  wp.py's T estimated separately for each batting team (the engine has no home-field term, audit R5)
Sparse cells are shrunk toward the coarser kernel with weight 20, as in wp.py.
Predicted gap of a policy = sum over its branch pitches of (g_chain - g_engine), in wins per team-season, on the same
branch pitches (R = 4, seed 1) as causal.py; the measured gap is binary - RB in results/causal_sim.json (simulator) and
results/causal.json (simulator "run").  A Monte Carlo
check runs causal.play itself (no challenges) from sampled branch states and compares its flip swing with the chain's g.
usage: python src/causal_gap.py [K_mc] [n_mc] [cached] | md -> results/causal_gap.json, data/gtable_<step>.npz"""
import json, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "src")
import wp

SHRINK = 20.0
CAT = dict(ball=0, blocked_ball=0, called_strike=0, swinging_strike=1, swinging_strike_blocked=1, foul_tip=1,
           missed_bunt=1, foul_bunt=1, bunt_foul_tip=1, swinging_pitchout=1, automatic_strike=1, foul=2,
           hit_into_play=3, hit_by_pitch=4, automatic_ball=5, pitchout=5)
STEPS = ["sim", "kc", "kcob", "run"]


def pitches():
    sc = pd.read_parquet("data/statcast_2026.parquet", columns=["game_pk", "at_bat_number", "pitch_number", "inning", "inning_topbot",
                         "outs_when_up", "on_1b", "on_2b", "on_3b", "balls", "strikes", "bat_score", "post_bat_score", "description", "type"])
    sc = sc.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    sc["cat"] = sc.description.map(CAT)
    sc.loc[(sc.description == "hit_into_play") & (sc.type != "X"), "cat"] = np.nan
    sc["bases"] = sc.on_1b.notna() * 1 + sc.on_2b.notna() * 2 + sc.on_3b.notna() * 4
    sc["half"] = (sc.inning_topbot == "Bot").astype(int)
    sc["cnt"] = sc.balls * 3 + sc.strikes
    return sc


def kernels(sc=None):
    """ev[c, o, b, 6] and kip[c, o, b, 160] (index 40 oe + 5 be + r) at each pooling level, the called-strike share of
    the simulator's called-pitch pool by count, and the between-pitch baserunning kernel M[o, b, 160]."""
    sc = pitches() if sc is None else sc
    e = sc[(sc.balls <= 3) & (sc.strikes <= 2)].dropna(subset=["cat"])
    E = np.zeros((12, 3, 8, 6)); np.add.at(E, (e.cnt.values, e.outs_when_up.values, e.bases.values, e.cat.astype(int).values), 1)
    ev_c = E.sum((1, 2)); ev_c = ev_c / ev_c.sum(1, keepdims=True)
    ev_cob = (E + SHRINK * ev_c[:, None, None, :]) / (E.sum(3, keepdims=True) + SHRINK)
    pa = sc.groupby(["game_pk", "at_bat_number"]).agg(inning=("inning", "first"), half=("half", "first"), outs=("outs_when_up", "first"),
            bases=("bases", "first"), end_score=("post_bat_score", "last"), start=("bat_score", "first"), last=("type", "last"),
            cnt=("cnt", "last"), ol=("outs_when_up", "last"), bl=("bases", "last"), sl=("bat_score", "last")).reset_index()
    nxt = pa.groupby("game_pk").shift(-1)
    same = (nxt.inning == pa.inning) & (nxt.half == pa.half)
    pa["oe"] = np.where(same, nxt.outs, 3).astype(int); pa["be"] = np.where(same, nxt.bases, 0).astype(int)
    pa["r"] = (pa.end_score - pa.start).clip(0, 4).astype(int)
    ip = pa[(pa["last"] == "X") & (pa.oe >= pa.outs)]
    # kernel (state at the PA's first pitch) and the count kernel (state and count at the in-play pitch)
    k_ob = np.zeros((3, 8, 160)); np.add.at(k_ob, (ip.outs.values, ip.bases.values, ip.oe.values * 40 + ip.be.values * 5 + ip.r.values), 1)
    k_ob = k_ob + 1e-12; k_ob = k_ob / k_ob.sum(2, keepdims=True)
    ipc = ip[(ip.cnt < 12) & (ip.oe >= ip.ol)]
    rl = (ipc.end_score - ipc.sl).clip(0, 4).astype(int).values
    Kc = np.zeros((12, 3, 8, 160)); np.add.at(Kc, (ipc.cnt.values, ipc.ol.values, ipc.bl.values, ipc.oe.values * 40 + ipc.be.values * 5 + rl), 1)
    k_ob2 = Kc.sum(0) + 1e-12; k_ob2 = k_ob2 / k_ob2.sum(2, keepdims=True)
    k_cob = (Kc + SHRINK * k_ob2[None]) / (Kc.sum(3, keepdims=True) + SHRINK)
    cp = pd.read_parquet("data/called_2026_w.parquet", columns=["balls", "strikes", "orig_strike"])
    ps = cp.groupby(cp.balls * 3 + cp.strikes).orig_strike.mean().reindex(range(12)).values
    # between consecutive pitches of the same PA: (outs, bases, runs) change without the PA ending
    nx = sc.groupby(["game_pk", "at_bat_number"]).shift(-1)
    ok = (nx.pitch_number.notna() & (sc.cnt < 12)).values
    o, b = sc.outs_when_up.values[ok], sc.bases.values[ok]
    o2, b2 = nx.outs_when_up.values[ok].astype(int), nx.bases.values[ok].astype(int)
    r = np.clip(nx.bat_score.values[ok] - sc.bat_score.values[ok], 0, 4).astype(int)
    keep = o2 >= o
    M = np.zeros((3, 8, 160)); np.add.at(M, (o[keep], b[keep], o2[keep] * 40 + b2[keep] * 5 + r[keep]), 1)
    nM = M.sum(2, keepdims=True); stay = np.zeros((3, 8, 160))
    for oo in range(3):
        for bb in range(8):
            stay[oo, bb, oo * 40 + bb * 5] = 1.0
    M = (M + SHRINK * stay) / (nM + SHRINK)
    pmove = 1 - np.array([M[oo, bb, oo * 40 + bb * 5] for oo in range(3) for bb in range(8)])
    return dict(ev_c=np.broadcast_to(ev_c[:, None, None, :], (12, 3, 8, 6)).copy(), ev_cob=ev_cob,
                k_ob=np.broadcast_to(k_ob[None], (12, 3, 8, 160)).copy(), k_cob=k_cob, ps=ps, M=M,
                p_move=float((pmove * nM.reshape(-1)).sum() / nM.sum()))


def _walk(b):
    if not b & 1: return b | 1, 0
    if not b & 2: return b | 3, 0
    if not b & 4: return 7, 0
    return 7, 1


def _shift(v, r0):
    """outcome distribution v [4, 8, 5] with r0 more runs (capped at 4)"""
    out = np.zeros_like(v)
    for r in range(5):
        out[:, :, min(r + r0, 4)] += v[:, :, r]
    return out


def _term(oe, be, r):
    v = np.zeros((4, 8, 5)); v[oe, 0 if oe >= 3 else be, min(r, 4)] = 1.0
    return v


def chain_T(ev, kip, ps, M=None):
    """PA kernel T[c, o, b, oe, be, r] of the pitch-level chain (wp.py layout).  ev[c, o, b, 6], kip[c, o, b, 160],
    ps[c] = P(strike | called); M[o, b, 160] = baserunning before every pitch (None: none).  State (count, outs, bases,
    runs so far in the PA); runs capped at 4 as in wp.py."""
    pre = np.zeros((12, 3, 8, 5, 4, 8, 5))                 # before the baserunning step at count c
    for bl in (3, 2, 1, 0):
        for st in (2, 1, 0):
            c = bl * 3 + st
            post = np.zeros((3, 8, 5, 4, 8, 5))            # after baserunning, the pitch is thrown
            for it in range(200):
                old = pre[c].copy()
                for o in range(3):
                    for b in range(8):
                        e = ev[c, o, b]; kp = kip[c, o, b].reshape(4, 8, 5).copy()
                        kp[3, 0] = kp[3].sum(0); kp[3, 1:] = 0                       # third out: bases cleared
                        nb, rw = _walk(b)
                        for r0 in range(5):
                            vs = _term(o + 1, b, r0) if st == 2 else pre[c + 1, o, b, r0]
                            vb = _term(o, nb, r0 + rw) if bl == 3 else pre[c + 3, o, b, r0]
                            vf = pre[c, o, b, r0] if st == 2 else vs
                            post[o, b, r0] = (e[0] * (ps[c] * vs + (1 - ps[c]) * vb) + e[1] * vs + e[2] * vf + e[3] * _shift(kp, r0)
                                              + e[4] * _term(o, nb, r0 + rw) + e[5] * vb)
                if M is None:
                    pre[c] = post
                else:
                    for o in range(3):
                        for b in range(8):
                            for r0 in range(5):
                                acc = np.zeros((4, 8, 5))
                                for k in np.nonzero(M[o, b])[0]:
                                    o2, b2, r = k // 40, (k % 40) // 5, k % 5
                                    acc += M[o, b, k] * (_term(3, 0, r0 + r) if o2 >= 3 else post[o2, b2, min(r0 + r, 4)])
                                pre[c, o, b, r0] = acc
                if (st < 2 and M is None) or np.abs(pre[c] - old).max() < 1e-12:
                    break
    return pre[:, :, :, 0]


def gtable(T, W):
    """g for every pre-pitch state [inn 1..10, half, outs, bases, diff +-10, balls, strikes] under (T, W), with
    stakes.Vcount's own logic (same layout as causal.stake_tables)."""
    import stakes
    stakes.W = W
    G = np.zeros((10, 2, 3, 8, 21, 4, 3))
    for i in range(10):
        for h in range(2):
            stakes.T = T[h] if T.ndim == 7 else T
            for o in range(3):
                for b in range(8):
                    for d in range(-10, 11):
                        for bl in range(4):
                            for st in range(3):
                                wb, _ = stakes.Vcount(i + 1, h, o, b, d, bl + 1, st)
                                ws, _ = stakes.Vcount(i + 1, h, o, b, d, bl, st + 1)
                                G[i, h, o, b, d + 10, bl, st] = max((1 if h else -1) * (wb - ws), 0.0)   # signed, as stakes.py
    return G


def build_tables(cached=False):
    z = np.load("data/wp_engine.npz"); Te, We = z["T"], z["W"]
    k = kernels()
    spec = dict(sim=(k["ev_c"], k["k_ob"], None), kc=(k["ev_c"], k["k_cob"], None), kcob=(k["ev_cob"], k["k_cob"], None),
                run=(k["ev_cob"], k["k_cob"], k["M"]))
    out = dict(engine=(Te, We, gtable(Te, We)))
    try:
        if not cached: raise FileNotFoundError
        z = np.load("data/gtable_homefield.npz"); out["homefield"] = (z["T"], z["W"], z["G"])
    except FileNotFoundError:                               # engine with one PA kernel per batting team (audit R5)
        Th = wp.build_T(by_half=True); Wh, _, _ = wp.solve(Th, verbose=False)
        out["homefield"] = (Th, Wh, gtable(Th, Wh)); np.savez("data/gtable_homefield.npz", T=Th, W=Wh, G=out["homefield"][2])
    for name, (ev, kip, M) in spec.items():
        t = time.time()
        if cached:
            try:
                z = np.load(f"data/gtable_{name}.npz"); out[name] = (z["T"], z["W"], z["G"]); continue
            except FileNotFoundError:
                pass
        T = chain_T(ev, kip, k["ps"], M)
        Wc, _, _ = wp.solve(T, verbose=False)
        out[name] = (T, Wc, gtable(T, Wc))
        np.savez(f"data/gtable_{name}.npz", T=T, W=Wc, G=out[name][2])
        print(name, "chain + WP + g", round(time.time() - t), "s", flush=True)
    return out, k


def main(K_mc=2000, n_mc=600):
    from headline import evaluate_all, spec_variant, MAIN
    import sim, dp_score, causal
    tabs, k = build_tables(cached=len(sys.argv) > 3 and sys.argv[3] == "cached")
    Ge = tabs["engine"][2]
    # 1) kernel-level differences: mean |T_chain - T_engine| and the count gradient of in-play outcomes (runs)
    runs_after_ip = {c: float((k["k_cob"][c].reshape(3, 8, 4, 8, 5).sum((2, 3)) * np.arange(5)).sum(-1).mean()) for c in range(12)}
    res = dict(p_move_between_pitches=k["p_move"], mean_inplay_runs_by_count={f"{c // 3}-{c % 3}": round(v, 4) for c, v in runs_after_ip.items()},
               T_l1_vs_engine={s: float(np.abs(tabs[s][0] - tabs["engine"][0]).sum((3, 4, 5)).mean()) for s in STEPS})
    # 2) the branch pitches of causal.py (same draws: R = 4, seed 1)
    v = spec_variant(MAIN)
    m, kin, out, draws = evaluate_all(v, R=4, seed=1, keep_draws=True)
    ii = np.minimum(m.inning.values, 10) - 1; hh = (~m.top.values).astype(int)
    I = lambda x: np.asarray(x).astype(np.int64)                  # state columns can arrive as float
    st = (I(ii), I(hh), I(m.outs_when_up.values), I(m.bases.values), I(np.clip(m.score_diff_home.values, -10, 10) + 10),
          I(m.balls.values), I(m.strikes.values))
    g_row = {s: tabs[s][2][st] for s in ["engine"] + STEPS}
    small = np.abs(m.score_diff_home.values) <= 10                 # the tables clip the score at +-10, as causal.py does
    assert np.allclose(g_row["engine"][small], m.g.values[small], atol=1e-9), "engine g table does not reproduce m.g"
    meas = json.load(open("results/causal_sim.json"))["game_bootstrap"]          # simulator (run)
    try:
        cr = json.load(open("results/causal.json"))
        meas_run = cr["game_bootstrap"] if cr.get("simulator") == "run" else None
    except FileNotFoundError:
        meas_run = None
    pred = {}
    allrows, allteam = [], []
    for pol in causal.POLICIES:
        acc = {s: 0.0 for s in STEPS}; extra = {s: 0.0 for s in STEPS}; nb = 0
        for dr in draws:
            first, _ = dr["firsts"][pol]
            gg, tt = np.nonzero(first >= 0); rows = first[gg, tt]
            for s in STEPS:
                d = g_row[s][rows] - g_row["engine"][rows]
                acc[s] += d.sum() / 4 / 30; extra[s] += d[m.inning.values[rows] >= 10].sum() / 4 / 30
            nb += len(rows) / 4
            if pol == "opt_score":
                allrows.append(rows); allteam.append(tt)
        pred[pol] = dict(measured_binary_minus_rb=meas[pol]["binary_minus_rb"],
                         measured_binary_minus_rb_run=meas_run[pol]["binary_minus_rb"] if meas_run else None, branches_per_season=nb / 30,
                         mean_g_branch_engine=float(np.mean(np.concatenate([g_row["engine"][first[first >= 0]] for first, _ in [d["firsts"][pol] for d in draws]]))),
                         predicted={s: round(acc[s], 4) for s in STEPS}, predicted_extra_innings={s: round(extra[s], 4) for s in STEPS})
        print(pol, pred[pol], flush=True)
    res["by_policy"] = pred
    # 3) by inning / score for the optimum: where the engine and the simulator disagree
    rows = np.concatenate(allrows)
    inn = np.minimum(m.inning.values[rows], 10); dd = np.abs(m.score_diff_home.values[rows])
    tab = []
    for lo, hi, lab in [(1, 3, "1-3"), (4, 6, "4-6"), (7, 8, "7-8"), (9, 9, "9"), (10, 10, "10+")]:
        for dlo, dhi, dlab in [(0, 0, "tied"), (1, 2, "1-2"), (3, 99, "3+")]:
            k_ = (inn >= lo) & (inn <= hi) & (dd >= dlo) & (dd <= dhi)
            if k_.sum() == 0: continue
            tab.append(dict(inning=lab, margin=dlab, n=int(k_.sum()), g_engine=round(float(g_row["engine"][rows][k_].mean()), 5),
                            **{f"rel_{s}": round(float(g_row[s][rows][k_].sum() / g_row["engine"][rows][k_].sum() - 1), 4) for s in STEPS},
                            share_of_gap_sim=round(float((g_row["sim"][rows][k_] - g_row["engine"][rows][k_]).sum() / (g_row["sim"][rows] - g_row["engine"][rows]).sum()), 3)))
    res["opt_score_by_inning_margin"] = tab
    # by count of the branch pitch
    bc = m.balls.values[rows] * 3 + m.strikes.values[rows]
    res["opt_score_by_count"] = [dict(count=f"{c // 3}-{c % 3}", n=int((bc == c).sum()),
                                      **{f"rel_{s}": round(float(g_row[s][rows][bc == c].sum() / g_row["engine"][rows][bc == c].sum() - 1), 4) for s in STEPS})
                                 for c in range(12) if (bc == c).sum() > 0]
    # 3b) home-field term (audit R5): g with one PA kernel per batting team, first-order effect on wins lost
    Wh = tabs["homefield"][1]
    g_hf = tabs["homefield"][2][st]
    S_opt = out["opt_score"][0].sum() / 30; S_obs = out["observed"][0].sum() / 30
    eps_opt = float(g_hf[rows].sum() / g_row["engine"][rows].sum() - 1)
    full_m, _ = sim.load(v); fm = full_m[full_m.ch & full_m.ch_over]
    fi = (I(np.minimum(fm.inning.values, 10) - 1), I((~fm.top.values).astype(int)), I(fm.outs_when_up.values), I(fm.bases.values),
          I(np.clip(fm.score_diff_home.values, -10, 10) + 10), I(fm.balls.values), I(fm.strikes.values))
    eps_obs = float(tabs["homefield"][2][fi].sum() / Ge[fi].sum() - 1)
    res["home_field"] = dict(home_wp_first_pitch=float(Wh[0, 0, 0, 0, 20]), home_wp_extra_start=float(Wh[9, 0, 0, 2, 20]),
                             rel_g_opt_branches=round(eps_opt, 4), rel_g_observed_successes=round(eps_obs, 4),
                             first_order_change_in_wins_lost=round(float(S_opt * eps_opt - S_obs * eps_obs), 4),
                             predicted_binary_minus_rb_vs_homefield_engine=round(float((g_row["sim"][rows] - g_hf[rows]).sum() / 4 / 30), 4))
    print("home field", res["home_field"], flush=True)
    # 4) Monte Carlo check: causal.play itself (no challenges at all) from n_mc sampled opt_score branch states, K_mc
    # paths per call with common random numbers -> the simulator's flip swing vs its exact chain's g and the engine's
    # g, and its home win rate after the original call vs the engine WP (calibration table), for both simulators
    full, _ = sim.load(v)
    cntf = (full.balls * 3 + full.strikes).values
    pidx = np.argsort(cntf, kind="stable").astype(np.int64)
    pcnt = np.bincount(cntf, minlength=12).astype(np.int64); poff = np.concatenate([[0], np.cumsum(pcnt)[:-1]]).astype(np.int64)
    orig = full.orig_strike.values.astype(np.bool_)
    rng = np.random.default_rng(5); keys = rows[rng.choice(len(rows), size=min(n_mc, len(rows)), replace=False)]
    wpo = np.where(m.orig_strike.values, m.wp_strike.values, m.wp_ball.values)[keys]
    res["mc_check"] = {}
    for level, chain in (("sim", "sim"), ("run", "run")):
        ev, kip, br = causal.event_model(level)
        sw, wo = np.zeros(len(keys)), np.zeros(len(keys)); t = time.time()
        for j, i in enumerate(keys):
            a_ = [causal.mc_win(K_mc, 99 + 1000 * j, int(m.inning.values[i]), int(not m.top.values[i]), int(m.outs_when_up.values[i]),
                                int(m.bases.values[i]), int(m.score_diff_home.values[i]), int(m.balls.values[i]), int(m.strikes.values[i]),
                                pend, ev, kip, br, poff, pcnt, pidx, orig) for pend in (1, 0)]
            # signed by the engine's direction (abs of a noisy difference would be biased upward)
            sw[j] = np.sign(m.wp_ball.values[i] - m.wp_strike.values[i]) * (a_[1] - a_[0]); wo[j] = a_[0] if m.orig_strike.values[i] else a_[1]
        gc, ge = g_row[chain][keys], g_row["engine"][keys]
        cal = []
        for lo in np.arange(0, 1, .1):
            k_ = (wpo >= lo) & (wpo < lo + .1)
            if k_.sum(): cal.append(dict(bin=f"{lo:.1f}-{lo + .1:.1f}", n=int(k_.sum()), engine=round(float(wpo[k_].mean()), 4), sim=round(float(wo[k_].mean()), 4)))
        res["mc_check"][level] = dict(K=K_mc, n=len(keys), secs=round(time.time() - t), mean_swing_mc=float(sw.mean()), mean_g_chain=float(gc.mean()),
                                      mean_g_engine=float(ge.mean()), mc_minus_chain=float((sw - gc).mean()),
                                      mc_minus_chain_se=float((sw - gc).std() / np.sqrt(len(keys))),
                                      mc_over_engine=float(sw.sum() / ge.sum()), chain_over_engine=float(gc.sum() / ge.sum()),
                                      home_win_mc_minus_engine=float((wo - wpo).mean()),
                                      home_win_mc_minus_engine_se=float((wo - wpo).std() / np.sqrt(len(keys))), calibration=cal)
        print(level, {k_: x for k_, x in res["mc_check"][level].items() if k_ != "calibration"}, flush=True)
    json.dump(res, open("results/causal_gap.json", "w"), indent=1)
    write_md(res)
    return res


def write_md(g):
    """results/causal_gap.md from results/causal_gap.json, causal.json (simulator "run") and causal_sim.json."""
    cr, cs = json.load(open("results/causal.json")), json.load(open("results/causal_sim.json"))
    b, bs = cr["game_bootstrap"], cs["game_bootstrap"]
    o = g["by_policy"]["opt_score"]; pr = o["predicted"]; mc = g["mc_check"]; hf = g["home_field"]
    sg = lambda x: f"{x:+.3f}"
    c3 = lambda x: f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"
    c2 = lambda x: f"{x[0]:.2f} [{x[1]:.2f}, {x[2]:.2f}]"
    L = ["## Audit R1: why the binary causal estimate sat below the RB one", "",
         "### Verdict", "",
         f"**A simulator bug, now fixed, not a property of the game or of the headline accounting.** The earlier pitch "
         f"simulator in `src/causal.py` moved a team's win probability by only **{100 * mc['sim']['chain_over_engine']:.1f} %** of the "
         f"engine's g when a call was flipped. The binary estimator values the branch flip in simulated wins, so it "
         f"inherited that shortfall: {c3(bs['opt_score']['binary_minus_rb'])} wins per team-season for the optimum "
         f"(earlier run, R = {cs['R']}, K = {cs['K']}). Pricing every branch with the old simulator's own exact g predicts "
         f"**{sg(pr['sim'])}**, which accounts for the whole measured gap with no free parameter. The corrected simulator reaches "
         f"{100 * mc['run']['chain_over_engine']:.1f} % of g. On it binary − RB is {c3(b['opt_score']['binary_minus_rb'])} for the optimum "
         f"(R = {cr['R']}, K = {cr['K']}), against {sg(pr['run'])} predicted, and it averages "
         f"{np.mean([b[p]['binary_minus_rb'][0] for p in b if p != 'observed']):+.3f} over the nine policies "
         f"(was {np.mean([bs[p]['binary_minus_rb'][0] for p in bs if p != 'observed']):+.3f}).", "",
         f"**The headline keeps the path-sum accounting priced in engine g.** The RB causal estimate agrees with it "
         f"({c3(b['opt_score']['causal_minus_path'])}). Measured in simulated wins (binary), teams lose "
         f"{c2(b['opt_score']['binary_minus_observed'])}, against {c2(b['opt_score']['causal_minus_observed'])} for RB; before the fix "
         f"the binary figure was {c2(bs['opt_score']['binary_minus_observed'])}. The reasons to keep the engine are:", "",
         "1. The engine is the calibrated object. Its PA kernel is the empirical end-of-PA distribution from every 2026 pitch, and it matches "
         "realised results as well as Statcast's own win expectancy (`results/wp_calibration.json`). The simulator is a "
         "generative approximation of the same data, used only to replay games after a flip.",
         f"2. With the simulator fixed, the two estimators differ by {b['opt_score']['binary_minus_rb'][0]:+.3f} for the optimum. "
         f"{pr['run']:+.3f} of that is the residual simulator mismatch predicted by the exact chain, and the rest is within the CI "
         f"half-width ({(b['opt_score']['binary_minus_rb'][2] - b['opt_score']['binary_minus_rb'][1]) / 2:.3f}). Across the nine policies the "
         "difference averages "
         f"{np.mean([b[p]['binary_minus_rb'][0] for p in b if p != 'observed']):+.3f}. The binary estimator adds win-indicator noise "
         "and no information.",
         "3. Any residual mismatch between the simulator and the engine biases the binary estimate and not the path-sum. The residual "
         f"is {100 * (1 - mc['run']['chain_over_engine']):.1f} % of g, predicted {sg(pr['run'])} wins.", "",
         "### Where the gap came from", "",
         "`src/causal_gap.py` writes every simulator as a pitch-level Markov chain and computes its PA kernel, its WP table "
         "and its g exactly, with no Monte Carlo. The simulator itself (`causal.mc_win`, K = "
         f"{mc['sim']['K']} paths from {mc['sim']['n']} branch states) reproduces its chain's flip swing to "
         f"{mc['sim']['mc_minus_chain']:+.4f} ± {mc['sim']['mc_minus_chain_se']:.4f} (old) and {mc['run']['mc_minus_chain']:+.4f} ± "
         f"{mc['run']['mc_minus_chain_se']:.4f} (new). The decomposition is therefore exact up to the second-order term "
         "(g mismatch at the team's later flips, A − A').", "",
         f"Predicted binary − RB for the optimum, wins per team-season, adding one ingredient at a time "
         f"({o['branches_per_season']:.0f} branch flips per team-season, mean engine g {o['mean_g_branch_engine']:.4f}):", "",
         "| Simulator | Events | In-play outcome | Between-pitch baserunning | g / engine g | Predicted binary − RB | Step |",
         "|---|---|---|---|---|---|---|"]
    steps = [("sim (earlier)", "by count", "by the PA's starting outs/bases", "none"),
             ("kc", "by count", "by count, outs, bases", "none"), ("kcob", "by count, outs, bases", "by count, outs, bases", "none"),
             ("run", "by count, outs, bases", "by count, outs, bases", f"yes ({100 * g['p_move_between_pitches']:.1f} % of pitches)")]
    prev = None
    for (lab, e, k, r), s_ in zip(steps, STEPS):
        ratio = mc[s_]["chain_over_engine"] if s_ in mc else None
        L.append(f"| {lab} | {e} | {k} | {r} | {f'{100 * ratio:.1f} %' if ratio else '—'} | {sg(pr[s_])} | "
                 f"{'—' if prev is None else sg(pr[s_] - pr[prev])} |")
        prev = s_
    L += [f"| engine (`wp.py`) | empirical end-of-PA kernel by count, outs, bases | | (included) | 100 % | 0 | {sg(-pr['run'])} |", "",
          "Measured binary − RB: "
          f"{c3(bs['opt_score']['binary_minus_rb'])} with the earlier simulator, {c3(b['opt_score']['binary_minus_rb'])} with the corrected one.", "",
          "Reading the steps:",
          f"- **Count-dependent contact** (sim → kc) closes {100 * (pr['kc'] - pr['sim']) / -pr['sim']:.0f} % of the gap. Batters ahead in the count hit for more damage "
          "(runs scored on the ball in play, averaged over the 24 base-out states: "
          + ", ".join(f"{c} {v:.3f}" for c, v in g["mean_inplay_runs_by_count"].items() if c in ("0-2", "1-2", "0-0", "2-0", "3-1")) + "). "
          "A count-blind in-play kernel compresses the value of moving from a strike count to a ball count. The old kernel was also "
          "indexed by the state at the PA's first pitch, and its runs included runners who moved during the PA.",
          f"- **Events by outs and bases** (kc → kcob) moves the other way ({100 * (pr['kcob'] - pr['kc']) / -pr['sim']:+.0f} %): pitch mix and swing rates depend on the base-out state.",
          f"- **Baserunning between pitches** (kcob → run) closes {100 * (pr['run'] - pr['kcob']) / -pr['sim']:.0f} %. Steals, wild pitches and pickoffs happen on about 1 % of pitches. "
          "Their chance accumulates over the extra pitches a ball buys, and the engine's empirical kernel contains them.", "",
          "By count of the branch pitch, the old simulator under-priced early-count flips most (g / engine g − 1):", "",
          "| Count | n | sim | kc | kcob | run |", "|---|---|---|---|---|---|"]
    L += [f"| {r['count']} | {r['n']} | {100 * r['rel_sim']:+.1f} % | {100 * r['rel_kc']:+.1f} % | {100 * r['rel_kcob']:+.1f} % | {100 * r['rel_run']:+.1f} % |"
          for r in g["opt_score_by_count"]]
    L += ["", "By inning and margin, the shortfall is spread over the whole game (share of the old gap in the last column):", "",
          "| Inning | Margin | n | engine g | sim | run | share of old gap |", "|---|---|---|---|---|---|---|"]
    L += [f"| {r['inning']} | {r['margin']} | {r['n']} | {r['g_engine']:.4f} | {100 * r['rel_sim']:+.1f} % | {100 * r['rel_run']:+.1f} % | {100 * r['share_of_gap_sim']:.0f} % |"
          for r in g["opt_score_by_inning_margin"]]
    L += ["", "### The other candidates", "",
          f"- **WP calibration of the simulator (levels).** From {mc['sim']['n']} branch states, the simulated home win rate "
          f"after the original call matches the engine WP to {mc['sim']['home_win_mc_minus_engine']:+.4f} ± "
          f"{mc['sim']['home_win_mc_minus_engine_se']:.4f} (old) and {mc['run']['home_win_mc_minus_engine']:+.4f} ± {mc['run']['home_win_mc_minus_engine_se']:.4f} "
          "(new) in every WP decile (`calibration` in `results/causal_gap.json`). This is why the audit's B-world check looked fine: "
          "the defect was in the **swing** of a flip, not in the level. The B-world offset of about −0.012 is the team giving up its "
          "challenges while the opponent keeps its own. It is the same at home "
          f"({cr['point']['opt_score']['calib_winB_minus_engine_home']:+.4f}) and away ({cr['point']['opt_score']['calib_winB_minus_engine_away']:+.4f}).",
          f"- **Missing home-field term.** Neither the engine nor the simulator has one, so it cannot separate them. Re-estimating the "
          f"engine with one PA kernel per batting team gives a home WP of {hf['home_wp_first_pitch']:.3f} at the first pitch and changes g "
          f"on the optimum's branch flips by {100 * hf['rel_g_opt_branches']:+.1f} % and on the observed successes by "
          f"{100 * hf['rel_g_observed_successes']:+.1f} %. The first-order effect on wins lost is {hf['first_order_change_in_wins_lost']:+.3f} (audit R5 closed: negligible).",
          f"- **Extra innings.** Both use the runner on second and the same top-up rule. Branches in extra innings contribute "
          f"{sg(o['predicted_extra_innings']['sim'])} of the old {sg(pr['sim'])}.",
          "- **The branch point (first successful challenge only).** Path-sum and both causal estimators share the recorded path up to "
          f"the branch by construction, so it cannot create a binary − RB difference. Its own effect is RB − path = "
          f"{c3(b['opt_score']['causal_minus_path'])}.",
          "- **Common random numbers.** Worlds A, A′ and B share seeds per (branch, k), which lowers the variance of A − A′. Both "
          "estimators are means of i.i.d. continuations, so CRN cannot bias them.",
          f"- **K / R Monte Carlo bias.** Both estimators are linear in sample means, hence unbiased in K and R. The old gap's CI "
          f"excluded 0, and the exact chain reproduces it ({sg(pr['sim'])} predicted vs {bs['opt_score']['binary_minus_rb'][0]:+.3f} measured), "
          "so it was not noise.", "",
          "### Recommended manuscript wording", "",
          "*§5 (causal check), one paragraph:* "
          "\"Summing g along the recorded game ignores that, after a flipped call, the rest of the game would have been "
          "different. We therefore replay every team-game from its first successful challenge in a pitch-level simulator: "
          "events and in-play outcomes by count, outs and bases, plus baserunning between pitches, all estimated from 2026 "
          "Statcast. The flipped call is compared with the original one, holding everything else fixed (common random "
          "numbers). The simulator reproduces the engine's win probabilities in every decile, and a flip moves its win "
          f"probability by \\SimGShareRun\\,\\% of g. Valued this way, the optimum's value changes by \\CausalMinusPath{{}} "
          f"\\CausalMinusPathCI{{}} wins per team-season relative to the path-sum, and teams lose \\CausalLost{{}} \\CausalLostCI. Using simulated "
          f"win indicators instead of g gives \\BinaryLost{{}} \\BinaryLostCI; the difference, \\BinaryMinusRB{{}} \\BinaryMinusRBCI, is "
          f"about the size the remaining \\SimGShareRunGap\\,\\% mismatch between simulator and engine predicts (\\GapPredRun). A simpler simulator that ignores the count when drawing contact and has no "
          f"baserunning moves win probability by only \\SimGShare\\,\\% of g, and for exactly that reason its win-indicator estimate "
          "differs from g-based accounting by \\BinaryMinusRBSim{} (Supplement~S\\ref{sec:causal-gap}).\"", "",
          "*Supplement section \"Engine versus simulator\":* the decomposition table above; the count and inning tables; the "
          "home-field paragraph; the Monte Carlo validation of the exact chains.", ""]
    open("results/causal_gap.md", "w").write("\n".join(L))


if __name__ == "__main__":
    if sys.argv[1:2] == ["md"]:
        write_md(json.load(open("results/causal_gap.json")))
    else:
        K = int(sys.argv[1]) if len(sys.argv) > 1 else 2000; n = int(sys.argv[2]) if len(sys.argv) > 2 else 600
        main(K, n)
