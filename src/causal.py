"""Causal wins (Arora's objection): re-simulate the game after a flipped call instead of summing WP swings
along the recorded continuation.

Path-sum accounting (sim.py, headline.py): a team's value = sum of g over its successful challenges, every g read off
the *recorded* game state, although after the first flip the recorded continuation is no longer the game that would
have been played.

Anchored re-simulation.  For each team-game and policy, replay the recorded called-pitch stream until the team's
first successful challenge under the policy (before it, the policy and the recorded game share the same path -- the
same assumption the path-sum makes).  At that pitch branch into
  A: the call is flipped; the game continues in a pitch-level simulator, the team follows the policy with the
     challenges it holds, the opponent follows its fitted behavioural rule (s > kappa) with its recorded remaining
     challenges;
  B: the original call stands, the team never challenges again, the opponent as in A.
  A': the original call stands and the team keeps following the policy (control; common random numbers with A).
The path-sum value of a team-game is g_branch + L_rec, L_rec = sum of g over the team's later successes on the
recorded continuation, an unbiased draw of the no-flip continuation value.  The causal value is g_branch + E[L | flip].
Causal correction (WP points, summed over team-games like the path-sum):
  RB      causal = path + E[(L_A - O_A) - (L_A' - O_A')],  L (O) = sum of g over the team's (the opponent's) later
          successful flips, g at the simulated state;
  binary  causal = path + E[win_A - win_A'] - g_branch  (win indicators; E[win_A - win_A'] = g_branch + E[(L_A - O_A) -
          (L_A' - O_A')] when the WP engine is the simulator's value function -- the win indicator is a martingale whose
          jumps at either team's flips are +-g -- so RB is the low-variance version and binary the check).
  in earlier versions RB summed the team's own later flips only (kept as causal_rb_own);
  a branch also changes the opponent's later opportunities and gains, which the signed version includes.
Scope: the branch point is the FIRST DIVERGENCE between the policy's game and
the recorded game, the first pitch where exactly one of them flips the call (policies.first_divergence). Before it
the two paths coincide. If the policy flips a call the team left standing, A (flip) is the policy's world and A' the
recorded one, as above; if the team flipped a call the policy leaves standing, A is the recorded world and A' the
policy's, and the correction changes sign. In earlier versions the branch was the policy's first success, also when the
team had flipped the same call (then the recorded continuation is already the flip world), and recorded overturns the
policy would have skipped were never branched on.
Differencing against A' cancels any level bias of the simulator's opportunity stream; E[L_A'] vs L_rec is reported as
a fidelity check, and the B world (original call, team never challenges again) checks the simulator's win
probability against the WP engine.  For the observed policy the recorded path *is* the realised game, so its
path-sum is already the causal value.

Simulator: count-level pitch outcomes from 2026 Statcast (called / swinging-or-foul-tip strike / foul / in play / HBP
/ automatic ball), in-play PA outcome kernel (outs, bases, runs) by (outs, bases), called pitches drawn from the 2026
called-pitch pool of the same count (location -> truth W, signal s, posterior pt, fitted kappa of that draw), runner on
second in extra innings, one challenge restored to a team at 0 at each extra inning, walk-off rules.  Mid-PA
baserunning (steals, wild pitches) is not modelled; the sim-vs-engine calibration is reported.
The original simulator (level "sim") drew events by count only and in-play outcomes independently of the
count, with no baserunning between pitches; its flip swing was ~3 % below the engine's g, which is what separated
the binary from the RB estimate (audit R1, results/causal_gap.md).  The default level "run" draws events and in-play
outcomes by (count, outs, bases) and adds between-pitch baserunning.
usage: python src/causal.py [R] [K] [level=run|sim]  -> results/causal[_sim].json, data/causal_pergame[_sim].pkl"""
import json, pickle, sys, time
import numpy as np, pandas as pd
from numba import njit, prange
sys.path.insert(0, "src")
from policies import decide
from causal_gap import CAT

DCLIP = 10
CAUSAL_BOOT = 2000            # game resamples of the per-game outputs at fixed fitted parameters (not \BootB)
POLICIES = ["opt_score", "opt", "stake_0.008", "stake_0.01", "savant", "obs_rule", "card_refit", "card_pub", "runs_dp"]


def event_model(level="run"):
    """cumulative event probabilities ev[count, outs, bases, 6] (called, other strike, foul, in play, HBP, automatic
    ball), in-play kernel kip[count, outs, bases, 160] over (outs_end, bases_end, runs) = 40 oe + 5 be + r, and the
    between-pitch baserunning kernel br[outs, bases, 160] (causal_gap.kernels).
    level "sim": the original simulator (events by count, in-play by the PA's starting outs/bases, no baserunning);
    "run": events and in-play outcomes by (count, outs, bases) plus baserunning, whose WP is the engine's
    up to 2-3 % in g (results/causal_gap.json)."""
    import causal_gap
    k = causal_gap.kernels()
    if level == "sim":
        ev, kp, br = k["ev_c"], k["k_ob"], None
    else:
        ev, kp, br = k["ev_cob"], k["k_cob"], k["M"]
    if br is None:
        br = np.zeros((3, 8, 160))
        for o in range(3):
            for b in range(8):
                br[o, b, o * 40 + b * 5] = 1.0
    ev, kp, br = np.cumsum(ev, -1), np.cumsum(kp, -1), np.cumsum(br, -1)
    ev[..., -1] = 1.0; kp[..., -1] = 1.0; br[..., -1] = 1.0
    return ev, kp, br


def stake_tables():
    """g and RV for every pre-pitch state [inn 1..10, half, outs, bases, diff +-DCLIP, balls, strikes], cached."""
    try:
        z = np.load("data/gtable.npz"); return z["G"], z["RV"]
    except FileNotFoundError:
        import stakes
        G = np.zeros((10, 2, 3, 8, 2 * DCLIP + 1, 4, 3)); RV = np.zeros_like(G)
        for i in range(10):
            for h in range(2):
                for o in range(3):
                    for b in range(8):
                        for d in range(-DCLIP, DCLIP + 1):
                            for bl in range(4):
                                for st in range(3):
                                    wb, rb = stakes.Vcount(i + 1, h, o, b, d, bl + 1, st)
                                    ws, rs = stakes.Vcount(i + 1, h, o, b, d, bl, st + 1)
                                    # Signed stake oriented to the batting team, as stakes.add_stakes
                                    G[i, h, o, b, d + DCLIP, bl, st] = max((1 if h else -1) * (wb - ws), 0.0); RV[i, h, o, b, d + DCLIP, bl, st] = max(rb - rs, 0.0)
        np.savez("data/gtable.npz", G=G, RV=RV)
        return G, RV


@njit(cache=True)
def _walk(bases):
    if not bases & 1: return bases | 1, 0
    if not bases & 2: return bases | 3, 0
    if not bases & 4: return 7, 0
    return 7, 1


@njit(cache=True)
def _slot(inn, half):
    return (inn - 1) * 2 + half if inn <= 9 else 18


@njit(cache=True)
def play(inn, half, outs, bases, diff, balls, strikes, pend, c_home, c_away, t, pol, kin_t, dbin_t,
         D, Cn, card, q1, q2, dconst, D4, ev, kip, br, poff, pcnt, pidx, orig, isopp, Wp, ptp, sp, kp, G, RVt, maxinn, DM):
    """Play one continuation to the end of the game, starting with the branch pitch's call pend (1 strike, 0 ball).
    pol < 0: team t never challenges. Returns (win of team t, 0.5 if unresolved; sum of g over t's later flips; sum of
    g over the opponent's later flips)."""
    c = np.array([c_away, c_home]); rb = 0.0; ob = 0.0
    kind = 1 if pend == 1 else 0
    first = True; ran = False
    while True:
        runs = 0; pa_end = False; cnt = balls * 3 + strikes
        moved = False
        if not first and not ran:                                   # baserunning before the pitch
            u = np.random.random(); k = 0
            while k < 159 and u > br[outs, bases, k]:
                k += 1
            if k != outs * 40 + bases * 5:
                moved = True; ran = True
                outs = k // 40; bases = (k % 40) // 5 if outs < 3 else 0; runs = k % 5
        if moved:
            kind = -1
        elif not first:
            ran = False
            u = np.random.random(); kind = 0
            while kind < 5 and u > ev[cnt, outs, bases, kind]:
                kind += 1
            if kind == 0:                                           # taken pitch: draw a called pitch of this count
                j = pidx[poff[cnt] + np.random.randint(pcnt[cnt])]
                strike = orig[j]
                if isopp[j]:
                    ch_team = (1 if half == 1 else 0) if strike else (0 if half == 1 else 1)
                    go = False
                    di = min(max(diff, -10), 10) + 10; ii = min(inn, 10) - 1
                    gcur = G[ii, half, outs, bases, di, balls, strikes]
                    if ch_team == t:
                        if c[t] > 0 and pol >= 0:
                            go = decide(pol, ptp[j], sp[j], kp[j, min(c[t], 2)], gcur, RVt[ii, half, outs, bases, di, balls, strikes],
                                        _slot(inn, half), kin_t, c[t], inn, balls, strikes, D, Cn, card, q1, q2, dconst,
                                        D4, dbin_t)
                        kin_t += 1
                    elif c[ch_team] > 0:
                        go = sp[j] > kp[j, min(c[ch_team], 2)]
                    if go:
                        if Wp[j]:
                            strike = not strike
                            if ch_team == t:
                                rb += gcur
                            else:
                                ob += gcur
                        else:
                            c[ch_team] -= 1
                kind = 1 if strike else 0
            elif kind == 5:
                kind = 0                                            # automatic ball / pitchout
        first = False
        if kind == 0:
            balls += 1
            if balls == 4:
                bases, runs = _walk(bases); pa_end = True
        elif kind == 1:
            strikes += 1
            if strikes == 3:
                outs += 1; pa_end = True
        elif kind == 2:
            if strikes < 2:
                strikes += 1
        elif kind == 3:
            u = np.random.random(); k = 0
            while k < 159 and u > kip[cnt, outs, bases, k]:
                k += 1
            outs = k // 40; bases = (k % 40) // 5; runs = k % 5; pa_end = True
        elif kind == 4:                                             # hit by pitch
            bases, runs = _walk(bases); pa_end = True
        if pa_end:
            balls = 0; strikes = 0
        diff += runs if half == 1 else -runs
        if half == 1 and inn >= 9 and diff > 0:
            break                                                   # walk-off
        if outs >= 3:
            old = _slot(inn, half)
            if half == 0:
                if inn >= 9 and diff > 0:
                    break
                half = 1
            else:
                if inn >= 9 and diff != 0:
                    break
                inn += 1; half = 0
                if inn > maxinn:
                    break
                if inn >= 10:
                    for q in range(2):
                        if c[q] == 0:
                            c[q] = 1
            outs = 0; balls = 0; strikes = 0; bases = 2 if inn >= 10 else 0; ran = False
            dbin_t = min(max(diff, -DM), DM) + DM
            if _slot(inn, half) != old:
                kin_t = 0
    if diff == 0:
        return 0.5, rb, ob
    win = (diff > 0) if t == 1 else (diff < 0)
    return (1.0 if win else 0.0), rb, ob


@njit(cache=True, parallel=True)
def branches(rows, teams, cts, copp, flip, inn, half, outs, bases, diff, balls, strikes, kin, dbin, pol,
             Dh, Da, Cnh, Cna, card, q1, q2, dconst, D4h, D4a, ev, kip, br, poff, pcnt, pidx, orig, isopp, Wp, ptp, sp, kp,
             G, RVt, K, seed, DM):
    """K continuations in worlds A (flip, policy), A' (no flip, policy) and B (no flip, never challenge) for every
    branch, common random numbers across worlds. Returns means over K: winA, LA, winA', LA', winB, OA, OA' (O = sum of g
    over the opponent's later flips)."""
    n = len(rows); out = np.zeros((n, 7))
    for b in prange(n):
        i = rows[b]; t = teams[b]
        D = Dh if t == 1 else Da; Cn = Cnh if t == 1 else Cna; D4 = D4h if t == 1 else D4a
        ch_home = cts[b] if t == 1 else copp[b]; ch_away = copp[b] if t == 1 else cts[b]
        for k in range(K):
            for world in range(3):
                np.random.seed(seed + 100003 * b + k)
                pend = flip[i] if world == 0 else 1 - flip[i]
                pw = pol if world < 2 else -1
                w, r, ob = play(inn[i], half[i], outs[i], bases[i], diff[i], balls[i], strikes[i], pend, ch_home, ch_away,
                            t, pw, kin[i] + 1, dbin[i], D, Cn, card, q1, q2, dconst, D4, ev, kip, br,
                            poff, pcnt, pidx, orig, isopp, Wp, ptp, sp, kp, G, RVt, 25, DM)
                if world == 0:
                    out[b, 0] += w / K; out[b, 1] += r / K; out[b, 5] += ob / K
                elif world == 1:
                    out[b, 2] += w / K; out[b, 3] += r / K; out[b, 6] += ob / K
                else:
                    out[b, 4] += w / K
    return out


@njit(cache=True, parallel=True)
def mc_win(K, seed, inn, half, outs, bases, diff, balls, strikes, pend, ev, kip, br, poff, pcnt, pidx, orig):
    """home win rate of K continuations from one pitch state with call pend and no challenges at all (checks the simulator against the exact chain of causal_gap.py and the WP engine)."""
    w = np.zeros(K)
    z2 = np.zeros((1, 1)); zc = np.zeros((4, 3, 2, 2)); z3 = np.zeros((1, 1, 1)); z4 = np.zeros((1, 1, 1, 1))
    nob = np.zeros(len(orig), np.bool_); zf = np.zeros(len(orig)); zk = np.zeros((len(orig), 3)); G = np.zeros((10, 2, 3, 8, 21, 4, 3))
    for k in prange(K):
        np.random.seed(seed + k)
        w[k], _, _ = play(inn, half, outs, bases, diff, balls, strikes, pend, 0, 0, 1, -1, 0, 0, z3, z2, zc, 0.0, 0.0, 0.0, z4,
                       ev, kip, br, poff, pcnt, pidx, orig, nob, nob, zf, zf, zk, G, G, 25, 1)
    return w.mean()


def run(R=3, K=40, level="run"):
    import sim, dp_score
    from headline import evaluate_all, spec_variant, MAIN
    v = spec_variant(MAIN)
    t0 = time.time()
    m, kin, out, draws = evaluate_all(v, R=R, seed=1, keep_draws=True)
    full, _ = sim.load(v); full = dp_score.add_dstart(full)
    mask = sim.opportunity_mask(full)                # The policies' opportunity set (headline.evaluate_all)
    assert mask.sum() == len(m)
    import policies as PP
    recsucc = (m.ch.values & m.ch_over.values).astype(np.bool_)   # the team flipped this call in 2026
    base = (m.gidx.values, m.is_home_team.values, m.new_extra.values, m.slot.values.astype(np.int64), kin.astype(np.int64),
            m.inning.values.astype(np.int64), m.balls.values.astype(np.int64), m.strikes.values.astype(np.int64),
            m.dbin.values.astype(np.int64))
    ev, kip, br = event_model(level); G, RVt = stake_tables()
    cntf = (full.balls * 3 + full.strikes).values
    pidx = np.argsort(cntf, kind="stable").astype(np.int64)
    pcnt = np.bincount(cntf, minlength=12).astype(np.int64); poff = np.concatenate([[0], np.cumsum(pcnt)[:-1]]).astype(np.int64)
    orig = full.orig_strike.values.astype(np.bool_); kp = sim.kappa_by_count(full)      # [n, 3]: criterion by challenges held
    I = lambda x: np.ascontiguousarray(np.asarray(x).astype(np.int64))
    inn, half, outs, bases = I(m.inning), I(~m.top.values), I(m.outs_when_up), I(m.bases)
    diff, balls, strikes, dbin = I(m.score_diff_home), I(m.balls), I(m.strikes), I(m.dbin)
    flip = I(~m.orig_strike.values)                                # flipped call: strike iff the original was a ball
    ng = int(m.gidx.max() + 1); gid = m.gidx.values
    # sim-vs-engine calibration: team's engine WP right after the original call of the branch pitch
    wp_orig_home = np.where(m.orig_strike.values, m.wp_strike.values, m.wp_ball.values)
    res, per = {}, {}
    for pol in POLICIES:
        acc = {k: np.zeros(ng) for k in ("path", "rb", "rb_own", "binary", "later_rec", "later_A", "later_A0", "opp_A", "opp_A0")}
        acc["n_branch"] = 0.0; cal_s, cal_e = [], []; brs = []
        for r, dr in enumerate(draws):
            pc, dc, card = dr["pols"][pol]
            # Branch at the FIRST DIVERGENCE between the policy's game and the
            # recorded one (policies.first_divergence): +1 the policy flips a call the team left standing (A = flip is
            # the policy's world, A' the recorded one), -1 the team flipped a call the policy leaves standing (A = the
            # recorded world, A' the policy's). It was the policy's first success, also when the team had flipped the
            # same call, where the recorded continuation is already the flip world.
            q1, q2 = dr["q"]
            first, typ, firstc, pre = PP.first_divergence(*base, dr["W"], dr["pt"], dr["s"], m.g.values, m.RV.values,
                                                          sim.kappa_by_count(m), pc, dr["Dh"], dr["Da"], dr["Cnh"], dr["Cna"],
                                                          card, q1, q2, dc, dr["D4h"], dr["D4a"], recsucc, ng)
            gg, tt = np.nonzero(first >= 0)
            rows = first[gg, tt].astype(np.int64); sgn = typ[gg, tt].astype(float); pol_flip = sgn > 0
            copp = np.where(tt == 1, m.left_away.values[rows], m.left_home.values[rows]).astype(np.int64)
            Wp = np.zeros(len(full), np.bool_); Wp[mask] = dr["W"]
            ptp = np.zeros(len(full)); ptp[mask] = dr["pt"]
            sp = np.full(len(full), -1e9); sp[mask] = dr["s"]
            o = branches(rows, tt.astype(np.int64), firstc[gg, tt].astype(np.int64), copp, flip, inn, half, outs, bases,
                         diff, balls, strikes, kin, dbin, pc, dr["Dh"], dr["Da"], dr["Cnh"], dr["Cna"], card, q1, q2,
                         dc, dr["D4h"], dr["D4a"], ev, kip, br, poff, pcnt, pidx, orig, mask, Wp, ptp, sp, kp, G, RVt,
                         K, 7 + 7919 * r, dp_score.DM)
            gb = m.g.values[rows]
            path_gt = dr["vals"][pol].sum(2)                      # [game, team]
            acc["path"] += path_gt.sum(1) / R
            # fidelity check on the policy-flip branches: the recorded continuation is a draw of the no-flip world
            pf = pol_flip
            np.add.at(acc["later_rec"], gg[pf], (path_gt[gg[pf], tt[pf]] - pre[gg[pf], tt[pf]] - gb[pf]) / R)
            np.add.at(acc["later_A0"], gg[pf], o[pf, 3] / R); np.add.at(acc["later_A"], gg[pf], o[pf, 1] / R)
            np.add.at(acc["opp_A"], gg, sgn * o[:, 5] / R); np.add.at(acc["opp_A0"], gg, sgn * o[:, 6] / R)
            # correction = (policy world) - (recorded world); sgn = -1 swaps A and A'
            np.add.at(acc["rb_own"], gg, sgn * (o[:, 1] - o[:, 3]) / R)                 # Team's own later flips only
            np.add.at(acc["rb"], gg, sgn * (o[:, 1] - o[:, 3] - o[:, 5] + o[:, 6]) / R) # Signed, both teams
            np.add.at(acc["binary"], gg, sgn * (o[:, 0] - o[:, 2] - gb) / R)
            acc["n_skip"] = acc.get("n_skip", 0.0) + float((~pol_flip).sum()) / R
            acc["n_branch"] += len(rows) / R
            cal_s.append(o[:, 4]); cal_e.append(np.where(tt == 1, wp_orig_home[rows], 1 - wp_orig_home[rows]))
            brs.append(dict(r=r, game=gg, team=tt, row=rows, g=gb, out=o, sgn=sgn))
        acc["rb"] += acc["path"]; acc["rb_own"] += acc["path"]; acc["binary"] += acc["path"]
        per[pol] = {k: acc[k] for k in ("path", "rb", "rb_own", "binary", "later_rec", "later_A", "later_A0", "opp_A", "opp_A0")}
        per[pol]["branches"] = brs
        cs, ce = np.concatenate(cal_s), np.concatenate(cal_e)
        home = np.concatenate([b["team"] for b in brs]) == 1
        gap = np.concatenate([b["sgn"] * (b["out"][:, 0] - b["out"][:, 2] - b["g"] - b["out"][:, 1] + b["out"][:, 3] + b["out"][:, 5] - b["out"][:, 6])
                              for b in brs])
        res[pol] = dict(path=acc["path"].sum() / 30, causal_rb=acc["rb"].sum() / 30, causal_rb_own=acc["rb_own"].sum() / 30,
                        causal_binary=acc["binary"].sum() / 30, opp_later_flip=acc["opp_A"].sum() / 30, opp_later_noflip=acc["opp_A0"].sum() / 30,
                        later_recorded=acc["later_rec"].sum() / 30, later_sim_flip=acc["later_A"].sum() / 30,
                        later_sim_noflip=acc["later_A0"].sum() / 30, branches_per_team_game=acc["n_branch"] / (2 * ng),
                        skip_branches_per_team_game=acc.get("n_skip", 0.0) / (2 * ng),
                        calib_winB_minus_engine=float((cs - ce).mean()), calib_slope=float(np.polyfit(ce, cs, 1)[0]),
                        calib_winB_minus_engine_home=float((cs - ce)[home].mean()), calib_winB_minus_engine_away=float((cs - ce)[~home].mean()),
                        binary_minus_rb_home=float(gap[home].sum() / R / 30), binary_minus_rb_away=float(gap[~home].sum() / R / 30))
        print(pol, {k: round(x, 4) for k, x in res[pol].items()}, round(time.time() - t0), flush=True)
    per["observed"] = dict(path=out["observed"][0].sum((1, 2)))
    res["observed"] = dict(path=per["observed"]["path"].sum() / 30)
    # game bootstrap of per-game sums (Monte Carlo of K continuations is inside each game's value)
    rng = np.random.default_rng(11); idx = rng.integers(0, ng, size=(CAUSAL_BOOT, ng))
    ci = lambda x: [round(float(x.sum() / 30), 4), *np.round(np.percentile(x[idx].sum(1) / 30, [2.5, 97.5]), 4).tolist()]
    obs = per["observed"]["path"]
    boot = {}
    for pol in POLICIES:
        p = per[pol]
        boot[pol] = dict(path=ci(p["path"]), causal_rb=ci(p["rb"]), causal_rb_own=ci(p["rb_own"]), causal_binary=ci(p["binary"]),
                         opp_effect=ci(p["opp_A0"] - p["opp_A"]), rb_own_minus_path=ci(p["rb_own"] - p["path"]),
                         causal_minus_path=ci(p["rb"] - p["path"]), binary_minus_rb=ci(p["binary"] - p["rb"]),
                         path_minus_observed=ci(p["path"] - obs), causal_minus_observed=ci(p["rb"] - obs),
                         binary_minus_observed=ci(p["binary"] - obs), fidelity_noflip_minus_recorded=ci(p["later_A0"] - p["later_rec"]))
    boot["observed"] = dict(path=ci(obs))
    tag = "" if level == "run" else f"_{level}"
    json.dump(dict(R=R, K=K, variant=MAIN, simulator=level, boot_resamples=CAUSAL_BOOT, point=res, game_bootstrap=boot), open(f"results/causal{tag}.json", "w"), indent=1)
    pickle.dump(dict(per=per, games=m.game_pk.unique()), open(f"data/causal_pergame{tag}.pkl", "wb"))
    return res, boot


if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 3; K = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    level = sys.argv[3] if len(sys.argv) > 3 else "run"
    res, boot = run(R, K, level)
    print(json.dumps(boot, indent=1))
