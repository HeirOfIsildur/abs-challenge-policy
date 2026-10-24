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
  up to RB summed the team's own later flips only (kept as causal_rb_own);
  a branch also changes the opponent's later opportunities and gains, which the signed version includes.
Scope: the branch point is the FIRST DIVERGENCE between the policy's game and
the recorded game, the first pitch where exactly one of them flips the call (policies.first_divergence). Before it
the two paths coincide. If the policy flips a call the team left standing, A (flip) is the policy's world and A' the
recorded one, as above; if the team flipped a call the policy leaves standing, A is the recorded world and A' the
policy's, and the correction changes sign. Up to the branch was the policy's first success, also when the
team had flipped the same call (then the recorded continuation is already the flip world), and recorded overturns the
policy would have skipped were never branched on.
Differencing against A' cancels any level bias of the simulator's opportunity stream; E[L_A'] vs L_rec is reported as
a fidelity check, and the B world (original call, team never challenges again) checks the simulator's win
probability against the WP engine.  For the observed policy the recorded path *is* the realised game, so its
path-sum is already the causal value.

Simulator: count-level pitch outcomes from 2026 Statcast (called / swinging-or-foul-tip strike / foul / in play / HBP
/ automatic ball), in-play PA outcome kernel (outs, bases, runs) by (outs, bases), called pitches drawn from the 2026
called-pitch pool of the same count (location -> truth W, signal s, posterior pt, fitted kappa of that draw), runner on
second in extra innings, one challenge restored to a team at 0 at each extra inning, walk-off rules.
a drawn pool pitch keeps its location, truth and signal, but the fitted criterion and the
player's posterior are evaluated at the SIMULATED state: kappa = the row's criterion with its inning, log g and stake
hinge terms replaced by those at the simulated inning and stake g (kappa_at; the count, region and player terms are
row constants), and the posterior is read from the draw's calibration map of the cell of the simulated stake tercile
(pt_at, posterior_by_tercile, sim.draw return_maps). They were the historical row's (its inning and stake), so the
opponent's rule and obs_rule used the wrong criterion and every p-based policy a posterior from the wrong cell. Also
after a half-inning change the score-aware DP's dbin now follows dp_score.add_dstart (extra top halves: topx =
DM + 1; half_start_dbin), and the opportunity index restarts in every half-inning (it ran on across extra halves,
which all share slot 18).
Full contrast: RB / binary above follow the policy in both A and A', so only the FIRST
divergence is re-simulated; later divergences are still credited on the recorded path. full_branches plays, from the
first divergence, world P (policy's call, then the policy) against world B (team's call, then its fitted behavioural
rule s > kappa at the simulated state), opponent on its rule in both; summed over team-games this is the policy's gain
over the fitted behavioural rule re-simulated end to end (full_vs_behaviour in results/causal.json: win count
E[win_P - win_B] and RB E[g_b (1{policy flips} - 1{team flipped}) + L_P - O_P - L_B + O_B]), to compare with the
path-sum loss against the simulated behavioural rule (path - path of obs_rule) and against the observed challenges.  Mid-PA
baserunning (steals, wild pitches) is not modelled; the sim-vs-engine calibration is reported.
the simulator (level "sim") drew events by count only and in-play outcomes independently of the
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
    level "sim": the simulator (events by count, in-play by the PA's starting outs/bases, no baserunning);
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
                                    # signed stake oriented to the batting team, as stakes.add_stakes
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
def half_start_dbin(inn, half, diff, DM):
    """the score-aware DP's dbin at the start of a half-inning, as dp_score.add_dstart encodes
    it: d + DM (d = home score difference clipped to +-DM) in regulation and in extra bottom halves (d <= 0 there), and
    dp_score.topx = DM + 1 for an extra-inning top half. play used d + DM for extra top halves too."""
    if inn >= 10 and half == 0:
        return DM + 1
    return min(max(diff, -DM), DM) + DM


@njit(cache=True)
def kfeat(g, inn, knots):
    """the state-dependent features of the fitted criterion (lik.design / gfeat / glow): log g
    (g clipped at 1e-4), clip(inning, 1, 10) / 9, the hinges (log g - k)+ at lik.G_KNOTS and the low-stake hinge
    min(log g - lik.G_LOW, 0); knots = [G_KNOTS..., G_LOW]."""
    lg = np.log(max(g, 1e-4)); f = np.zeros(6)
    f[0] = lg; f[1] = min(max(inn, 1), 10) / 9.0
    for q in range(3):
        f[2 + q] = max(lg - knots[q], 0.0)
    f[5] = min(lg - knots[3], 0.0)
    return f


@njit(cache=True)
def kappa_at(j, c, inn, g, kb, isoffp, kcoef, knots):
    """the fitted criterion of pool row j with c challenges held, at the simulated inning and
    stake g. kb[j, c] = kp[j, c] minus the row's own state terms (kappa_base); kcoef[side] (0 def, 1 off) holds the
    coefficients of kfeat. The count, region and player terms are row constants (the pool is drawn by count)."""
    f = kfeat(g, inn, knots); sd = 1 if isoffp[j] else 0; k = kb[j, c]
    for q in range(6):
        k += kcoef[sd, q] * f[q]
    return k


@njit(cache=True)
def pt_at(j, g, pt3, isoffp, cuts):
    """the player's posterior of pool row j in the calibration cell of the simulated stake
    tercile (sim.cal_groups: tercile = number of the side's cut points <= g); pt3[j, q] = row j's posterior under
    tercile q of its side."""
    sd = 1 if isoffp[j] else 0
    q = (1 if g >= cuts[sd, 0] else 0) + (1 if g >= cuts[sd, 1] else 0)
    return pt3[j, q]


@njit(cache=True)
def play(inn, half, outs, bases, diff, balls, strikes, pend, c_home, c_away, t, pol, kin_t, dbin_t,
         D, Cn, card, q1, q2, dconst, D4, ev, kip, br, poff, pcnt, pidx, orig, isopp, Wp, pt3, sp, kb, G, RVt, maxinn, DM,
         isoffp, kcoef, knots, cuts):
    """Play one continuation to the end of the game, starting with the branch pitch's call pend (1 strike, 0 ball).
    pol < 0: team t never challenges. Returns (win of team t, 0.5 if unresolved; sum of g over t's later flips; sum of
    g over the opponent's later flips, ).
    a drawn pool pitch j keeps its location, truth and signal, but its criterion and the
    player's posterior are re-evaluated at the simulated state (kappa_at, pt_at); they were row j's, i.e. at its
    recorded inning and stake."""
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
                    gcur = G[ii, half, outs, bases, di, balls, strikes]   # same g for either challenger (zero-sum WP)
                    if ch_team == t:
                        if c[t] > 0 and pol >= 0:
                            go = decide(pol, pt_at(j, gcur, pt3, isoffp, cuts), sp[j],
                                        kappa_at(j, min(c[t], 2), inn, gcur, kb, isoffp, kcoef, knots), gcur,
                                        RVt[ii, half, outs, bases, di, balls, strikes],
                                        _slot(inn, half), kin_t, c[t], inn, balls, strikes, D, Cn, card, q1, q2, dconst,
                                        D4, dbin_t)
                        kin_t += 1
                    elif c[ch_team] > 0:
                        go = sp[j] > kappa_at(j, min(c[ch_team], 2), inn, gcur, kb, isoffp, kcoef, knots)
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
            # the DP's extra-inning encoding (half_start_dbin), and the opportunity index restarts
            # in every half-inning (sim.kin_index); it was reset only when the slot changed, and all extra halves share
            # slot 18, so it ran on across extra innings
            dbin_t = half_start_dbin(inn, half, diff, DM)
            kin_t = 0
    if diff == 0:
        return 0.5, rb, ob
    win = (diff > 0) if t == 1 else (diff < 0)
    return (1.0 if win else 0.0), rb, ob


@njit(cache=True, parallel=True)
def branches(rows, teams, cts, copp, flip, inn, half, outs, bases, diff, balls, strikes, kin, dbin, pol,
             Dh, Da, Cnh, Cna, card, q1, q2, dconst, D4h, D4a, ev, kip, br, poff, pcnt, pidx, orig, isopp, Wp, pt3, sp, kb,
             G, RVt, K, seed, DM, isoffp, kcoef, knots, cuts):
    """K continuations in worlds A (flip, policy), A' (no flip, policy) and B (no flip, never challenge) for every
    branch, common random numbers across worlds. Returns means over K: winA, LA, winA', LA', winB, OA, OA' (O = sum of g
    over the opponent's later flips, )."""
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
                            poff, pcnt, pidx, orig, isopp, Wp, pt3, sp, kb, G, RVt, 25, DM, isoffp, kcoef, knots, cuts)
                if world == 0:
                    out[b, 0] += w / K; out[b, 1] += r / K; out[b, 5] += ob / K
                elif world == 1:
                    out[b, 2] += w / K; out[b, 3] += r / K; out[b, 6] += ob / K
                else:
                    out[b, 4] += w / K
    return out


@njit(cache=True, parallel=True)
def full_branches(rows, teams, cP, cB, copp, pendP, inn, half, outs, bases, diff, balls, strikes, kin, dbin, pol,
                  Dh, Da, Cnh, Cna, card, q1, q2, dconst, D4h, D4a, ev, kip, br, poff, pcnt, pidx, orig, isopp, Wp, pt3, sp,
                  kb, G, RVt, K, seed, DM, isoffp, kcoef, knots, cuts):
    """the FULL contrast from the first divergence, K continuations with common random
    numbers in world P (the call as the policy leaves it, the team then follows the policy, holding cP) and world B
    (the call as the team left it, the team then follows its fitted behavioural rule, pol 4, holding cB); the opponent
    follows its behavioural rule in both. Returns means over K: winP, LP, OP, winB, LB, OB.
    Why: one retained challenge that always succeeds; the recorded team never challenges;
    opportunities 1 (+0.1, later ones follow), 2 (+0.2, a challenge here ends the team's opportunities) and 3 (+0.1).
    A policy that always challenges gains 0.1 + 0.2 = 0.3. Path sum: 0.1 + 0.2 + 0.1 = 0.4. RB from the first
    divergence (opportunity 1), with the policy followed in both A and A': 0.4 + (0.5 - 0.4) - 0.1 = 0.4, because the
    second divergence (opportunity 2) is still credited on the recorded path. Full contrast: world P = 0.1 + 0.2 = 0.3,
    world B (team's rule, never challenges) = 0, gain 0.3 -- exact; pre-divergence values are equal by construction."""
    n = len(rows); out = np.zeros((n, 6))
    for b in prange(n):
        i = rows[b]; t = teams[b]
        D = Dh if t == 1 else Da; Cn = Cnh if t == 1 else Cna; D4 = D4h if t == 1 else D4a
        for k in range(K):
            for world in range(2):
                np.random.seed(seed + 100003 * b + k)
                ct = cP[b] if world == 0 else cB[b]
                ch_home = ct if t == 1 else copp[b]; ch_away = copp[b] if t == 1 else ct
                pend = pendP[b] if world == 0 else 1 - pendP[b]
                pw = pol if world == 0 else 4
                w, r, ob = play(inn[i], half[i], outs[i], bases[i], diff[i], balls[i], strikes[i], pend, ch_home, ch_away,
                                t, pw, kin[i] + 1, dbin[i], D, Cn, card, q1, q2, dconst, D4, ev, kip,
                                br, poff, pcnt, pidx, orig, isopp, Wp, pt3, sp, kb, G, RVt, 25, DM, isoffp, kcoef, knots,
                                cuts)
                out[b, 3 * world] += w / K; out[b, 3 * world + 1] += r / K; out[b, 3 * world + 2] += ob / K
    return out


@njit(cache=True, parallel=True)
def mc_win(K, seed, inn, half, outs, bases, diff, balls, strikes, pend, ev, kip, br, poff, pcnt, pidx, orig):
    """home win rate of K continuations from one pitch state with call pend and no challenges at all (    checks the simulator against the exact chain of causal_gap.py and the WP engine)."""
    w = np.zeros(K)
    z2 = np.zeros((1, 1)); zc = np.zeros((4, 3, 2, 2)); z3 = np.zeros((1, 1, 1)); z4 = np.zeros((1, 1, 1, 1))
    nob = np.zeros(len(orig), np.bool_); zf = np.zeros(len(orig)); zk = np.zeros((len(orig), 3)); G = np.zeros((10, 2, 3, 8, 21, 4, 3))
    zcoef = np.zeros((2, 6)); zkn = np.zeros(4); zcut = np.zeros((2, 2))
    for k in prange(K):
        np.random.seed(seed + k)
        w[k], _, _ = play(inn, half, outs, bases, diff, balls, strikes, pend, 0, 0, 1, -1, 0, 0, z3, z2, zc, 0.0, 0.0, 0.0, z4,
                       ev, kip, br, poff, pcnt, pidx, orig, nob, nob, zk, zf, zk, G, G, 25, 1, nob, zcoef, zkn, zcut)
    return w.mean()


def kappa_coefs(variant):
    """[2 (0 def, 1 off), 6] coefficients of kfeat in the fitted criterion (lik.unpack): beta of
    log g and of inning/9, the stake hinges kg (kappa_g) and the low-stake hinge kgl (kappa_glow); 0 where absent.
    Also returns knots = [lik.G_KNOTS..., lik.G_LOW]."""
    import lik
    th = np.asarray(variant["theta_override"]) if variant.get("theta_override") is not None else np.load(variant["theta"])
    P = lik.unpack(th, variant["spec"]); co = np.zeros((2, 6))
    for sd, side in ((0, "def"), (1, "off")):
        q = P[side]; b = np.asarray(q["beta"], float)
        co[sd, 0], co[sd, 1] = b[1], b[3]
        if "kg" in q: co[sd, 2:5] = np.asarray(q["kg"], float)
        if "kgl" in q: co[sd, 5] = float(q["kgl"])
    return co, np.array([*lik.G_KNOTS, lik.G_LOW], float)


def kappa_base(kp, g, inn, isoffp, kcoef, knots):
    """kp [n, 3] minus each row's own state terms, so that kappa_at(j, c, inn_j, g_j) = kp[j, c]."""
    kb = kp.copy()
    for j in range(len(kb)):
        f = kfeat(g[j], inn[j], knots)
        kb[j] -= kcoef[1 if isoffp[j] else 0] @ f
    return np.ascontiguousarray(kb)


def posterior_by_tercile(m, s, pt, maps):
    """[n, 3] the posterior of every opportunity row under each stake tercile of its side, from
    the draw's calibration maps (sim.draw return_maps; cells sim.cal_groups = <pgrp>_<count class>_<tercile>). A cell
    absent from the data (no map) keeps the row's own posterior. Column q = the row's own tercile reproduces pt."""
    n = len(m); pt3 = np.repeat(pt[:, None], 3, 1)
    if maps is None:                                          # flat prior: the posterior does not depend on the cell
        return pt3, 0
    pre = np.array([c.rsplit("_", 1)[0] for c in m.cal.values])
    key0 = np.char.add(np.char.add(m.side.values.astype(str), "|"), pre)
    b = np.clip(np.floor((s + 15) / 0.1).astype(int), 0, 300); miss = 0
    for q in range(3):
        key = np.char.add(key0, f"_{q}")
        for k in np.unique(key):
            r = key == k
            if k in maps:
                pt3[r, q] = maps[k][b[r]]
            else:
                miss += int(r.sum())
    return pt3, miss


def run(R=3, K=40, level="run"):
    import sim, dp_score
    from headline import evaluate_all, spec_variant, MAIN
    v = spec_variant(MAIN)
    t0 = time.time()
    m, kin, out, draws = evaluate_all(v, R=R, seed=1, keep_draws=True, keep_maps=True)
    full, _ = sim.load(v); full = dp_score.add_dstart(full)
    mask = sim.opportunity_mask(full)                # the policies' opportunity set (headline.evaluate_all)
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
    # the criterion and the posterior of a drawn pool pitch at the simulated state
    isoffp = np.ascontiguousarray((full.side == "off").values)
    assert (isoffp == orig).all()                             # the batter challenges called strikes, the catcher balls
    kcoef, knots = kappa_coefs(v)
    kb = kappa_base(kp, full.g.values.astype(float), full.inning.values.astype(np.int64), isoffp, kcoef, knots)
    kpm = sim.kappa_by_count(m)
    assert dp_score.topx() == dp_score.DM + 1                # half_start_dbin's extra-top encoding
    # the full contrast's behavioural world plays obs_rule as s > kappa; a lapse-rate variant would also need the lapse
    if "lapse" in m and m.lapse.max() > 0:
        raise NotImplementedError("full_vs_behaviour: lapse-rate behavioural rule not simulated")
    lt = np.where(m.is_home_team.values, m.left_home.values, m.left_away.values).astype(np.int64)  # team's held count (pre-pitch)
    recfail = (m.ch.values & ~m.ch_over.values)                                                    # the team challenged and lost
    cu = sim.stake_cuts(m); cuts = np.array([cu["def"], cu["off"]], float)     # the cuts of evaluate_all's cal_groups
    nmiss = []
    I = lambda x: np.ascontiguousarray(np.asarray(x).astype(np.int64))
    inn, half, outs, bases = I(m.inning), I(~m.top.values), I(m.outs_when_up), I(m.bases)
    diff, balls, strikes, dbin = I(m.score_diff_home), I(m.balls), I(m.strikes), I(m.dbin)
    flip = I(~m.orig_strike.values)                                # flipped call: strike iff the original was a ball
    ng = int(m.gidx.max() + 1); gid = m.gidx.values
    # sim-vs-engine calibration: team's engine WP right after the original call of the branch pitch
    wp_orig_home = np.where(m.orig_strike.values, m.wp_strike.values, m.wp_ball.values)
    res, per = {}, {}
    Dt = lambda t_, a, b: a if t_ == 1 else b
    for pol in POLICIES:
        acc = {k: np.zeros(ng) for k in ("path", "rb", "rb_own", "binary", "later_rec", "later_A", "later_A0", "opp_A", "opp_A0",
                                         "full_win", "full_rb", "pair_win", "pair_rb", "pair_path")}
        acc["n_branch"] = 0.0; cal_s, cal_e = [], []; brs = []
        for r, dr in enumerate(draws):
            pc, dc, card = dr["pols"][pol]
            # branch at the FIRST DIVERGENCE between the policy's game and the
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
            if pol == POLICIES[0]:
                p3, mi = posterior_by_tercile(m, dr["s"], dr["pt"], dr["maps"]); nmiss.append(mi)
                dr["pt3"] = p3
                tq = np.where((m.side == "off").values, np.searchsorted(cuts[1], m.g.values, side="right"),
                              np.searchsorted(cuts[0], m.g.values, side="right"))
                assert np.allclose(p3[np.arange(len(m)), tq], dr["pt"])      # own tercile = the draw's posterior
            pt3 = np.zeros((len(full), 3)); pt3[mask] = dr["pt3"]
            sp = np.full(len(full), -1e9); sp[mask] = dr["s"]
            o = branches(rows, tt.astype(np.int64), firstc[gg, tt].astype(np.int64), copp, flip, inn, half, outs, bases,
                         diff, balls, strikes, kin, dbin, pc, dr["Dh"], dr["Da"], dr["Cnh"], dr["Cna"], card, q1, q2,
                         dc, dr["D4h"], dr["D4a"], ev, kip, br, poff, pcnt, pidx, orig, mask, Wp, pt3, sp, kb, G, RVt,
                         K, 7 + 7919 * r, dp_score.DM, isoffp, kcoef, knots, cuts)
            gb = m.g.values[rows]
            # the full contrast policy vs fitted behavioural rule from the first divergence.
            # Held counts after the branch pitch: the policy loses a token if it challenged and failed there (only
            # possible when the team flipped the call, typ = -1); the team (recorded count left_*) if it challenged and
            # lost (only possible when typ = +1).
            cP = firstc[gg, tt].astype(np.int64).copy()
            for e in np.flatnonzero(~pol_flip):
                i_ = rows[e]; t_ = tt[e]
                if cP[e] > 0 and PP.decide(pc, dr["pt"][i_], dr["s"][i_], kpm[i_, min(cP[e], 2)], m.g.values[i_], m.RV.values[i_],
                                           int(m.slot.values[i_]), int(kin[i_]), int(cP[e]), int(m.inning.values[i_]),
                                           int(m.balls.values[i_]), int(m.strikes.values[i_]), Dt(t_, dr["Dh"], dr["Da"]),
                                           Dt(t_, dr["Cnh"], dr["Cna"]), card, q1, q2, dc, Dt(t_, dr["D4h"], dr["D4a"]),
                                           int(m.dbin.values[i_])) and not dr["W"][i_]:
                    cP[e] -= 1
            cB = (lt[rows] - (pol_flip & recfail[rows])).astype(np.int64)
            pendP = np.where(pol_flip, flip[rows], 1 - flip[rows]).astype(np.int64)
            fo = full_branches(rows, tt.astype(np.int64), cP, cB, copp, pendP, inn, half, outs, bases, diff, balls, strikes,
                               kin, dbin, pc, dr["Dh"], dr["Da"], dr["Cnh"], dr["Cna"], card, q1, q2, dc, dr["D4h"], dr["D4a"],
                               ev, kip, br, poff, pcnt, pidx, orig, mask, Wp, pt3, sp, kb, G, RVt, K, 7 + 7919 * r,
                               dp_score.DM, isoffp, kcoef, knots, cuts)
            np.add.at(acc["full_win"], gg, (fo[:, 0] - fo[:, 3]) / R)
            np.add.at(acc["full_rb"], gg, (sgn * gb + fo[:, 1] - fo[:, 2] - fo[:, 4] + fo[:, 5]) / R)
            # the PAIRED contrast. full_branches above starts from the RECORDED path (first
            # divergence against the recorded challenges, recorded inventory) and continues with the fitted rule, so it
            # compares with path - observed. Here both worlds are simulated policies: the fitted behavioural rule is
            # played along the stream in the same draw (path_trace), the policy is branched against IT at their first
            # divergence, both inventories come from their own simulated paths, and the result compares with
            # path(policy) - path(obs_rule) on the same draws -- like with like.
            pB, dB, cardB = dr["pols"]["obs_rule"]
            sB, fB, hB = PP.path_trace(*base, dr["W"], dr["pt"], dr["s"], m.g.values, m.RV.values, kpm, pB, dr["Dh"],
                                       dr["Da"], dr["Cnh"], dr["Cna"], cardB, q1, q2, dB, dr["D4h"], dr["D4a"])
            f2, typ2, fc2, _ = PP.first_divergence(*base, dr["W"], dr["pt"], dr["s"], m.g.values, m.RV.values, kpm, pc,
                                                   dr["Dh"], dr["Da"], dr["Cnh"], dr["Cna"], card, q1, q2, dc, dr["D4h"],
                                                   dr["D4a"], sB, ng)
            g2, t2 = np.nonzero(f2 >= 0)
            rows2 = f2[g2, t2].astype(np.int64); sgn2 = typ2[g2, t2].astype(float); pf2 = sgn2 > 0
            cP2 = fc2[g2, t2].astype(np.int64).copy()
            for e in np.flatnonzero(~pf2):           # the policy challenged the branch pitch and lost
                i_ = rows2[e]; t_ = t2[e]
                if cP2[e] > 0 and PP.decide(pc, dr["pt"][i_], dr["s"][i_], kpm[i_, min(cP2[e], 2)], m.g.values[i_],
                                            m.RV.values[i_], int(m.slot.values[i_]), int(kin[i_]), int(cP2[e]),
                                            int(m.inning.values[i_]), int(m.balls.values[i_]), int(m.strikes.values[i_]),
                                            Dt(t_, dr["Dh"], dr["Da"]), Dt(t_, dr["Cnh"], dr["Cna"]), card, q1, q2, dc,
                                            Dt(t_, dr["D4h"], dr["D4a"]), int(m.dbin.values[i_])) and not dr["W"][i_]:
                    cP2[e] -= 1
            cB2 = (hB[rows2, t2] - fB[rows2]).astype(np.int64)           # the rule's count after the branch pitch
            copp2 = hB[rows2, 1 - t2].astype(np.int64)                   # opponent: its rule's simulated count
            pend2 = np.where(pf2, flip[rows2], 1 - flip[rows2]).astype(np.int64)
            fo2 = full_branches(rows2, t2.astype(np.int64), cP2, cB2, copp2, pend2, inn, half, outs, bases, diff, balls,
                                strikes, kin, dbin, pc, dr["Dh"], dr["Da"], dr["Cnh"], dr["Cna"], card, q1, q2, dc,
                                dr["D4h"], dr["D4a"], ev, kip, br, poff, pcnt, pidx, orig, mask, Wp, pt3, sp, kb, G, RVt,
                                K, 13 + 7919 * r, dp_score.DM, isoffp, kcoef, knots, cuts)
            gb2 = m.g.values[rows2]
            np.add.at(acc["pair_win"], g2, (fo2[:, 0] - fo2[:, 3]) / R)
            np.add.at(acc["pair_rb"], g2, (sgn2 * gb2 + fo2[:, 1] - fo2[:, 2] - fo2[:, 4] + fo2[:, 5]) / R)
            acc["pair_path"] += (dr["vals"][pol].sum(2) - dr["vals"]["obs_rule"].sum(2)).sum(1) / R
            acc["n_pair_branch"] = acc.get("n_pair_branch", 0.0) + len(rows2) / R
            path_gt = dr["vals"][pol].sum(2)                      # [game, team]
            acc["path"] += path_gt.sum(1) / R
            # fidelity check on the policy-flip branches: the recorded continuation is a draw of the no-flip world
            pf = pol_flip
            np.add.at(acc["later_rec"], gg[pf], (path_gt[gg[pf], tt[pf]] - pre[gg[pf], tt[pf]] - gb[pf]) / R)
            np.add.at(acc["later_A0"], gg[pf], o[pf, 3] / R); np.add.at(acc["later_A"], gg[pf], o[pf, 1] / R)
            np.add.at(acc["opp_A"], gg, sgn * o[:, 5] / R); np.add.at(acc["opp_A0"], gg, sgn * o[:, 6] / R)
            # correction = (policy world) - (recorded world); sgn = -1 swaps A and A'
            np.add.at(acc["rb_own"], gg, sgn * (o[:, 1] - o[:, 3]) / R)                 # team's own later flips only
            np.add.at(acc["rb"], gg, sgn * (o[:, 1] - o[:, 3] - o[:, 5] + o[:, 6]) / R) # signed, both teams
            np.add.at(acc["binary"], gg, sgn * (o[:, 0] - o[:, 2] - gb) / R)
            acc["n_skip"] = acc.get("n_skip", 0.0) + float((~pol_flip).sum()) / R
            acc["n_branch"] += len(rows) / R
            cal_s.append(o[:, 4]); cal_e.append(np.where(tt == 1, wp_orig_home[rows], 1 - wp_orig_home[rows]))
            brs.append(dict(r=r, game=gg, team=tt, row=rows, g=gb, out=o, sgn=sgn))
        acc["rb"] += acc["path"]; acc["rb_own"] += acc["path"]; acc["binary"] += acc["path"]
        per[pol] = {k: acc[k] for k in ("path", "rb", "rb_own", "binary", "later_rec", "later_A", "later_A0", "opp_A", "opp_A0",
                                        "full_win", "full_rb", "pair_win", "pair_rb", "pair_path")}
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
                        binary_minus_rb_home=float(gap[home].sum() / R / 30), binary_minus_rb_away=float(gap[~home].sum() / R / 30),
                        full_vs_behaviour_win=acc["full_win"].sum() / 30, full_vs_behaviour_rb=acc["full_rb"].sum() / 30,
                        paired_win=acc["pair_win"].sum() / 30, paired_rb=acc["pair_rb"].sum() / 30,
                        paired_path=acc["pair_path"].sum() / 30,
                        paired_branches_per_team_game=acc.get("n_pair_branch", 0.0) / (2 * ng))
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
    # wins lost against the fitted behavioural rule, re-simulated end to end from the first
    # divergence (win count and RB), next to the path-sum loss against the simulated behavioural rule and against the
    # observed challenges
    fvb = {}
    for pol in POLICIES:
        p = per[pol]
        fvb[pol] = dict(win=ci(p["full_win"]), rb=ci(p["full_rb"]), path_minus_obs_rule=ci(p["path"] - per["obs_rule"]["path"]),
                        path_minus_observed=ci(p["path"] - obs),
                        # like with like -- the recorded-start contrast against path - observed, and the
                        # paired simulated contrast against path(policy) - path(obs_rule) on the same draws
                        rb_minus_path_minus_observed=ci(p["full_rb"] - (p["path"] - obs)),
                        win_minus_rb=ci(p["full_win"] - p["full_rb"]),
                        paired_win=ci(p["pair_win"]), paired_rb=ci(p["pair_rb"]), paired_path=ci(p["pair_path"]),
                        paired_rb_minus_path=ci(p["pair_rb"] - p["pair_path"]),
                        paired_win_minus_rb=ci(p["pair_win"] - p["pair_rb"]))
    tag = "" if level == "run" else f"_{level}"
    json.dump(dict(R=R, K=K, variant=MAIN, simulator=level, boot_resamples=CAUSAL_BOOT, point=res, game_bootstrap=boot,
                   state_criterion="state-dependent criterion and belief", posterior_rows_without_cell_map=int(sum(nmiss)), full_vs_behaviour=fvb),
              open(f"results/causal{tag}.json", "w"), indent=1)
    pickle.dump(dict(per=per, games=m.game_pk.unique()), open(f"data/causal_pergame{tag}.pkl", "wb"))
    return res, boot


if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 3; K = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    level = sys.argv[3] if len(sys.argv) > 3 else "run"
    res, boot = run(R, K, level)
    print(json.dumps(boot, indent=1))
