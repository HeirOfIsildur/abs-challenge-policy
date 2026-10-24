"""Generalised joint likelihood of the missed-call (truth) model and the perception/challenge model (HONZA-73).

Same structure as joint.py, written in JAX for exact gradients/Hessians:
  w_true = w_meas + eta, eta ~ N(0, sm_edge^2); W = 1{w_true > 0} (call wrong under ABS);
  s = w_true + eps, eps = sr * e, e ~ F (unit-variance law); challenge iff s > kappa = X beta_side.
Extensions (all switchable via a `spec` dict):
  law      : "normal" | "logistic" | "t<nu>"  (F with unit variance, so sr is always the noise SD)
  sr_edge  : log sr shift for the horizontal edge
  sr_region: log sr shifts by zone region (bottom = base, top, inside = horizontal edge on the batter's side of the
             plate, outside = the far horizontal edge; split by batter handedness, see region())
  sr_slope : log sr = ... + gamma * (|w_meas| - 1)  (noise grows/shrinks away from the edge)
  kappa_g  : kappa gets a piecewise-linear term in log g, hinges (log g - k)+ at G_KNOTS (HONZA-106: does a
             non-linear stake response of the criterion explain the realised-vs-simulated gap, and does it move sigma_r?)
  re       : player random effects (a_i on kappa, b_i on log sr), bivariate normal, Gauss-Hermite marginal
Challenged rows: P(ch, W=y | w) = int_{W=y} phi_m(eta) F((w+eta-kappa)/sr) d eta, Gauss-Legendre in u = Phi(eta/sm).
Not challenged: closed form for the normal law, Gauss-Hermite in eta otherwise.
HONZA-88: on the corrected geometry (raw Statcast 2026 coordinates, src/geometry.py; round-ball corners since HONZA-120
and the count-based join since HONZA-132) the ABS ruling is a deterministic function of the tracked location for all
but a handful of challenges (results/geometry_audit.json), so the main specs use det=True: W = 1{w_meas > 0}
(sm = 0, mu = 0) and the ruling disagrees with W with probability pi (contam). Then
  challenged: P(ch, y | w) = F((w - kappa)/sr) [(1 - pi) 1{y = W} + pi 1{y != W}],  not challenged: 1 - F((w - kappa)/sr).
The smooth truth model (mu, sm by edge or region) is kept as the variants sm_edge / sm_region for the LR check."""
import json
import numpy as np, pandas as pd
import jax, jax.numpy as jnp
from jax.scipy.special import ndtri, ndtr, log_ndtr, betainc, logsumexp
jax.config.update("jax_enable_x64", True)

GL_X, GL_W = np.polynomial.legendre.leggauss(48)
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(40); GH_W = GH_W / GH_W.sum()
# HONZA-130 (Fable review): the horizontal edges are split by BATTER side (region(): inside / outside), not by the
# pitcher's arm / glove side. REGIONS keeps the legacy "arm" / "glove" keys of the stored JSON outputs (mu_arm,
# sr_region_mult["arm"], ... are read by paper/make_numbers.py and paper/supplement_numbers.py): "arm" = inside,
# "glove" = outside. REGION_NAMES are the labels for new human-readable outputs.
REGIONS = ["bottom", "top", "arm", "glove"]
REGION_NAMES = ["bottom", "top", "inside", "outside"]
G_KNOTS = (-5.0, -4.0, -3.2)       # log g at about the 40th, 80th and 95th percentiles of the eligible pitches
G_LOW = np.log(0.001)              # HONZA-122 (Fable review): low-stake hinge, a criterion that can say "nothing at stake"
LG_MID = -5.0                      # centre of log g for the stake-dependent noise term (sr_g)
PLAYER = dict(off="batter", **{"def": "fielder_2"})   # batter decides on called strikes, catcher on called balls


def design(e):
    lg = np.log(np.clip(e.g.values, 1e-4, None))
    return np.column_stack([np.ones(len(e)), lg, (e.left == 1).astype(float), np.clip(e.inning.values, 1, 10) / 9,
                            (e.strikes == 2).astype(float), (e.balls == 3).astype(float)])


def glow(e):
    """(log g - log 0.001)_-: the criterion's extra rise below g = 0.001 (blowouts, mercy pitches)."""
    lg = np.log(np.clip(e.g.values, 1e-4, None))
    return np.minimum(lg - G_LOW, 0.0)


SECOND_HALF = "2026-07-01"          # HONZA-132: the time-varying criterion / noise model splits the season here


def tfeat(e):
    """HONZA-132 (Sol, Astra reviews): time-varying criterion -- [h2, h2 * (log g - LG_MID)] with h2 = 1 from 1 July
    (the same split as the temporal out-of-sample check). Frames without a date get zeros."""
    h2 = (pd.to_datetime(e.date).values >= np.datetime64(SECOND_HALF)).astype(float) if "date" in e else np.zeros(len(e))
    lg = np.log(np.clip(e.g.values, 1e-4, None)) - LG_MID
    return np.column_stack([h2, h2 * lg])


STATE_TERMS = ["abs_score_6", "late_close", "two_outs", "risp", "on_1b"]


def sfeat(e):
    """HONZA-134 (Opus round-5 review): game-state terms of the criterion -- |score difference| (capped at 6) / 6,
    late and close (inning >= 7, |score difference| <= 1), two outs, runner in scoring position, runner on first."""
    ad = np.minimum(np.abs(e.score_diff_home.values), 6)
    return np.column_stack([ad / 6, ((e.inning.values >= 7) & (ad <= 1)).astype(float), (e.outs.values == 2).astype(float),
                            (e.on_2b.notna() | e.on_3b.notna()).values.astype(float), e.on_1b.notna().values.astype(float)])


def gfeat(e):
    lg = np.log(np.clip(e.g.values, 1e-4, None))
    return np.column_stack([np.clip(lg - k, 0, None) for k in G_KNOTS])


def region(e):
    """nearest zone edge: bottom/top (vertical edge) or inside/outside relative to the batter (horizontal edge):
    0 bottom, 1 top, 2 inside (key "arm" in REGIONS), 3 outside (key "glove")."""
    mid = (e.sz_top.values + e.sz_bot.values) / 2
    top = e.pz_mid.values > mid
    inside = np.where(e.stand.values == "R", e.px_mid.values < 0, e.px_mid.values > 0)
    return np.where(e.edge.values == "vert", np.where(top, 1, 0), np.where(inside, 2, 3))


def eligible_mask(m, side):
    """the likelihood sample of one side: challenge opportunities (side, a challenge held, not a position player
    pitching, HONZA-122) inside the fitting window |w| < 8 in. The window trims the fit only; the simulated policies
    see every located called pitch (sim.opportunity_mask)."""
    ok = ~m.pos_pitcher.values if "pos_pitcher" in m else np.ones(len(m), bool)
    return (m.side == side).values & (m.left > 0).values & (m.w.abs() < 8).values & ok


def prepare(m, players=False):
    """m: called-pitch table with side/left/w (preliminary)/g. Returns dict side -> numpy arrays."""
    out = {}
    for side in ("off", "def"):
        e = m[eligible_mask(m, side)]
        e = e.iloc[np.argsort(~e.ch.values, kind="stable")]        # challenged rows first
        pid = e[PLAYER[side]].fillna(-1).astype(np.int64).values if players else np.zeros(len(e), np.int64)
        uniq, pidx = np.unique(pid, return_inverse=True)
        out[side] = dict(d=e.dist_in.values, vert=(e.edge.values == "vert").astype(float), os=e.orig_strike.values.astype(float),
                         X=design(e), G=gfeat(e), GL=glow(e), TF=tfeat(e), SF=sfeat(e), LG=np.log(np.clip(e.g.values, 1e-4, None)) - LG_MID,
                         ch=e.ch.values.astype(bool), y=e.ch_over.values.astype(bool),
                         reg=region(e), pidx=pidx, players=uniq, n=len(e), nch=int(e.ch.sum()))
    return out


def cdf_unit(x, law):
    if law == "normal":
        return ndtr(x)
    if law == "logistic":
        return jax.nn.sigmoid(x * jnp.pi / jnp.sqrt(3.0))
    nu = int(law[1:]); z = x / jnp.sqrt((nu - 2) / nu)
    return t_cdf_int(z, nu)


def t_cdf_int(z, nu):
    """Student-t CDF for integer nu in closed form (Abramowitz & Stegun 26.7.3-4); differentiable and fast."""
    th = jnp.arctan(z / jnp.sqrt(nu)); c2 = jnp.cos(th) ** 2; s = jnp.sin(th)
    if nu % 2:
        term, tot = jnp.ones_like(z), jnp.ones_like(z)
        for k in range(1, (nu - 1) // 2):
            term = term * c2 * (2 * k) / (2 * k + 1); tot = tot + term
        A = (2 / jnp.pi) * (th + (s * jnp.cos(th) * tot if nu > 1 else 0.0))
    else:
        term, tot = jnp.ones_like(z), jnp.ones_like(z)
        for k in range(1, nu // 2):
            term = term * c2 * (2 * k - 1) / (2 * k); tot = tot + term
        A = s * tot
    return 0.5 * (1 + A)


def spec_of(variant):
    """lik spec of a named variant (robust_fit.VARIANTS): deterministic truth + ruling mislabel pi for every
    perception variant; the smooth truth model only for sm_edge (offset/scale by edge, no pi) and sm_region (by
    region, with pi; the pre-HONZA-88 baseline)."""
    from robust_fit import VARIANTS
    truth = ({} if variant == "sm_edge" else {"truth_region": True, "contam": True} if variant == "sm_region"
             else {"det": True, "contam": True})
    return dict(VARIANTS[variant], **truth)


def n_truth(spec):
    """truth parameters: (mu, log sm) by edge (vert, horiz) or by zone region (bottom, top, inside, outside); with det
    only the mislabel rate pi."""
    if spec.get("det"):
        return int(bool(spec.get("contam")))
    return (8 if spec.get("truth_region") else 4) + bool(spec.get("contam"))


def truth_mu(th, spec, vert, reg):
    """fitted offset mu for each row (numpy)."""
    if spec.get("det"):
        return np.zeros(len(vert))
    P = unpack(np.asarray(th), spec)
    return np.asarray(P["mu"])[reg] if spec.get("truth_region") else np.where(vert, float(P["mu"][0]), float(P["mu"][1]))


def n_side(spec):
    """number of parameters per side (beta 6, log sr, optional extensions)."""
    return (7 + 3 * bool(spec.get("kappa_region")) + bool(spec.get("sr_edge")) + 3 * bool(spec.get("sr_region")) + bool(spec.get("sr_slope"))
            + len(G_KNOTS) * bool(spec.get("kappa_g")) + 3 * bool(spec.get("re")) + bool(spec.get("lapse"))
            + bool(spec.get("kappa_glow")) + bool(spec.get("sr_g")) + bool(spec.get("lapse_low"))
            + 2 * bool(spec.get("kappa_time")) + bool(spec.get("sr_time")) + 5 * bool(spec.get("kappa_state")))


def unpack(th, spec):
    nt = n_truth(spec); nb = nt - bool(spec.get("contam"))
    if spec.get("det"):        # deterministic truth: mu = 0, sm = 0 in all four regions (numpy consumers)
        P = dict(mu=jnp.zeros(4), lsm=jnp.full(4, -jnp.inf))
    else:
        P = dict(mu=th[0:nb // 2], lsm=th[nb // 2:nb])
    i = nt
    if spec.get("contam"): P["lpi"] = th[nb]
    for side in ("off", "def"):
        q = dict(beta=th[i:i + 6], lsr=th[i + 6]); j = i + 7
        if spec.get("kappa_region"): q["kreg"] = jnp.concatenate([jnp.zeros(1), th[j:j + 3]]); j += 3
        if spec.get("sr_edge"): q["edge"] = th[j]; j += 1
        if spec.get("sr_region"): q["region"] = jnp.concatenate([jnp.zeros(1), th[j:j + 3]]); j += 3
        if spec.get("sr_slope"): q["slope"] = th[j]; j += 1
        if spec.get("kappa_g"): q["kg"] = th[j:j + len(G_KNOTS)]; j += len(G_KNOTS)
        if spec.get("re"): q["re"] = th[j:j + 3]; j += 3
        if spec.get("lapse"): q["llapse"] = th[j]; j += 1
        if spec.get("kappa_glow"): q["kgl"] = th[j]; j += 1
        if spec.get("sr_g"): q["srg"] = th[j]; j += 1
        if spec.get("lapse_low"): q["llapse_low"] = th[j]; j += 1      # HONZA-130: lapse logit shift at log g <= LG_MID
        if spec.get("kappa_time"): q["kt"] = th[j:j + 2]; j += 2           # HONZA-132: second-half criterion shift
        if spec.get("sr_time"): q["srt"] = th[j]; j += 1                 # HONZA-132: second-half log noise shift
        if spec.get("kappa_state"): q["ks"] = th[j:j + 5]; j += 5          # HONZA-134: game-state criterion terms
        P[side] = q; i = j
    return P


def init(spec, base=None):
    """start values from a plain model theta: 4 (edge) or 8 (region) truth params followed by 2 x (beta 6, log sr)."""
    base = np.asarray(base) if base is not None else np.r_[-0.4, -0.2, np.log(.8), np.log(.5), np.zeros(6), 1.0, np.zeros(6), 0.8]
    nt = 8 if len(base) in (22, 23) else 4
    tr, rest = base[:nt], base[len(base) - 14:]
    if spec.get("det"):
        tr = np.zeros(0)
    elif spec.get("truth_region") and nt == 4:
        tr = np.r_[tr[0], tr[0], tr[1], tr[1], tr[2], tr[2], tr[3], tr[3]]
    elif not spec.get("truth_region") and nt == 8:
        tr = np.r_[tr[:2].mean(), tr[2:4].mean(), tr[4:6].mean(), tr[6:8].mean()]
    th = list(tr) + ([np.log(0.005 / 0.995)] if spec.get("contam") else []); i = 0
    for side in ("off", "def"):
        th += list(rest[i:i + 7]); i += 7
        if spec.get("kappa_region"): th += [0.0] * 3
        if spec.get("sr_edge"): th += [0.0]
        if spec.get("sr_region"): th += [0.0] * 3
        if spec.get("sr_slope"): th += [0.0]
        if spec.get("kappa_g"): th += [0.0] * len(G_KNOTS)
        if spec.get("re"): th += [np.log(0.3), np.log(0.2), 0.0]
        if spec.get("lapse"): th += [-3.0]
        if spec.get("kappa_glow"): th += [0.0]
        if spec.get("sr_g"): th += [0.0]
        if spec.get("lapse_low"): th += [0.0]
        if spec.get("kappa_time"): th += [0.0, 0.0]
        if spec.get("sr_time"): th += [0.0]
        if spec.get("kappa_state"): th += [0.0] * 5
    return np.array(th, float)


def row_terms(D, P, q, spec, a=None, b=None):
    """log-likelihood per row, challenged rows first then the rest (D["order"]); with random-effect nodes a, b of
    shape (1, Q) the result is (n, Q). Challenged and non-challenged rows are evaluated on separate arrays so that
    no masked branch can leak NaN into derivatives."""
    law = spec.get("law", "normal")
    if spec.get("det"):
        return row_terms_det(D, P, q, spec, a, b)
    vert = D["vert"]
    if spec.get("truth_region"):
        mu = P["mu"][D["reg"]]; sm = jnp.exp(P["lsm"][D["reg"]])
    else:
        mu = vert * P["mu"][0] + (1 - vert) * P["mu"][1]
        sm = jnp.exp(vert * P["lsm"][0] + (1 - vert) * P["lsm"][1])
    dc = D["d"] - mu
    w = jnp.where(D["os"] > 0, -dc, dc)
    lsr = q["lsr"] + jnp.zeros_like(w)
    if spec.get("sr_edge"): lsr = lsr + q["edge"] * (1 - vert)
    if spec.get("sr_region"): lsr = lsr + q["region"][D["reg"]]
    if spec.get("sr_slope"): lsr = lsr + q["slope"] * (jnp.abs(w) - 1)
    kap = D["X"] @ q["beta"]
    if spec.get("kappa_region"): kap = kap + q["kreg"][D["reg"]]
    if spec.get("kappa_g"): kap = kap + D["G"] @ q["kg"]
    y = D["y"]
    if a is not None:
        w, sm, kap, lsr, y = w[:, None], sm[:, None], kap[:, None] + a, lsr[:, None] + b, y[:, None]
    sr = jnp.exp(lsr)
    nc = D["nch"]                        # rows [0, nc) challenged, [nc, n) not challenged
    wn, smn, kn, srn = w[nc:], sm[nc:], kap[nc:], sr[nc:]
    if law == "normal":
        l_nc = log_ndtr(-(wn - kn) / jnp.sqrt(smn ** 2 + srn ** 2))
    else:
        eta = smn[..., None] * GH_X
        p = (cdf_unit((wn[..., None] + eta - kn[..., None]) / srn[..., None], law) * GH_W).sum(-1)
        l_nc = jnp.log(jnp.clip(1 - p, 1e-300))
    # challenged: split the eta integral at the ABS boundary eta = -w. Region W=y is mapped to v in (0, ub) with
    # ub = Phi(+-w/sm) and eta = -+ sm * Phi^-1(v), so both ends stay away from 1 and log ub is exact.
    wc, smc, kc, src, yc = w[:nc], sm[:nc], kap[:nc], sr[:nc], y[:nc]
    sg = jnp.where(yc, -1.0, 1.0)                      # W=1: eta = -sm*ndtri(v) > -w ; W=0: eta = sm*ndtri(v) < -w
    lub = log_ndtr(-sg * wc / smc)
    v = jnp.exp(jnp.maximum(lub, -250.0))[..., None] * (GL_X + 1) / 2   # keeps ndtri and its 2nd derivative finite
    eta = sg[..., None] * smc[..., None] * ndtri(v)
    f = cdf_unit((wc[..., None] + eta - kc[..., None]) / src[..., None], law)
    l_c = jnp.log((f * GL_W).sum(-1) / 2 + 1e-300) + lub
    if spec.get("contam"):
        # the ruling disagrees with the tracked truth with probability pi (data/attribution errors)
        sg2 = -sg; lub2 = log_ndtr(-sg2 * wc / smc)
        eta2 = sg2[..., None] * smc[..., None] * ndtri(jnp.exp(jnp.maximum(lub2, -250.0))[..., None] * (GL_X + 1) / 2)
        f2 = cdf_unit((wc[..., None] + eta2 - kc[..., None]) / src[..., None], law)
        l_c2 = jnp.log((f2 * GL_W).sum(-1) / 2 + 1e-300) + lub2
        l_c = jnp.logaddexp(jax.nn.log_sigmoid(-P["lpi"]) + l_c, jax.nn.log_sigmoid(P["lpi"]) + l_c2)
    return jnp.concatenate([l_c, l_nc], axis=0)


def side_terms(D, q, spec, w):
    """log sr and kappa per row (shared by the smooth and the deterministic truth model)."""
    vert = D["vert"]
    lsr = q["lsr"] + jnp.zeros_like(w)
    if spec.get("sr_edge"): lsr = lsr + q["edge"] * (1 - vert)
    if spec.get("sr_region"): lsr = lsr + q["region"][D["reg"]]
    if spec.get("sr_slope"): lsr = lsr + q["slope"] * (jnp.abs(w) - 1)
    if spec.get("sr_g"): lsr = lsr + q["srg"] * D["LG"]
    if spec.get("sr_time"): lsr = lsr + q["srt"] * D["TF"][:, 0]
    kap = D["X"] @ q["beta"]
    if spec.get("kappa_region"): kap = kap + q["kreg"][D["reg"]]
    if spec.get("kappa_g"): kap = kap + D["G"] @ q["kg"]
    if spec.get("kappa_glow"): kap = kap + D["GL"] * q["kgl"]
    if spec.get("kappa_time"): kap = kap + D["TF"] @ q["kt"]
    if spec.get("kappa_state"): kap = kap + D["SF"] @ q["ks"]
    return lsr, kap


def row_terms_det(D, P, q, spec, a=None, b=None):
    """deterministic truth W = 1{w > 0}; the ruling y disagrees with W with probability pi (HONZA-88)."""
    law = spec.get("law", "normal")
    w = jnp.where(D["os"] > 0, -D["d"], D["d"])
    lsr, kap = side_terms(D, q, spec, w)
    y = D["y"]
    if a is not None:
        w, kap, lsr, y = w[:, None], kap[:, None] + a, lsr[:, None] + b, y[:, None]
    z = (w - kap) / jnp.exp(lsr)
    nc = D["nch"]
    if law == "normal" and spec.get("lapse"):
        # HONZA-114: lapse rate lambda, P(challenge | w) = (1 - lambda) Phi(z): the challenge curve tops out below 1
        lam = q["llapse"]
        if spec.get("lapse_low"):        # HONZA-130 (opt-in): the lapse rate may differ at low stakes (log g <= LG_MID)
            lam = lam + q["llapse_low"] * (D["LG"] <= 0)
            if a is not None: lam = lam[:, None]
        ll1, ll0 = jax.nn.log_sigmoid(-lam), jax.nn.log_sigmoid(lam)
        l_ch = ll1 + log_ndtr(z); l_no = jnp.logaddexp(log_ndtr(-z), ll0 + log_ndtr(z))
    elif law == "normal":
        l_ch, l_no = log_ndtr(z), log_ndtr(-z)
    else:
        F = cdf_unit(z, law)
        l_ch, l_no = jnp.log(jnp.clip(F, 1e-300)), jnp.log(jnp.clip(1 - F, 1e-300))
    agree = y[:nc] == (w[:nc] > 0)
    if spec.get("contam"):
        l_rule = jnp.where(agree, jax.nn.log_sigmoid(-P["lpi"]), jax.nn.log_sigmoid(P["lpi"]))
    else:
        l_rule = jnp.where(agree, 0.0, -690.0)
    return jnp.concatenate([l_ch[:nc] + l_rule, l_no[nc:]], axis=0)


def gh2(k=7):
    x, wt = np.polynomial.hermite_e.hermegauss(k); wt = wt / wt.sum()
    X, Y = np.meshgrid(x, x); W = np.outer(wt, wt)
    return X.ravel(), Y.ravel(), np.log(W.ravel())


def re_nodes(q, k):
    z1, z2, lw = gh2(k)
    sa, sb, r = jnp.exp(q["re"][0]), jnp.exp(q["re"][1]), jnp.tanh(q["re"][2])
    return sa * z1, sb * (r * z1 + jnp.sqrt(1 - r ** 2) * z2), lw


def side_ll(D, P, side, spec):
    q = P[side]
    if not spec.get("re"):
        return row_terms(D, P, q, spec).sum()
    a, b, lw = re_nodes(q, spec.get("gh", 7))
    # only challenged rows need the (n, Q) quadrature; split to keep memory down
    L = row_terms(D, P, q, spec, a[None, :], b[None, :])
    S = jax.ops.segment_sum(L, D["pidx"], num_segments=len(D["players"]))
    return logsumexp(S + lw[None, :], axis=1).sum()


def re_posterior(data, th, side, spec):
    """EB posterior mean / sd of (a_i, b_i) per player, with rows and challenges per player."""
    D = {k: (jnp.asarray(v) if k not in ("players", "n", "nch") else v) for k, v in data[side].items()}
    P = unpack(jnp.asarray(th), spec); q = P[side]
    a, b, lw = (np.asarray(x) for x in re_nodes(q, spec.get("gh", 7)))
    L = np.asarray(row_terms(D, P, q, spec, jnp.asarray(a)[None, :], jnp.asarray(b)[None, :]))
    npl = len(data[side]["players"])
    S = np.zeros((npl, len(a))); np.add.at(S, data[side]["pidx"], L)
    S = S + lw[None, :]; S -= S.max(1, keepdims=True); W = np.exp(S); W /= W.sum(1, keepdims=True)
    ea, eb = W @ a, W @ b
    # HONZA-132 (Sol #7, Astra #6): the posterior covariance of (a_i, b_i), for joint draws (sim.apply_lik re="draw")
    return pd.DataFrame(dict(player=data[side]["players"], a=ea, b=eb, sd_a=np.sqrt(np.clip(W @ a ** 2 - ea ** 2, 0, None)),
                             sd_b=np.sqrt(np.clip(W @ b ** 2 - eb ** 2, 0, None)), cov_ab=W @ (a * b) - ea * eb,
                             n=np.bincount(data[side]["pidx"], minlength=npl),
                             n_ch=np.bincount(data[side]["pidx"], weights=data[side]["ch"], minlength=npl)))


def make_nll(data, spec):
    Dj = {s: {k: (jnp.asarray(v) if k not in ("players", "n", "nch") else v) for k, v in data[s].items()} for s in data}
    def nll(th):
        P = unpack(th, spec)
        return -(side_ll(Dj["off"], P, "off", spec) + side_ll(Dj["def"], P, "def", spec))
    return jax.jit(nll)


def fit(data, spec, th0=None, base=None, maxiter=200, verbose=False):
    """BFGS warm-up, then Newton trust region with the exact JAX Hessian."""
    from scipy.optimize import minimize
    nll = make_nll(data, spec)
    vg = jax.jit(jax.value_and_grad(nll)); hs = jax.jit(jax.hessian(nll))
    th0 = init(spec, base) if th0 is None else np.asarray(th0, float)
    f = lambda th: tuple(np.asarray(v, float) for v in vg(jnp.asarray(th)))
    r = minimize(f, th0, jac=True, method="L-BFGS-B", options=dict(maxiter=60 if not spec.get("re") else 1000, maxcor=30))
    if spec.get("re"):
        # the exact Hessian over player nodes does not fit in memory. HONZA-132 (Sol #7, Astra #6: max gradients 0.13
        # and 1.28 at the L-BFGS stop): Newton refinement with a central-difference Hessian of the exact JAX gradient
        r = newton_fd(vg, r, maxiter=maxiter if maxiter < 200 else 12, verbose=verbose)
        if verbose:
            print(spec, "nll", round(float(r.fun), 2), r.message, r.nit, "max|grad|", float(np.abs(r.jac).max()), flush=True)
        return r, nll
    r = minimize(f, r.x, jac=True, hess=lambda th: np.asarray(hs(jnp.asarray(th))), method="trust-exact",
                 options=dict(maxiter=maxiter, gtol=1e-4))
    if verbose:
        print(spec, "nll", round(float(r.fun), 2), r.message, r.nit, flush=True)
    return r, nll


def fd_hessian(vg, th, h=1e-4):
    """central-difference Hessian of the exact gradient (symmetrised); vg = jit(value_and_grad(nll))."""
    th = np.asarray(th, float); k = len(th); H = np.zeros((k, k))
    for i in range(k):
        e = np.zeros(k); e[i] = h
        H[i] = (np.asarray(vg(jnp.asarray(th + e))[1]) - np.asarray(vg(jnp.asarray(th - e))[1])) / (2 * h)
    return (H + H.T) / 2


def newton_fd(vg, r, maxiter=12, gtol=1e-3, verbose=False):
    """Newton steps with a finite-difference Hessian and step halving, from an L-BFGS result r; r.hess_fd keeps the
    last Hessian (for a Wald / parametric-bootstrap covariance). Stops at max|grad| < gtol."""
    from scipy.optimize import OptimizeResult
    th = np.asarray(r.x, float); f, g = (np.asarray(v, float) for v in vg(jnp.asarray(th))); H = None; it = 0
    for it in range(maxiter):
        if np.abs(g).max() < gtol:
            break
        H = fd_hessian(vg, th)
        w, V = np.linalg.eigh(H); w = np.maximum(w, 1e-6 * max(w.max(), 1.0))      # keep the step a descent direction
        step = -(V / w) @ (V.T @ g); t = 1.0
        while t > 1e-4:
            fn, gn = (np.asarray(v, float) for v in vg(jnp.asarray(th + t * step)))
            if np.isfinite(fn) and fn <= f + 1e-4 * t * g @ step:
                break
            t /= 2
        if t <= 1e-4:
            break
        th, f, g = th + t * step, fn, gn
        if verbose:
            print("  newton", it, round(float(f), 4), "max|grad|", float(np.abs(g).max()), flush=True)
    if H is None or np.abs(g).max() >= gtol or it > 0:
        H = fd_hessian(vg, th)
    return OptimizeResult(x=th, fun=float(f), jac=g, success=bool(np.abs(g).max() < max(gtol, 1e-2)),
                          message=f"newton_fd: max|grad| {np.abs(g).max():.2e}", nit=int(r.nit) + it, hess_fd=H)


def hessian(nll, th):
    return np.asarray(jax.hessian(nll)(jnp.asarray(th)))


def readable(th, spec):
    P = unpack(jnp.asarray(th), spec)
    labels = REGIONS if spec.get("truth_region") else ["vert", "horiz"]
    if spec.get("det"): labels = REGIONS
    out = {f"mu_{l}": float(P["mu"][k]) for k, l in enumerate(labels)}
    out.update({f"sm_{l}": float(np.exp(P["lsm"][k])) for k, l in enumerate(labels)})
    if "lpi" in P: out["contam_pi"] = float(jax.nn.sigmoid(P["lpi"]))
    for s in ("off", "def"):
        q = P[s]; out[f"{s}_sr"] = float(np.exp(q["lsr"])); out[f"{s}_beta"] = np.asarray(q["beta"]).round(4).tolist()
        if "edge" in q: out[f"{s}_sr_horiz_mult"] = float(np.exp(q["edge"]))
        if "kreg" in q: out[f"{s}_kappa_region_shift"] = dict(zip(REGIONS, np.asarray(q["kreg"]).round(3).tolist()))
        if "region" in q: out[f"{s}_sr_region_mult"] = dict(zip(REGIONS, np.exp(np.asarray(q["region"])).round(3).tolist()))
        if "slope" in q: out[f"{s}_sr_slope_per_in"] = float(q["slope"])
        if "kg" in q: out[f"{s}_kappa_logg_hinges"] = dict(zip(map(str, G_KNOTS), np.asarray(q["kg"]).round(4).tolist()))
        if "llapse" in q: out[f"{s}_lapse"] = float(jax.nn.sigmoid(q["llapse"]))
        if "llapse_low" in q: out[f"{s}_lapse_low_stake"] = float(jax.nn.sigmoid(q["llapse"] + q["llapse_low"]))
        if "kgl" in q: out[f"{s}_kappa_lowg_per_logg"] = float(q["kgl"])
        if "srg" in q: out[f"{s}_logsr_per_logg"] = float(q["srg"])
        if "kt" in q: out[f"{s}_kappa_second_half"] = dict(shift=float(q["kt"][0]), logg_slope_shift=float(q["kt"][1]))
        if "srt" in q: out[f"{s}_sr_second_half_mult"] = float(np.exp(q["srt"]))
        if "ks" in q: out[f"{s}_kappa_state"] = dict(zip(STATE_TERMS, np.asarray(q["ks"]).round(4).tolist()))
        if "re" in q: out[f"{s}_re"] = dict(sd_kappa=float(np.exp(q["re"][0])), sd_logsr=float(np.exp(q["re"][1])), rho=float(np.tanh(q["re"][2])))
    return out


def base_theta():
    return np.array(json.load(open("data/joint.json"))["theta"])


def predict(data, th, spec, side, re_post=None):
    """Per row (data order): P(challenge | w, state) and P(challenge, call wrong | w, state) under the fitted model.
    Random effects: plug in the EB posterior means (re_post) or, if None, integrate over the population."""
    Dn = data[side]; n = Dn["n"]
    P = unpack(jnp.asarray(th), spec); q = P[side]
    base = {k: (jnp.asarray(v) if k not in ("players", "n", "nch") else v) for k, v in Dn.items()}
    out = []
    for y in (True, False):
        D = dict(base, nch=n, y=jnp.full(n, y))
        if spec.get("re") and re_post is None:
            a, b, lw = re_nodes(q, spec.get("gh", 7))
            L = row_terms(D, P, q, spec, a[None, :], b[None, :])
            out.append(np.asarray((jnp.exp(L) * jnp.exp(lw)[None, :]).sum(1)))
        elif spec.get("re"):
            out.append(np.asarray(jnp.exp(row_terms_rowwise(D, P, q, spec, re_post, Dn))))
        else:
            out.append(np.asarray(jnp.exp(row_terms(D, P, q, spec))))
    p1, p0 = out
    return p1 + p0, p1


def row_terms_rowwise(D, P, q, spec, re_post, Dn):
    """row terms with each row's player effect plugged in: a enters kappa as an offset column, b shifts log sr."""
    a = jnp.asarray(re_post.a.values[Dn["pidx"]]); b = jnp.asarray(re_post.b.values[Dn["pidx"]])
    q2 = dict(q); spec2 = {k: v for k, v in spec.items() if k != "re"}
    D2 = dict(D, X=jnp.column_stack([D["X"], a]))
    q2["beta"] = jnp.concatenate([q["beta"], jnp.ones(1)])
    q2["lsr"] = q["lsr"] + b
    return row_terms(D2, P, q2, spec2)
