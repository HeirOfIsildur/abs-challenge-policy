(* Kernel checks for the theory section (theory.md).
   Model: t = opportunities left, c = challenges held; at each opportunity (g, p) ~ F_t, g > 0 stake, p calibrated
   success probability.  Retention: success gains g and keeps the challenge, failure loses it.
   V_t(c) = V_{t-1}(c) + phi_t(Delta_{t-1}(c)),  phi_t(d) = E_t[(p g - (1-p) d)^+],  V_t(0) = V_0(.) = 0.
   Output: results/proofs.json.  Deterministic only (exact rationals, closed forms, fixed-point iteration). *)
dir = DirectoryName[$InputFileName] <> "../results/";
res = <|"wolframVersion" -> $Version|>;
str[e_] := ToString[InputForm[e]];
pos[x_] := Max[x, 0];
ts = {10, 100, 1000, 10^4, 10^5, 10^6};

(* ===================== T1. Monotone thresholds ===================== *)
(* T1a pointwise lemmas: for fixed (p,g), 0<=p<=1, g>=0, j(x) = (p g - (1-p) x)^+ is nonincreasing in x with slope
   in [-1,0], so x + j(x) is nondecreasing; expectations inherit all three, for ANY law F_t. *)
j[x_] := Max[p g - (1 - p) x, 0];
res["T1a_phi_nonincreasing"] = str[Reduce[ForAll[{p, g, x1, x2},
     0 <= p <= 1 && g >= 0 && x1 <= x2, j[x2] <= j[x1]], Reals]];
res["T1a_psi_nondecreasing"] = str[Reduce[ForAll[{p, g, x1, x2},
     0 <= p <= 1 && g >= 0 && x1 <= x2, x1 + j[x1] <= x2 + j[x2]], Reals]];
res["T1a_phi_slope_in_[-1,0]"] = str[Reduce[ForAll[{p, g, x1, x2},
     0 <= p <= 1 && g >= 0 && x1 < x2, -1 <= (j[x2] - j[x1])/(x2 - x1) <= 0], Reals]];
(* T1b induction step.  fa = phi(a) etc. satisfy the lemma (phi >= 0 nonincreasing, d + phi(d) nondecreasing).
   a = Delta_{t-1}(c-1) >= b = Delta_{t-1}(c) >= e = Delta_{t-1}(c+1) >= 0;
   Delta_t(c) = b + fb - fa,  Delta_t(c+1) = e + fe - fb. *)
hyp = a >= b >= e >= 0 && fa <= fb <= fe && fa >= 0 && e + fe <= b + fb && b + fb <= a + fa;
res["T1b_concavity_step"] = str[Reduce[ForAll[{a, b, e, fa, fb, fe}, hyp, e + fe - fb <= b + fb - fa], Reals]];
res["T1b_nonneg_step"] = str[Reduce[ForAll[{a, b, e, fa, fb, fe}, hyp, b + fb - fa >= 0], Reals]];
res["T1b_monotone_in_t_step"] = str[Reduce[ForAll[{a, b, e, fa, fb, fe}, hyp, b + fb - fa >= b], Reals]];
(* c = 1 boundary: Delta_t(1) = b + fb (no phi term for c = 0), Delta_t(2) = e + fe - fb *)
hyp1 = b >= e >= 0 && fb <= fe && fb >= 0 && e + fe <= b + fb;
res["T1b_concavity_step_c1"] = str[Reduce[ForAll[{b, e, fb, fe}, hyp1, e + fe - fb <= b + fb], Reals]];
(* T1c exact rational DP on seeded random discrete laws: iid and time-varying, p and g dependent *)
SeedRandom[74];
randLaw[] := Module[{n = RandomInteger[{1, 5}], ps, gs, ws},
   ps = RandomChoice[Join[Range[0, 20]/20, {1/3, 2/3, 99/100}], n]; gs = RandomInteger[{1, 40}, n]/10;
   ws = RandomInteger[{1, 9}, n]; {ps, gs, ws/Total[ws]}];
phiL[{ps_, gs_, ws_}, d_, "ret"] := Total[ws (pos /@ (ps gs - (1 - ps) d))];
phiL[{ps_, gs_, ws_}, d_, "cons"] := Total[ws (pos /@ (ps gs - d))];
dpExact[laws_, C_, rule_] := Module[{V = ConstantArray[0, C + 1], hist = {}, dl, inc},
   Do[dl = Differences[V];
    inc = Table[phiL[laws[[k]], dl[[c]], rule], {c, C}];
    V = Join[{0}, Rest[V] + inc]; AppendTo[hist, Differences[V]], {k, Length[laws]}];
   hist];
checkLaws[rule_, varying_] := Module[{ok = True, okT = True, okN = True, h, laws},
   Do[laws = If[varying, Table[randLaw[], {25}], ConstantArray[randLaw[], 25]];
    h = dpExact[laws, 4, rule];
    okT = okT && And @@ Flatten[Table[h[[t + 1, c]] >= h[[t, c]], {t, 24}, {c, 4}]];
    ok = ok && And @@ Flatten[Table[h[[t, c + 1]] <= h[[t, c]], {t, 25}, {c, 3}]];
    okN = okN && And @@ Flatten[Map[# >= 0 &, h, {2}]], {100}];
   <|"Delta nondecreasing in t" -> okT, "Delta nonincreasing in c" -> ok, "Delta >= 0" -> okN, "nLaws" -> 100,
     "arithmetic" -> "exact rationals"|>];
res["T1c_exact_random_laws"] = <|"retention_iid" -> checkLaws["ret", False], "retention_time_varying" -> checkLaws["ret", True],
   "consumption_time_varying" -> checkLaws["cons", True]|>;
(* T1d the result needs (p,g) exogenous to c: if opportunity quality depends on c, concavity can fail *)
cdep = Module[{V = {0, 0, 0}, dl},
   Do[dl = Differences[V]; V = {0, V[[2]] + pos[1/10 - 9/10 dl[[1]]], V[[3]] + pos[9/10 - 1/10 dl[[2]]]}, {5}]; Differences[V]];
res["T1d_c_dependent_law_Delta(1),Delta(2) (p=1/10 at c=1, p=9/10 at c=2, g=1)"] = str[cdep];
res["T1d_concavity_fails"] = cdep[[2]] > cdep[[1]];

(* ===================== T2. sqrt(t) law ===================== *)
(* T2a phi for q = 1-p with density a q^(a-1) (fixed g); generic smooth q-density h + h1 q + h2 q^2 *)
phiA = FullSimplify[Integrate[(g - q (g + d)) aa q^(aa - 1), {q, 0, g/(g + d)}, Assumptions -> g > 0 && d > 0 && aa > 0],
   g > 0 && d > 0 && aa > 0];
res["T2a_phi_powerlaw"] = str[phiA];
res["T2a_lim_d^a_phi"] = str[Limit[d^aa phiA, d -> Infinity, Assumptions -> g > 0 && aa > 0]];
phiG = Integrate[(g - q (g + d)) (h + h1 q + h2 q^2), {q, 0, g/(g + d)}, Assumptions -> g > 0 && d > 0];
res["T2a_lim_d_phi_generic_density"] = str[Limit[d phiG, d -> Infinity, Assumptions -> g > 0]];
res["T2a_phi_uniform_p"] = str[FullSimplify[Integrate[p (g + d) - d, {p, d/(g + d), 1}, Assumptions -> g > 0 && d > 0]]];
(* T2b square increments: Delta_t(1)^2 - Delta_{t-1}(1)^2 = kappa(Delta_{t-1}) := 2 Delta phi + phi^2 *)
res["T2b_square_increment"] = str[FullSimplify[(dd + ph)^2 - dd^2 - (2 dd ph + ph^2)]];
(* T2c fluid limit Delta' = K_a Delta^-a  =>  Delta = ((1+a) K_a t)^(1/(1+a)) *)
res["T2c_fluid_solution"] = str[DSolve[{y'[s] == kk y[s]^-aa, y[0] == 0}, y[s], s]];
(* T2d iterate Delta_t(1) to 1e6 for three laws of p near 1; g in {1/2,1,2} w.p. {1/2,1/3,1/6} indep of p *)
gv = {1/2, 1, 2}; gw = {1/2, 1/3, 1/6};
phi1[gg_, d_] = FullSimplify[Integrate[(p (gg + d) - d) 2 p, {p, d/(gg + d), 1}], gg > 0 && d > 0];        (* f_p = 2p, h = 2 *)
phi2[gg_, d_] = FullSimplify[Integrate[(p (gg + d) - d) 2 (1 - p), {p, d/(gg + d), 1}], gg > 0 && d > 0];  (* f_p = 2(1-p), alpha = 2 *)
phi3[gg_, d_] = FullSimplify[(1/10) gg + (9/10) Integrate[p (gg + d) - d, {p, d/(gg + d), 1}], gg > 0 && d > 0]; (* atom at p = 1 *)
res["T2d_phi_density2p"] = str[phi1[gg, d]]; res["T2d_phi_density2(1-p)"] = str[phi2[gg, d]];
K1 = Limit[d Total[gw (phi1[#, d] & /@ gv)], d -> Infinity];
K2 = Limit[d^2 Total[gw (phi2[#, d] & /@ gv)], d -> Infinity];
res["T2d_K1=lim d phi"] = str[K1]; res["T2d_E[h g^2]/2"] = str[Total[gw 2 gv^2]/2];
res["T2d_K2=lim d^2 phi"] = str[K2]; res["T2d_E[g^3]/3"] = str[Total[gw gv^3]/3];
f1 = Compile[{{d, _Real}}, Evaluate[Total[gw (phi1[#, d] & /@ gv)]]];
f2 = Compile[{{d, _Real}}, Evaluate[Total[gw (phi2[#, d] & /@ gv)]]];
f3 = Compile[{{d, _Real}}, Evaluate[Total[gw (phi3[#, d] & /@ gv)]]];
iter[f_] := Module[{x = 0., out = {}}, Do[x = x + f[x]; If[MemberQ[ts, t], AppendTo[out, {t, x}]], {t, 1, Max[ts]}]; out];
r1 = iter[f1]; r2 = iter[f2]; r3 = iter[f3];
res["T2d_alpha1_Delta/sqrt(2 K1 t)"] = {#[[1]], #[[2]]/Sqrt[2 N[K1] #[[1]]]} & /@ r1;
res["T2d_alpha2_Delta/(3 K2 t)^(1/3)"] = {#[[1]], #[[2]]/(3 N[K2] #[[1]])^(1/3)} & /@ r2;
res["T2d_atom_Delta/t"] = {#[[1]], #[[2]]/#[[1]]} & /@ r3;
res["T2d_atom_limit E[g; p=1]"] = str[(1/10) Total[gw gv]];
(* T2e c >= 2: fluid constants Delta_t(c) ~ k_c sqrt(2 K t), k_1 = 1, k_c^2 + k_c/k_{c-1} = 1 *)
kc = NestList[(-1/# + Sqrt[1/#^2 + 4])/2 &, 1, 5];
res["T2e_k2_exact"] = str[FullSimplify[kc[[2]]]];
res["T2e_k2_equals_GoldenRatio-1"] = FullSimplify[kc[[2]] == GoldenRatio - 1];
res["T2e_k_c c=1..6"] = N[kc]; res["T2e_cumsum_k (V_t(c)/sqrt(2Kt))"] = N[Accumulate[kc]];
(* drift of y = Delta(c)^2/(2K t) in log-time: y' = 1 - sqrt(y)/k_{c-1} - y; unique root k_c^2, stable *)
res["T2e_drift_root_c2"] = str[Solve[1 - Sqrt[yy] - yy == 0 && yy > 0, yy, Reals]];
res["T2e_drift_root_check"] = FullSimplify[((GoldenRatio - 1)^2) == (3 - Sqrt[5])/2];
res["T2e_drift_decreasing"] = str[Reduce[ForAll[yy, yy > 0, D[1 - Sqrt[yy]/kp - yy, yy] < 0], kp, Reals] // Quiet];
stepC = Compile[{{v, _Real, 1}}, Module[{ph = 1/(2 (1 + v))}, v + ph - Join[{0.}, Most[ph]]]];
(* uniform p, g = 1: K = 1/2, sqrt(2Kt) = sqrt(t) *)
rc = Module[{v = {0., 0., 0., 0.}, out = {}}, Do[v = stepC[v]; If[MemberQ[ts, t], AppendTo[out, {t, v/Sqrt[t]}]], {t, 1, 10^6}]; out];
res["T2e_uniform_Delta_t(c)/sqrt(t) c=1..4"] = rc;
(* T2f consumption: x + E(Y - x)^+ <= s whenever x <= s and Y <= s a.s.  =>  Delta^cons_t(c) <= ess sup(p g) *)
res["T2f_consumption_bound_step"] = str[Reduce[ForAll[{x, y, s}, x <= s && y <= s, x + Max[y - x, 0] <= s], Reals]];
res["T2f_uniform_cons_Delta at t=1e6 (sup pg = 1)"] = Module[{x = 0.}, Do[x = x + (1 - x)^2/2, {10^6}]; x];
(* T2g pre-asymptotic kappa(d) = 2 d phi + phi^2 for uniform p, fixed g *)
kap = FullSimplify[2 d (g^2/(2 (g + d))) + (g^2/(2 (g + d)))^2];
res["T2g_kappa_uniform"] = str[kap];
res["T2g_dkappa/dd"] = str[FullSimplify[D[kap, d], g > 0 && d > 0]];
res["T2g_kappa(0)/g^2, kappa(inf)/g^2"] = str[{Simplify[kap /. d -> 0]/g^2, Limit[kap, d -> Infinity, Assumptions -> g > 0]/g^2}];
res["T2g_argmin_kappa (g=1, d>=0)"] = str[Minimize[{kap /. g -> 1, d >= 0}, d]];
res["T2g_kappa(d)/g^2 at d/g = 0, .25, .5, 1, 2"] = N[(kap /. g -> 1) /. d -> {0, 1/4, 1/2, 1, 2}];
(* T2h time-varying stakes: g_s = 3 every third opportunity else 1, uniform p: Delta_t^2 / sum_s g_s^2 -> 1 *)
r4 = Module[{x = 0., ks = 0., out = {}, gg},
   Do[gg = If[Mod[t, 3] == 0, 3., 1.]; x = x + gg^2/(2 (gg + x)); ks += gg^2; If[MemberQ[ts, t], AppendTo[out, {t, x^2/ks}]], {t, 1, 10^6}]; out];
res["T2h_time_varying_Delta^2/sum(g_s^2)"] = r4;

(* T2i calibrated 2026 DP (results/delta_by_remaining.json rows {home, k, n_remaining, Delta1, Delta2}):
   the square-increment identity predicts Delta^2 ~ linear in n; the c=2 / c=1 ratio vs the fluid limit k_2 = 0.618;
   T1 monotonicity on the calibrated DP (it adds slots and the extra-inning top-up, outside the theorem). *)
cal = Import[dir <> "delta_by_remaining.json", "RawJSON"];
calSide[home_] := Module[{d = Select[cal, #[[1]] == home &], lmSq},
   lmSq = LinearModelFit[{#[[3]], #[[4]]^2} & /@ d, {n}, n];
   <|"Delta1^2 = a + b n: coef" -> lmSq["BestFitParameters"], "R2" -> lmSq["RSquared"],
     "Delta2/Delta1 at n_first, mid" -> {d[[1, 5]]/d[[1, 4]], d[[Ceiling[Length[d]/2], 5]]/d[[Ceiling[Length[d]/2], 4]]},
     "Delta1>=Delta2 all rows" -> And @@ (#[[4]] >= #[[5]] & /@ d),
     "Delta1, Delta2 nondecreasing in n" -> (And @@ (#[[1]] >= #[[2]] & /@ Partition[d[[All, 4]], 2, 1]) &&
        And @@ (#[[1]] >= #[[2]] & /@ Partition[d[[All, 5]], 2, 1]))|>];
res["T2i_calibrated_home"] = calSide[1]; res["T2i_calibrated_away"] = calSide[0];

(* ===================== T3. Flat optimum ===================== *)
(* T3a pointwise regret of a constant-delta rule against continuation D (= Delta_{t-1}(c)).
   Mis-decision for p between pD = D/(g+D) and pd = de/(g+de); Q-gap there is |p(g+D) - D|. *)
pD = dD/(g + dD); pd = de/(g + de);
res["T3a_gap_at_pd"] = str[FullSimplify[Abs[pd (g + dD) - dD] == g Abs[dD - de]/(g + de), g > 0 && dD > 0 && de > 0]];
res["T3a_cutoff_distance"] = str[FullSimplify[pD - pd, g > 0 && dD > 0 && de > 0]];
regU = FullSimplify[Integrate[(g + dD) (p - pD), {p, pD, pd}, Assumptions -> g > 0 && de > dD > 0], g > 0 && de > dD > 0];
regL = FullSimplify[Integrate[dD - p (g + dD), {p, pd, pD}, Assumptions -> g > 0 && dD > de > 0], g > 0 && dD > de > 0];
res["T3a_regret_density1_(de>D)"] = str[regU];
res["T3a_regret_closed_form_both_sides"] = str[FullSimplify[{regU == g^2 (dD - de)^2/(2 (g + dD) (g + de)^2),
     regL == g^2 (dD - de)^2/(2 (g + dD) (g + de)^2)}, g > 0 && dD > 0 && de > 0 && de != dD]];
rq = g^2 (dD - de)^2/(2 (g + dD) (g + de)^2);
res["T3a_envelope: d/dde, d2/dde2 at de=D"] = str[{Simplify[D[rq, de] /. de -> dD], FullSimplify[D[rq, {de, 2}] /. de -> dD]}];
res["T3a_g^2/((g+D)(g+de)^2) <= 1/(4 de)"] = str[Reduce[ForAll[{g, dD, de}, g > 0 && dD >= 0 && de > 0,
     g^2/((g + dD) (g + de)^2) <= 1/(4 de)], Reals]];
(* T3b stylized team-game: T opportunities, c0 = 2, stakes rise late, p mostly low with some mass near 1 *)
fdens[p_] := 9/10 4 (1 - p)^3 + 1/10 4 p^3;
res["T3b_f_convex"] = str[Reduce[ForAll[p, 0 <= p <= 1, D[fdens[p], {p, 2}] >= 0], Reals]];
I0[a_, b_] = Integrate[fdens[p], {p, a, b}]; I1[a_, b_] = Integrate[p fdens[p], {p, a, b}];
TT = 60; gBase = {3, 8, 20, 50}/1000; gW = {4, 3, 2, 1}/10;
gAt[t_] := N[gBase (6/10 + 12/10 (1 - t/TT))];   (* t = opportunities left: leverage rises late *)
gainOpt[gs_, d_] := Total[gW Table[With[{gg = gs[[k]]}, (gg + d) I1[d/(gg + d), 1] - d I0[d/(gg + d), 1]], {k, 4}]];
solveOpt[] := Module[{V = {0., 0., 0.}, hist = {}},
   Do[AppendTo[hist, Rest[Differences[Prepend[V, 0.]]]];   (* hist[[t]] = {Delta_{t-1}(1), Delta_{t-1}(2)} *)
    V = {0., V[[2]] + gainOpt[gAt[t], V[[2]] - V[[1]]], V[[3]] + gainOpt[gAt[t], V[[3]] - V[[2]]]}, {t, 1, TT}];
   {V, hist}];
evalDelta[de_] := Module[{V = {0., 0., 0.}, dl},
   Do[V = Join[{0.}, Table[dl = V[[c + 1]] - V[[c]];
        V[[c + 1]] + Total[gW Table[With[{gg = gAt[t][[k]]}, (gg + dl) I1[de/(gg + de), 1] - dl I0[de/(gg + de), 1]], {k, 4}]], {c, 2}]], {t, 1, TT}];
   V];
{Vstar, hist} = solveOpt[];
lossRel[de_?NumericQ] := 1 - evalDelta[de][[3]]/Vstar[[3]];
grid = Table[{de, lossRel[de]}, {de, 0.001, 0.06, 0.001}];
best = First[SortBy[grid, Last]];
deBest = de /. Last[FindMinimum[lossRel[de], {de, best[[1]], Max[best[[1]] - 0.001, 0.0002], best[[1]] + 0.001}]];
(* occupation measure of c under the delta rule (forward in time), exact regret identity, and the T3 bound *)
occAndTerms[de_] := Module[{pc = {0., 0., 1.}, exact = 0., bnd = 0., crude = 0., Dl, gs, lo, hi, M, new, used = {}, wsum = 0.},
   Do[gs = gAt[t];
    Do[If[pc[[c + 1]] > 0, Dl = hist[[t, c]];
      Do[With[{gg = gs[[k]]}, lo = Min[Dl/(gg + Dl), de/(gg + de)]; hi = Max[Dl/(gg + Dl), de/(gg + de)];
         M = Max[fdens[lo], fdens[hi]];   (* f convex (T3b_f_convex): max on [lo,hi] at an endpoint *)
         exact += pc[[c + 1]] gW[[k]] Abs[(gg + Dl) I1[lo, hi] - Dl I0[lo, hi]];
         bnd += pc[[c + 1]] gW[[k]] M gg^2 (Dl - de)^2/(2 (gg + Dl) (gg + de)^2)], {k, 4}];
      crude += pc[[c + 1]] Max[fdens[0], fdens[1]] (Dl - de)^2/(8 de);
      AppendTo[used, Dl]], {c, 2}];
    new = pc;
    Do[With[{lq = Total[gW Table[With[{gg = gs[[k]]}, I0[de/(gg + de), 1] - I1[de/(gg + de), 1]], {k, 4}]]},
       new[[c + 1]] -= pc[[c + 1]] lq; new[[c]] += pc[[c + 1]] lq], {c, 1, 2}]; pc = new, {t, TT, 1, -1}];
   <|"exact" -> exact, "bound" -> bnd, "crude" -> crude, "DeltaRange" -> MinMax[used]|>];
oc = occAndTerms[deBest];
vd = evalDelta[deBest][[3]];
res["T3b_setup"] = "T=60, c0=2, g in {3,8,20,50}/1000 x (0.6+1.2(1-t/T)), w {.4,.3,.2,.1}, f(p)=0.9*4(1-p)^3+0.1*4p^3";
res["T3b_Vstar"] = Vstar[[3]]; res["T3b_best_delta"] = deBest; res["T3b_V_best_delta"] = vd;
res["T3b_rel_loss_best_delta"] = 1 - vd/Vstar[[3]];
res["T3b_regret_identity {Vstar-Vdelta, sum occupation-weighted gaps}"] = {Vstar[[3]] - vd, oc["exact"]};
res["T3b_quadratic_bound {abs, rel}"] = {oc["bound"], oc["bound"]/Vstar[[3]]};
res["T3b_crude_bound M(D-de)^2/(8de) {abs, rel}"] = {oc["crude"], oc["crude"]/Vstar[[3]]};
res["T3b_Delta_range_on_path"] = oc["DeltaRange"];
res["T3b_rel_loss_vs_delta"] = Select[grid, MemberQ[{1, 2, 5, 10, 20, 30, 50}, Round[1000 #[[1]]]] &];

(* ===================== T4. Rule comparison ===================== *)
(* retention (MLB), consumption (classic), NBA cap: state A (first challenge; success -> B, failure -> none),
   state B (one consumable challenge).  NBA-A gain E[(p(g+V_B) - V_A)^+]. *)
tsR = {1, 2, 3, 5, 10, 20, 50, 100, 1000, 10^4};
gcU[d_] := If[d < 1, (1 - d)^2/2, 0.];
uniRules = Module[{R = {0., 0., 0.}, Cn = {0., 0., 0.}, nA = 0., nB = 0., out = {}},
   Do[R = {0., R[[2]] + 1/(2 (1 + R[[2]])), R[[3]] + 1/(2 (1 + R[[3]] - R[[2]]))};
    {nA, nB} = {nA + If[1 + nB > nA, (1 + nB - nA)^2/(2 (1 + nB)), 0.], nB + gcU[nB]};
    Cn = {0., Cn[[2]] + gcU[Cn[[2]]], Cn[[3]] + gcU[Cn[[3]] - Cn[[2]]]};
    If[MemberQ[tsR, t], AppendTo[out, <|"t" -> t, "cons1" -> Cn[[2]], "NBA" -> nA, "ret1" -> R[[2]], "cons2" -> Cn[[3]], "ret2" -> R[[3]],
       "ordering cons1<=NBA<=min(ret1,cons2), cons2<=ret2" -> (Cn[[2]] <= nA <= Min[R[[2]], Cn[[3]]] && Cn[[3]] <= R[[3]])|>]], {t, 1, 10^4}];
   out];
res["T4_uniform_p_g1"] = uniRules;
consC[c_, tmax_] := Module[{V = ConstantArray[0., c + 1], out = {}},
   Do[V = Join[{0.}, Table[V[[k + 1]] + gcU[V[[k + 1]] - V[[k]]], {k, c}]]; AppendTo[out, V[[c + 1]]], {tmax}]; out];
ret1 = Rest[NestList[# + 1/(2 (1 + #)) &, 0., 400]];
(* first t from which retention with ONE challenge strictly beats consumption with c (for every later t <= 400) *)
crossFrom[c_] := Module[{cc = consC[c, 400]}, 1 + Max[Select[Range[400], ret1[[#]] <= cc[[#]] &]]];
res["T4_ret1>cons2_for_all_t>="] = crossFrom[2]; res["T4_ret1>cons3_for_all_t>="] = crossFrom[3];
res["T4_ret1>cons5_for_all_t>="] = crossFrom[5];
gainCons[gs_, d_] := Total[gW Table[With[{gg = gs[[k]]}, If[d < gg, gg I1[d/gg, 1] - d I0[d/gg, 1], 0.]], {k, 4}]];
gainNBA[gs_, a_, b_] := Total[gW Table[With[{gg = gs[[k]]}, If[a < gg + b, (gg + b) I1[a/(gg + b), 1] - a I0[a/(gg + b), 1], 0.]], {k, 4}]];
mlb = Module[{R = {0., 0., 0.}, Cn = {0., 0., 0.}, nA = 0., nB = 0., gs},
   Do[gs = gAt[t];
    R = {0., R[[2]] + gainOpt[gs, R[[2]]], R[[3]] + gainOpt[gs, R[[3]] - R[[2]]]};
    {nA, nB} = {nA + gainNBA[gs, nA, nB], nB + gainCons[gs, nB]};
    Cn = {0., Cn[[2]] + gainCons[gs, Cn[[2]]], Cn[[3]] + gainCons[gs, Cn[[3]] - Cn[[2]]]}, {t, 1, TT}];
   <|"cons1" -> Cn[[2]], "NBA" -> nA, "ret1" -> R[[2]], "cons2" -> Cn[[3]], "ret2" -> R[[3]]|>];
res["T4_T3b_law_T60"] = mlb;

Export[dir <> "proofs.json", res, "JSON"];
Print[ExportString[res, "JSON"]];
