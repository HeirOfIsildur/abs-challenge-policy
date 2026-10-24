(* HONZA-92: kernel checks of the algebraic steps that the Supplementary Material (S1) adds to theory.md.
   S1  substitution q = u g/(g+d): pg - (1-p) d = g (1-u) on the line, and dq = g/(g+d) du.
   S2  domination in the proof of Theorem 3: with c = x0/(1-x0),
         g <= c d  =>  u g/(g+d) <= x0 for u in [0,1]   (the local density bound applies),
         g >  c d  =>  d g^2/(g+d) <= g^2/c              (the large-stake part is bounded by g^2/c).
   S3  bound (ii): on the disagreement interval the integrand is (g+D)|p - D/(g+D)|; with density <= M,
         M (g+D) L^2/2 with L the interval length equals M g^2 (D-de)^2 / (2 (g+D)(g+de)^2).
   S4  rate t^(1/(1+a)): (x+f)^(1+a) - x^(1+a) = (1+a) xi^a f for some xi in [x, x+f] (mean value form): the
         derivative and its monotonicity symbolically, the bracketing (1+a) x^a f <= ... <= (1+a) (x+f)^a f at fixed a.
   S5  crude bound: g^2/((g+D)(g+de)^2) <= g/(g+de)^2 <= 1/(4 de).
   S6  Theorem S2(c) (HONZA-114). (a) the old hypothesis P(p > 1-eps, g > 0) > 0 for all eps is not enough:
         q = g = U ~ Unif(0,1) satisfies it, but phi(L) = E (g - q (g+L))^+ = 0 for L >= 1, so Delta_t(1) <= 1;
       (b) the corrected hypothesis (some g0 > 0 with P(p > 1-eps, g > g0) > 0 for all eps) gives phi(L) > 0:
         with eps = g0/(2 (g0+L)), every (q, g) with q <= eps, g >= g0 has g - q (g+L) >= g0/2. *)
dir = DirectoryName[$InputFileName] <> "../results/";
res = <|"wolframVersion" -> $Version|>;
SetAttributes[ok, HoldRest];
ok[name_, expr0_] := Module[{expr = TimeConstrained[expr0, 60, $Aborted]}, Print[name, " ", InputForm[expr]]; res[name] = ToString[InputForm[expr]]; If[expr =!= True, Print["FAILED ", name]]];
pos = g > 0 && d > 0 && 0 <= u <= 1;

ok["S1_integrand_on_line", FullSimplify[(1 - q) g - q d == g (1 - u) /. q -> u g/(g + d), pos]];
ok["S1_jacobian", FullSimplify[D[u g/(g + d), u] == g/(g + d), pos]];
ok["S2_small_stake_in_local_window", Resolve[ForAll[{g, d, u, x0}, g > 0 && d > 0 && 0 <= u <= 1 && 0 < x0 < 1 && g <= x0/(1 - x0) d,
    u g/(g + d) <= x0], Reals]];
ok["S2_large_stake_bound", Resolve[ForAll[{g, d, x0}, g > 0 && d > 0 && 0 < x0 < 1 && g > x0/(1 - x0) d,
    d g^2/(g + d) <= g^2 (1 - x0)/x0], Reals]];
ok["S2_small_stake_bound", Resolve[ForAll[{g, d}, g > 0 && d > 0, d g^2/(g + d) <= g^2], Reals]];
With[{a0 = DD/(g + DD), a1 = de/(g + de), as = g > 0 && DD > 0 && de > 0},
  ok["S3_integrand_form", FullSimplify[Abs[p (g + DD) - DD] == (g + DD) Abs[p - a0], as && Element[p, Reals]]];
  ok["S3_bound_equals_closed_form", FullSimplify[MM (g + DD) (a0 - a1)^2/2 == MM g^2 (DD - de)^2/(2 (g + DD) (g + de)^2), as]];
  ok["S3_density_integral_is_upper_bound", FullSimplify[Integrate[(g + DD) (a0 - p), {p, a1, a0}] == (g + DD) (a0 - a1)^2/2, as && DD > de]]];
(* S4 via the mean value theorem: y -> y^(1+a) has derivative (1+a) y^a, which is nondecreasing for a >= 0, so the
   increment over [x, x+f] lies between (1+a) x^a f and (1+a) (x+f)^a f *)
ok["S4_derivative", FullSimplify[D[y^(1 + a), y] == (1 + a) y^a, y > 0 && a >= 0]];
ok["S4_derivative_nondecreasing", FullSimplify[D[(1 + a) y^a, y] >= 0, y > 0 && a >= 0]];
ok["S4_bracket_a_in_{0,1/2,1,2,3}", And @@ Table[Resolve[ForAll[{x, f}, x > 0 && f >= 0,
     (1 + a0) x^a0 f <= (x + f)^(1 + a0) - x^(1 + a0) <= (1 + a0) (x + f)^a0 f], Reals], {a0, {0, 1/2, 1, 2, 3}}]];
ok["S5_crude", Resolve[ForAll[{g, DD, de}, g > 0 && DD >= 0 && de > 0, g^2/((g + DD) (g + de)^2) <= 1/(4 de)], Reals]];
With[{phiU = Integrate[u (1 - u - L), {u, 0, 1 - L}, Assumptions -> 0 < L < 1]},
  ok["S6a_counterexample_phi_positive_below_1", Resolve[ForAll[L, 0 < L < 1, phiU > 0], Reals]];
  ok["S6a_counterexample_phi_zero_from_1", Resolve[ForAll[{u, L}, 0 <= u <= 1 && L >= 1, u (1 - u - L) <= 0], Reals]];
  ok["S6a_counterexample_hypothesis_old", Resolve[ForAll[eps, 0 < eps < 1, Integrate[1, {u, 0, eps}] > 0], Reals]]];
ok["S6b_corrected_hypothesis_bound", Resolve[ForAll[{g, q, g0, L}, g0 > 0 && L >= 0 && g >= g0 && 0 <= q <= g0/(2 (g0 + L)),
    g - q (g + L) >= g0/2], Reals]];
(* the exact recursion Delta_t = Delta_{t-1} + phi(Delta_{t-1}) under q = g = U stays below 1 *)
With[{phi = Function[L, If[L >= 1, 0, (1 - L)^3/6]]},
  ok["S6a_phi_closed_form", FullSimplify[Integrate[u (1 - u - L), {u, 0, 1 - L}] == (1 - L)^3/6, 0 < L < 1]];
  ok["S6a_Delta_2000_below_1", Module[{x = 0.}, Do[x = x + phi[x], {2000}]; res["S6a_Delta_2000"] = ToString[N[x, 6]]; 0 < x < 1]]];
Export[dir <> "supplement-check.json", res, "JSON"];
Print[ExportString[res, "JSON"]];
