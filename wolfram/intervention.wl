(* HONZA-132: exact checks of the intervention model (src/intervention.py, sim.draw rho).
   Player signal s_p = w + Sqrt[rho] sr e1; behavioural signal s_b = s_p + Sqrt[1 - rho] sr e2; challenge iff s_b > kappa.
   (1) s_b - w ~ N(0, sr^2) for every rho in (0, 1]; (2) so P(challenge | w) does not depend on rho (the likelihood
   cannot identify rho); (3) one-opportunity value of the calibrated cutoff decision (challenge iff p g >= (1 - p) D,
   gain g on success, cost D on failure) under a N(m, tau^2) prior on w is decreasing in the signal noise, so lowering
   rho can only raise the benchmark (deterministic numerical integration on a grid). *)
$Assumptions = sr > 0 && 0 < rho <= 1 && Element[{w, kappa}, Reals];
d1 = NormalDistribution[0, Sqrt[rho] sr]; d2 = NormalDistribution[0, Sqrt[1 - rho] sr];
conv = TransformedDistribution[x + y, {x \[Distributed] d1, y \[Distributed] d2}];
pdfOK = FullSimplify[PDF[conv, z] == PDF[NormalDistribution[0, sr], z], Assumptions -> $Assumptions && 0 < rho < 1 && Element[z, Reals]];
pch = 1 - CDF[NormalDistribution[w, sr], kappa];
dRho = D[pch, rho];
(* value of the cutoff decision with noise sig: prior w ~ N(m, tau), truth W = 1{w > 0}, signal s = w + sig e *)
value[sig_?NumericQ, m_, tau_, g_, dd_] := Module[{p, f},
  p[s_] := Module[{v = 1/(1/tau^2 + 1/sig^2), mu}, mu = v (m/tau^2 + s/sig^2); CDF[NormalDistribution[], mu/Sqrt[v]]];
  f[s_] := PDF[NormalDistribution[m, Sqrt[tau^2 + sig^2]], s] Max[p[s] g - (1 - p[s]) dd, 0];
  NIntegrate[f[s], {s, -Infinity, Infinity}, WorkingPrecision -> 30, PrecisionGoal -> 12, MaxRecursion -> 30]];
grid = Flatten[Table[{m, g, dd}, {m, {-3, -1, 0}}, {g, {1/100, 3/100}}, {dd, {1/250, 1/100}}], 2];
sigs = {1, 3/2, 2, 5/2, 3, 7/2};
vals = Table[{pt, Table[value[s, pt[[1]], 5/2, pt[[2]], pt[[3]]], {s, sigs}]}, {pt, grid}];
mono = And @@ (Max[Differences[#[[2]]]] <= 0 & /@ vals);
res = <|"wolframVersion" -> $Version,
  "convolution_is_N0sr" -> ToString[pdfOK],
  "dPchallenge_drho" -> ToString[InputForm[Simplify[dRho]]],
  "value_decreasing_in_noise_on_grid" -> mono,
  "grid_points" -> Length[grid], "noise_levels" -> ToString[InputForm[sigs]],
  "values" -> ToString[InputForm[Map[{#[[1]], N[#[[2]], 8]} &, vals]]]|>;
Export["results/intervention-check.json", res, "JSON"];
Print[res["wolframVersion"]]; Print["conv: ", pdfOK, "  dP/drho: ", res["dPchallenge_drho"], "  monotone: ", mono];
If[pdfOK =!= True || Simplify[dRho] =!= 0 || mono =!= True, Exit[1]];
