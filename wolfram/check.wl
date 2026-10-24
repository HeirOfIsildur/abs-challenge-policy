(* ABS decisive test (HONZA-71): kernel checks of the closed forms used by the Python pipeline. *)
dir = DirectoryName[$InputFileName] <> "../results/";
res = <|"wolframVersion" -> $Version|>;
phi[x_] := PDF[NormalDistribution[], x]; Phi[x_] := CDF[NormalDistribution[], x];

(* A. Retained-challenge cutoff: challenge iff p(g+V1)+(1-p)V0 >= V1, Delta = V1 - V0 > 0 *)
a = Reduce[p (g + v0 + dl) + (1 - p) v0 >= v0 + dl && g > 0 && dl > 0 && 0 <= p <= 1, p, Reals];
res["A_cutoff"] = ToString[InputForm[a]];
res["A_equiv_counterexample"] = ToString[InputForm[Reduce[g > 0 && dl > 0 && 0 <= p <= 1 &&
   Xor[p (g + dl) >= dl, p >= dl/(g + dl)], {p, g, dl}, Reals]]];

(* B. Threshold monotone: increasing in Delta, decreasing in stake g *)
pst = dl/(g + dl);
res["B_dDelta"] = ToString[InputForm[FullSimplify[D[pst, dl], g > 0 && dl > 0]]];
res["B_dg"] = ToString[InputForm[FullSimplify[D[pst, g], g > 0 && dl > 0]]];
res["B_signs"] = {Simplify[D[pst, dl] > 0, g > 0 && dl > 0], Simplify[D[pst, g] < 0, g > 0 && dl > 0]};

(* C. Probit convolution: E_eta Phi((w+eta-k)/sr) = Phi((w-k)/Sqrt[sm^2+sr^2]) *)
convNum = Table[With[{w = w0, k = k0, sm = sm0, sr = sr0},
    NIntegrate[PDF[NormalDistribution[0, sm], e] Phi[(w + e - k)/sr], {e, -Infinity, Infinity}, WorkingPrecision -> 30] -
     Phi[(w - k)/Sqrt[sm^2 + sr^2]]],
   {w0, {-3, 0, 3/2}}, {k0, {-1, 2}}, {sm0, {518/1000, 849/1000}}, {sr0, {228/100, 3228/1000}}];
res["C_conv_maxabs"] = N[Max[Abs[Flatten[convNum]]]];
(* symbolic route: d/dw of both sides are equal Gaussian densities (convolution of normals), and both -> 0 as w -> -Inf *)
convSym = FullSimplify[Integrate[PDF[NormalDistribution[0, sm], e] PDF[NormalDistribution[0, sr], x - e], {e, -Infinity, Infinity},
     Assumptions -> sm > 0 && sr > 0 && Element[x, Reals]] == PDF[NormalDistribution[0, Sqrt[sm^2 + sr^2]], x], sm > 0 && sr > 0 && Element[x, Reals]];
res["C_conv_symbolic_density"] = ToString[InputForm[convSym]];

(* D. Selection-corrected likelihood term = bivariate normal orthant probability:
   P(W=1, ch=1 | w) = P(eta > -w, eta + eps > k - w), corr = sm/Sqrt[sm^2+sr^2] *)
bvn[w_, k_, sm_, sr_] := Module[{s2 = Sqrt[sm^2 + sr^2], r},
   r = sm/s2;
   CDF[BinormalDistribution[{0, 0}, {1, 1}, r], {w/sm, (w - k)/s2}]];
quad[w_, k_, sm_, sr_] := NIntegrate[PDF[NormalDistribution[0, sm], e] Phi[(w + e - k)/sr], {e, -w, Infinity}, WorkingPrecision -> 25];
dTab = Table[{w, k, N[bvn[w, k, 849/1000, 3228/1000], 25], quad[w, k, 849/1000, 3228/1000]}, {w, {-3/2, -3/10, 2/5, 2}}, {k, {1/2, 3}}];
dTab = Flatten[dTab, 1];
res["D_bvn_vs_quad_maxabs"] = N[Max[Abs[dTab[[All, 3]] - dTab[[All, 4]]]]];

(* E. Flat-prior posterior: w_true ~ flat, s = w_true + eps  =>  P(w_true > 0 | s) = Phi(s/sr) *)
post = Integrate[PDF[NormalDistribution[s, sr], x], {x, 0, Infinity}, Assumptions -> sr > 0 && Element[s, Reals]];
res["E_flat_posterior"] = ToString[InputForm[FullSimplify[post == Phi[s/sr], sr > 0 && Element[s, Reals]]]];

(* F. sqrt law on the calibrated DP: fit Delta_1 = a + b Sqrt[n_remaining] (+ c n) per team side *)
rows = Import[dir <> "delta_by_remaining.json", "RawJSON"];
fitF[home_] := Module[{d = Select[rows, #[[1]] == home &], lm1, lm2},
   lm1 = LinearModelFit[{#[[3]], #[[4]]} & /@ d, {Sqrt[n]}, n];
   lm2 = LinearModelFit[{#[[3]], #[[4]]} & /@ d, {n}, n];
   <|"sqrt_coef" -> lm1["BestFitParameters"], "sqrt_R2" -> lm1["RSquared"],
     "linear_R2" -> lm2["RSquared"], "D1_first" -> d[[1, 4]], "D1_last" -> d[[-1, 4]],
     "n_first" -> d[[1, 3]], "n_last" -> d[[-1, 3]]|>];
res["F_home"] = fitF[1]; res["F_away"] = fitF[0];

Export[dir <> "check.json", res, "JSON"];
Print[ExportString[res, "JSON"]];
