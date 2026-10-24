(* Cricket DRS replication (HONZA-76): kernel checks of the closed forms used by src/drs/. *)
dir = DirectoryName[$InputFileName] <> "../results/drs/";
res = <|"wolframVersion" -> $Version|>;
Phi[x_] := CDF[NormalDistribution[], x];
asm = s > 0 && Element[{mu, x, sS, t}, Reals];

(* A. Posterior: X ~ N(mu,1), S = X + eps, eps ~ N(0,s^2).  X | S ~ N(m, v), m = (mu s^2 + S)/(1+s^2), v = s^2/(1+s^2),
   so p = P(X > 0 | S) = Phi(m/Sqrt[v]).  Check: joint density / marginal of S equals the N(m, v) density. *)
joint = PDF[NormalDistribution[mu, 1], x] PDF[NormalDistribution[0, s], sS - x];
marg = PDF[NormalDistribution[mu, Sqrt[1 + s^2]], sS];
m = (mu s^2 + sS)/(1 + s^2); v = s^2/(1 + s^2);
res["A_posterior"] = ToString[InputForm[FullSimplify[joint/marg == PDF[NormalDistribution[m, Sqrt[v]], x], asm]]];
res["A_marginal"] = ToString[InputForm[FullSimplify[Integrate[joint, {x, -Infinity, Infinity}, Assumptions -> asm] == marg, asm]]];

(* B. Inversion of the review rule: Phi(m(Sstar)/Sqrt[v]) = tau  <=>  Sstar = (1+s^2) Sqrt[v] Phi^-1(tau) - mu s^2 *)
sst = (1 + s^2) Sqrt[v] z - mu s^2;     (* z = Phi^-1(tau) *)
res["B_sstar"] = ToString[InputForm[FullSimplify[(m /. sS -> sst)/Sqrt[v] == z, asm && Element[z, Reals]]]];
res["B_p_increasing_in_S"] = ToString[InputForm[FullSimplify[D[m/Sqrt[v], sS] > 0, asm]]];

(* C. Orthant: P(X > t, S > Sstar) = BVN-CDF at (mu - t, (mu - Sstar)/Sqrt[1+s^2]) with corr 1/Sqrt[1+s^2]
   (i) symbolic: covariance of (-(X-mu), -(S-mu)/Sqrt[1+s^2]) *)
cov = {{1, 1/Sqrt[1 + s^2]}, {1/Sqrt[1 + s^2], 1}};
res["C_corr"] = ToString[InputForm[FullSimplify[Covariance[TransformedDistribution[{-(xx - mu), -(xx + e - mu)/Sqrt[1 + s^2]},
      {xx \[Distributed] NormalDistribution[mu, 1], e \[Distributed] NormalDistribution[0, s]}]] == cov, s > 0]]];
(* (ii) Python's Owen's-T bvn(h,k,r) against the kernel's bivariate normal CDF, 30 digits *)
py = {{0.3, -1.2, 0.6, 0.10842550424680719}, {-0.5, -0.4, 0.9, 0.260231802548192}, {1.1, 2.0, 0.2, 0.84754730089356989},
   {-2, 0.5, 0.7, 0.022710017733099441}, {0.0, 0.4, 0.5, 0.40401032420466043}, {-0.515, 0.8, 0.894, 0.30315935683890149}};
res["C_bvn_maxabs"] = N[Max[Table[With[{h = Rationalize[c[[1]], 0], k = Rationalize[c[[2]], 0], r = Rationalize[c[[3]], 0]},
      Abs[N[CDF[BinormalDistribution[r], {h, k}], 30] - c[[4]]]], {c, py}]]];
(* (iii) the model's q and J(-kappa) at the Test fit (mu = -0.515, s = 1/2, kappa = 0.48, tau = 0.2) by direct integration *)
With[{mu0 = -515/1000, s0 = 1/2, ka = 48/100, tau = 1/5},
  Module[{z0 = InverseCDF[NormalDistribution[], tau], S0, q, J},
   S0 = (1 + s0^2) Sqrt[s0^2/(1 + s0^2)] z0 - mu0 s0^2;
   q = NIntegrate[PDF[NormalDistribution[mu0, Sqrt[1 + s0^2]], y], {y, S0, Infinity}, WorkingPrecision -> 30];
   J = NIntegrate[PDF[NormalDistribution[mu0, 1], xx] (1 - Phi[(S0 - xx)/s0]), {xx, -ka, Infinity}, WorkingPrecision -> 30];
   res["C_sstar"] = N[S0, 17];
   res["C_q_J_diff_vs_python"] = N[{q - 4384197691548738/10^16, J - 38637545304409338/10^17}]]];

(* D. DP review rule with a kept umpire's call.  Review value p(g+U) + (1-p) U1 - l dU vs U1 (no review: wicket falls).
   (i) general: review iff p (g + A) >= l dU with A = U - U1;  (ii) constant rho: l = (1-rho)(1-p) gives p >= D/(g+A+D), D = (1-rho) dU *)
res["D_general"] = ToString[InputForm[FullSimplify[(p (g + u) + (1 - p) u1 - l dU >= u1) == (p (g + u - u1) >= l dU)]]];
res["D_const_counterexample"] = ToString[InputForm[Reduce[g > 0 && aa >= 0 && dU > 0 && 0 <= rho < 1 && 0 <= p <= 1 &&
     Xor[p (g + aa) >= (1 - rho) (1 - p) dU, p >= (1 - rho) dU/(g + aa + (1 - rho) dU)], {p, g, aa, dU, rho}, Reals]]];
(* (iii) a kept umpire's call lowers the cutoff: dp*/drho < 0 *)
res["D_drho"] = ToString[InputForm[FullSimplify[D[(1 - rho) dU/(g + aa + (1 - rho) dU), rho], g > 0 && aa >= 0 && dU > 0]]];

(* D (iv) HONZA-122 no-extra-opportunity DP: an overturn moves on to w+1, review value p g + u1 - l dU vs u1,
   so review iff p g >= l dU (the general rule with A = 0). *)
res["D_noextra"] = ToString[InputForm[FullSimplify[(p g + u1 - l dU >= u1) == (p g >= l dU)]]];
res["D_noextra_is_A0"] = ToString[InputForm[FullSimplify[((p (g + aa) >= l dU) /. aa -> 0) == (p g >= l dU)]]];

(* E. Option value of the last review against expected decisions left: sqrt vs linear vs power fits (calibrated DP) *)
Do[With[{j = Import[dir <> "dp_" <> f <> ".json", "RawJSON"]["main"]["sqrt_law"]},
   Module[{n = j["N"], y = j["delta1"], d, lm1, lm2, pw},
    d = Transpose[{n, y}];
    lm1 = LinearModelFit[Transpose[{Sqrt[n], y}], xx, xx]; lm2 = LinearModelFit[d, xx, xx];
    pw = LinearModelFit[Log[Select[d, #[[2]] > 10^-4 &]], xx, xx];
    res["E_" <> f] = <|"R2_sqrt" -> lm1["RSquared"], "R2_linear" -> lm2["RSquared"], "power_exponent" -> pw["BestFitParameters"][[2]],
       "power_exponent_CI" -> pw["ParameterConfidenceIntervals"][[2]]|>]], {f, {"Test", "ODI", "T20"}}];

Export[dir <> "check.json", res, "JSON"];
Print[ExportString[res, "JSON"]];
