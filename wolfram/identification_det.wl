(* identification on the corrected geometry, where the ABS truth is deterministic in the tracked location.
   Model (one side): W = 1{w > 0}; eps ~ N(0, sr^2); challenge iff w + eps > k; ruling y = W with prob 1 - pi.
     Pch(w)        = Phi((w - k)/sr)
     P(ch, y | w)  = Phi((w - k)/sr) ((1 - pi) 1{y = W} + pi 1{y != W})
   Claims checked:
     D1  the deterministic model is the sm -> 0 limit of the smooth truth model (orthant form of I2).
     D2  (sr, k) are identified by the challenge curve alone: the 2-point Jacobian determinant of (Pch(w1), Pch(w2))
         w.r.t. (sr, k) is phi1 phi2 (w1 - w2)/sr^3 (non-zero iff w1 != w2); rank 2 on the data grid.
     D3  pi separates: the log-likelihood is additive in pi and (sr, k), so the cross information is exactly 0.
     D4  tracking noise c between measured and ABS location: Pch integrates to Phi((w - k)/Sqrt[sr^2 + c^2]) exactly,
         so the deterministic fit estimates Sqrt[sr^2 + c^2]; relative bias at c = 0.3 in, the fitted sr.
     D5  the old split problem (I1/I3) is gone: there is no sm to trade against sr. *)
dir = DirectoryName[$InputFileName] <> "../results/";
res = <|"wolframVersion" -> $Version|>;
(* sigma_r is read from the current main-model fit (results/perception_variants.json,
   kappa_sr_region_g = src/headline.py MAIN, bottom-edge sigma_r = paper \SrBatter / \SrCatcher) instead of the
   pre-revision 3.12 / 1.95 in; exact rationals of the stored doubles *)
(* the file carries Python NaN / Infinity tokens (not JSON): map them to null before parsing *)
pv = ImportString[StringReplace[Import[dir <> "perception_variants.json", "Text"], RegularExpression["\\b-?(NaN|Infinity)\\b"] -> "null"],
    "RawJSON"]["kappa_sr_region_g"]["params"];
srB = Rationalize[pv["off_sr"], 0]; srC = Rationalize[pv["def_sr"], 0];
res["sr_batter"] = N[srB, 8]; res["sr_catcher"] = N[srC, 8];
Phi[x_] := CDF[NormalDistribution[], x];
phi[x_] := PDF[NormalDistribution[], x];

(* D1: smooth-model orthant probability at sm -> 0 vs the deterministic form (w != 0) *)
pcwBin[w_, sm_, sr_, k_] := With[{st = Sqrt[sm^2 + sr^2]}, CDF[BinormalDistribution[{0, 0}, {1, 1}, sm/st], {w/sm, (w - k)/st}]];
d1 = Table[N[pcwBin[w, 1/1000, sr, k] - UnitStep[w] Phi[(w - k)/sr], 20],
   {w, {-3/2, -1/2, -1/10, 1/10, 1/2, 3/2}}, {sr, {srC, srB}}, {k, {2, 4}}];
res["D1_limit_maxabs"] = ToString[InputForm[N[Max[Abs[Flatten[d1]]], 5]]];

(* D2: two-point Jacobian determinant, symbolic *)
pch[w_, sr_, k_] := Phi[(w - k)/sr];
J2 = {{D[pch[w1, sr, k], sr], D[pch[w1, sr, k], k]}, {D[pch[w2, sr, k], sr], D[pch[w2, sr, k], k]}};
det = FullSimplify[Det[J2] - phi[(w1 - k)/sr] phi[(w2 - k)/sr] (w1 - w2)/sr^3, sr > 0];
res["D2_det_minus_closed_form"] = ToString[InputForm[det]];
wgrid = {-1, -1/2, 0, 1/4, 1/2, 1, 3/2, 2};
Do[With[{p = sides},
   res["D2_rank_" <> ToString[p[[1]]]] = MatrixRank[N[Table[{D[pch[w, sr, k], sr], D[pch[w, sr, k], k]} /. {sr -> p[[2]], k -> p[[3]]}, {w, wgrid}], 30], Tolerance -> 10^-12];
   res["D2_sv_" <> ToString[p[[1]]]] = ToString[InputForm@N[SingularValueList[N[Table[{D[pch[w, sr, k], sr], D[pch[w, sr, k], k]} /. {sr -> p[[2]], k -> p[[3]]}, {w, wgrid}], 30]], 6]]],
  {sides, {{"batter", srB, 4}, {"catcher", srC, 2}}}];

(* D3: cross derivatives of the challenged-row log-likelihood between pi and (sr, k) *)
ll = Log[Phi[(w - k)/sr]] + a Log[1 - pi] + (1 - a) Log[pi];
res["D3_cross_pi_sr"] = ToString[InputForm[Simplify[D[ll, pi, sr]]]];
res["D3_cross_pi_k"] = ToString[InputForm[Simplify[D[ll, pi, k]]]];

(* D4: integrate the tracking noise out exactly *)
(* exact identity E Phi(a + b Z) = Phi(a / Sqrt[1 + b^2]); checked at 25 digits on a grid by NIntegrate *)
d4 = Table[NIntegrate[PDF[NormalDistribution[0, c], e] Phi[(w + e - k)/sr], {e, -Infinity, Infinity}, WorkingPrecision -> 25] -
     N[Phi[(w - k)/Sqrt[sr^2 + c^2]], 25], {w, {-1, 0, 1}}, {c, {1/10, 3/10}}, {sr, {srC, srB}}, {k, {2, 4}}];
res["D4_integral_minus_closed_form_maxabs"] = ToString[InputForm[N[Max[Abs[Flatten[d4]]], 5]]];
res["D4_rel_bias_pct_c03"] = ToString[N[Table[{s, 100 (Sqrt[s^2 + (3/10)^2] - s)/s}, {s, {srC, srB}}], 6]];

(* D5: with sm absent, the Pch block has full rank in (sr, k) -- contrast I3 (rank 2 in (sm, sr, k)) *)
res["D5_note"] = "no sm parameter: the (sm, sr) null direction of the earlier smooth-truth model does not exist";

Export[dir <> "identification-det-check.json", res, "JSON"];
Print[res];
