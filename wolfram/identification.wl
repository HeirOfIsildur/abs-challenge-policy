(* HONZA-73: why the missed-call noise sm and the perception noise sr are separately identified.
   Model (one side, one region): eta ~ N(0, sm^2), eps ~ N(0, sr^2), W = 1{w + eta > 0}, challenge iff w + eta + eps > k.
     Pch(w) = P(challenge | w)          = Phi((w - k)/st),  st = Sqrt[sm^2 + sr^2]
     Pcw(w) = P(challenge, W = 1 | w)   = Int_{-w}^Inf phi_sm(eta) Phi((w + eta - k)/sr) d eta
   Claims checked:
     I1  Pch depends on (sm, sr) only through st  -> non-challenged pitches identify st (and k), not the split.
     I2  Pcw closed form = bivariate normal orthant with correlation sm/st.
     I3  Jacobian of (Pch(w_j), Pcw(w_j))_j w.r.t. (sm, sr, k) has full rank 3 at the fitted values; the Pch block
         alone has rank 2 (null direction = moving sm, sr along a circle of constant st).
     I4  Selection limits of the overturn rate q(w) = Pcw/Pch: sr -> Inf gives Phi(w/sm) (no selection); sr -> 0 with
         k > 0 gives q = 1 for every challenged pitch.  So, at fixed st, the level of q identifies sr. *)
dir = DirectoryName[$InputFileName] <> "../results/";
res = <|"wolframVersion" -> $Version|>;
Phi[x_] := CDF[NormalDistribution[], x];

(* I1 *)
pch[w_, sm_, sr_, k_] := Phi[(w - k)/Sqrt[sm^2 + sr^2]];
i1 = FullSimplify[sr D[pch[w, sm, sr, k], sm] - sm D[pch[w, sm, sr, k], sr], sm > 0 && sr > 0];
res["I1_sr_dPdsm_minus_sm_dPdsr"] = ToString[InputForm[i1]];

(* I2: integral = bivariate normal CDF (numerical check at 30 digits on a grid) *)
pcwInt[w_?NumericQ, sm_?NumericQ, sr_?NumericQ, k_?NumericQ] :=
  NIntegrate[PDF[NormalDistribution[0, sm], e] Phi[(w + e - k)/sr], {e, -w, Infinity}, WorkingPrecision -> 30, PrecisionGoal -> 20];
pcwBin[w_, sm_, sr_, k_] := With[{st = Sqrt[sm^2 + sr^2]},
   CDF[BinormalDistribution[{0, 0}, {1, 1}, sm/st], {w/sm, (w - k)/st}]];
i2 = Table[N[pcwInt[w, sm, sr, k] - pcwBin[w, sm, sr, k], 25],
   {w, {-1/2, 0, 1/2, 3/2}}, {sm, {212/1000, 306/1000}}, {sr, {2022/1000, 3273/1000}}, {k, {1806/1000, 3741/1000}}];
res["I2_orthant_maxabs"] = ToString[InputForm[N[Max[Abs[Flatten[i2]]], 5]]];

(* I3: Jacobian rank at the fitted values (region bottom; batter and catcher sides).
   Derivatives by central differences (step h = 10^-10) of the 30-digit integral representation. *)
wgrid = {-1, -1/2, 0, 1/4, 1/2, 1, 3/2, 2};
h = 10^-10;
fvec[{sm_, sr_, k_}] := Join[Table[N[pch[w, sm, sr, k], 30], {w, wgrid}], Table[pcwInt[w, sm, sr, k], {w, wgrid}]];
jac[p0_] := Transpose[Table[(fvec[p0 + h UnitVector[3, j]] - fvec[p0 - h UnitVector[3, j]])/(2 h), {j, 3}]];
sides = <|"batter" -> {212/1000, 3273/1000, 3741/1000}, "catcher" -> {212/1000, 2022/1000, 1806/1000}|>;
Do[With[{J = jac[sides[s]]},
   res["I3_" <> s <> "_rank_full"] = MatrixRank[J, Tolerance -> 10^-12];
   res["I3_" <> s <> "_sv_full"] = ToString[InputForm@N[SingularValueList[J, Tolerance -> 0], 6]];
   res["I3_" <> s <> "_rank_challenge_only"] = MatrixRank[J[[1 ;; Length[wgrid]]], Tolerance -> 10^-12];
   res["I3_" <> s <> "_sv_challenge_only"] = ToString[InputForm@N[SingularValueList[J[[1 ;; Length[wgrid]]], Tolerance -> 0], 6]];
   (* null direction of the challenge-only block is (sm, sr) tangent to the circle st = const *)
   res["I3_" <> s <> "_null_dir_challenge_only"] = ToString[InputForm[N[Last[Transpose[SingularValueDecomposition[J[[1 ;; Length[wgrid]]]][[3]]]], 6]]];
   res["I3_" <> s <> "_tangent_st_const"] = ToString[N[Normalize[{-sides[s][[2]], sides[s][[1]], 0}], 6]]],
  {s, Keys[sides]}];

(* I4: limits of the overturn rate given a challenge *)
q[w_, sm_, sr_, k_] := pcwInt[w, sm, sr, k]/N[pch[w, sm, sr, k], 30];
res["I4_sr_large_minus_Phi_w_over_sm"] = ToString[InputForm@N[Table[q[w, 212/1000, 10^4, 3741/1000] - Phi[w/(212/1000)], {w, {-1/2, 0, 1/2}}], 6]];
res["I4_sr_small_k_pos"] = ToString[N[Table[q[w, 212/1000, 10^-3, 3741/1000], {w, {-1/2, 0, 1/2}}], 8]];
(* at fixed st (= fitted batter value), q at w = 0 as the split moves from sm small to sm large *)
st0 = Sqrt[(212/1000)^2 + (3273/1000)^2];
res["I4_q_w0_along_constant_st"] = ToString[N[Table[{smv, q[0, smv, Sqrt[st0^2 - smv^2], 3741/1000]},
     {smv, {1/10, 212/1000, 1/2, 1, 2, 3}}], 6]];

Export[dir <> "identification-check.json", res, "JSON"];
Print[res];
