(* kernel checks (signed opponent term): flat-optimum per-state regret, causal (martingale) decomposition, rule equivalences *)
res = <|"wolframVersion" -> $Version|>;
ok[name_, expr_] := (res[name] = ToString[InputForm[expr]]; If[expr =!= True, Print["FAILED ", name, ": ", InputForm[expr]]]);
(* 1. per-state regret of a constant-delta rule: integral of |p(g+D)-D| over the disagreement interval *)
a = {g > 0, D > 0, de > 0};
r1 = Integrate[D - p (g + D), {p, de/(g + de), D/(g + D)}, Assumptions -> Join[a, {D > de}]];
r2 = Integrate[p (g + D) - D, {p, D/(g + D), de/(g + de)}, Assumptions -> Join[a, {D < de}]];
target = g^2 (D - de)^2/(2 (g + D) (g + de)^2);
ok["T1_regret_D_gt_delta", FullSimplify[r1 == target, Join[a, {D > de}]]];
ok["T1_regret_D_lt_delta", FullSimplify[r2 == target, Join[a, {D < de}]]];
ok["T1_envelope_r'(D)=0", FullSimplify[(D[target, de] /. de -> D) == 0, a]];
(* 2. causal decomposition: two-stage game. Branch pitch moves value from v0 to v0+g (flip). Later the team meets an
   opportunity with prob q in state A (after flip) or qB (no flip); stake gA/gB, success prob p; flipping adds the stake.
   Terminal win probs are the engine values; E[win] - E[win | no flip, never challenge] = g + (later gains on A path). *)
vA = v0 + g; vB = v0;
winA = (1 - q) vA + q ((1 - ch) vA + ch (p (vA + gA) + (1 - p) vA));   (* challenge w.p. ch; success adds gA *)
winB = vB;
ok["T2_causal_decomposition", FullSimplify[winA - winB == g + q ch p gA]];
winA0 = (1 - qB) vB + qB ((1 - ch) vB + ch (p (vB + gB) + (1 - p) vB));
ok["T2_rb_difference", FullSimplify[(winA - winA0) - g == q ch p gA - qB ch p gB]];
(* with an opponent that also challenges later (prob qo / qoB of an opportunity,
   challenge prob cho, success po, stake goA / goB, which lowers the team's win probability), the win difference is the
   branch g plus the team's own later gains minus the opponent's: the RB estimator must difference signed gains. *)
winAo = winA - qo cho po goA; winA0o = winA0 - qoB cho po goB;
ok["T2_rb_difference_signed", FullSimplify[(winAo - winA0o) - g == (q ch p gA - qo cho po goA) - (qB ch p gB - qoB cho po goB)]];
ok["T2_rb_own_only_misses_opponent", FullSimplify[((winAo - winA0o) - g) - (q ch p gA - qB ch p gB) == -(qo cho po goA - qoB cho po goB)]];
(* 3. rule equivalences *)
ok["T3_runs_rule", Resolve[ForAll[{pp, rv, cc}, 0 < pp < 1 && rv > 0 && cc > 0, Equivalent[pp rv > (1 - pp) cc, pp > cc/(rv + cc)]], Reals]];
ok["T3_stake_rule", Resolve[ForAll[{pp, gg, dd}, 0 < pp < 1 && gg > 0 && dd > 0, Equivalent[pp (gg + dd) >= dd, pp >= dd/(gg + dd)]], Reals]];
Export["results/headline-check.json", res, "JSON"];
Print[res];
