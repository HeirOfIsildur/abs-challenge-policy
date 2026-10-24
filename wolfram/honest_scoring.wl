(* exact checks behind the full policy cross-fit (main Lemma "Honest scoring", supplement S5).
   (1) Selection optimism without the split: with an estimated stake gh = g + e, e ~ N(0, s^2), a policy that
       challenges iff gh > c and is scored at gh has E[1{gh > c} gh] - g P(gh > c) = s phi((c - g)/s) > 0.
   (2) With the split: the decision uses an independent estimate gd = g + e1 and the score gh = g + e2 (e1, e2
       independent): E[1{gd > c} gh] = g P(gd > c), no optimism.
   (3) Distinct games in a bootstrap resample of n games: 1 - (1 - 1/n)^n, at n = 2425 and its limit 1 - 1/e. *)
$Assumptions = s > 0 && Element[{g, c}, Reals];
dE = NormalDistribution[0, s];
opt = FullSimplify[Expectation[Boole[g + e > c] (g + e), e \[Distributed] dE] - g Probability[g + e > c, e \[Distributed] dE]];
optOK = FullSimplify[opt == s PDF[NormalDistribution[], (c - g)/s]];
split = FullSimplify[Expectation[Boole[g + e1 > c] (g + e2), {e1 \[Distributed] dE, e2 \[Distributed] dE}] -
    g Probability[g + e1 > c, e1 \[Distributed] dE]];
distinct = 1 - (1 - 1/2425)^2425;
lim = Limit[1 - (1 - 1/n)^n, n -> Infinity];
res = <|"wolframVersion" -> $Version,
  "optimism_without_split" -> ToString[InputForm[opt]], "optimism_equals_s_phi" -> ToString[optOK],
  "optimism_with_split" -> ToString[InputForm[split]],
  "distinct_share_n2425" -> N[distinct, 10], "distinct_share_limit" -> ToString[InputForm[lim]]|>;
Export["results/honest-scoring-check.json", res, "JSON"];
Print[res["wolframVersion"]]; Print[res];
If[optOK =!= True || split =!= 0 || lim =!= 1 - 1/E, Exit[1]];
