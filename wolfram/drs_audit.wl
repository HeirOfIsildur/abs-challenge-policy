(* independent audit of the DRS band model: (i) the three-way likelihood components at the Test fit by direct
   integration over (X, S), (ii) the posterior loss probability l = P(X < -kappa | S) used by dp.py.
   Python values from src/drs/model.py probs() and dp.draw(). *)
dir = DirectoryName[$InputFileName] <> "../results/drs/";
res = <|"wolframVersion" -> $Version|>;
With[{mu = -515/1000, s = 1/2, ka = 48/100, tau = 1/5},
 Module[{z = InverseCDF[NormalDistribution[], tau], S0, f, J0, Jk, q},
  S0 = (1 + s^2) Sqrt[s^2/(1 + s^2)] z - mu s^2;
  (* P(X in A, S >= S0) = Int_A phi(x - mu) P(eps >= S0 - x) dx *)
  f[a_, b_] := NIntegrate[PDF[NormalDistribution[mu, 1], x] (1 - CDF[NormalDistribution[0, s], S0 - x]), {x, a, b}, WorkingPrecision -> 30];
  q = f[-Infinity, Infinity]; J0 = f[0, Infinity]; Jk = f[-ka, Infinity];
  res["success_diff"] = N[J0 - 281528528819683/10^15];
  res["uc_diff"] = N[(Jk - J0) - 10484692422441039/10^17];
  res["lost_diff"] = N[(q - Jk) - 5204431611078042/10^17];
  res["sum_to_q"] = N[(J0 + (Jk - J0) + (q - Jk)) - q]]];
(* posterior loss probability: X | S ~ N(m, v); P(X < -kappa | S) at S = 0.3 vs python 0.0838466101551329 *)
With[{mu = -515/1000, s = 1/2, ka = 48/100, S = 3/10},
 Module[{post},
  post = NIntegrate[PDF[NormalDistribution[mu, 1], x] PDF[NormalDistribution[0, s], S - x], {x, -Infinity, -ka}, WorkingPrecision -> 30]/
    PDF[NormalDistribution[mu, Sqrt[1 + s^2]], S];
  res["l_diff"] = N[post - 838466101551329/10^16]]];
Export[dir <> "audit-check.json", res, "JSON"];
Print[ExportString[res, "JSON"]];
