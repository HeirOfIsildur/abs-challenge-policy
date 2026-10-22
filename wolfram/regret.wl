(* The closed form r(delta) = g^2 (D-delta)^2 / (2 (g+D) (g+delta)^2) is the
   integral of the Q-gap |p (g+D) - D| over the error interval I with UNIT density, i.e. what bound (ii) integrates;
   it is not the per-state regret of a general conditional law of p.  run: wolframscript -file wolfram/regret.wl *)
dir = "results/";
q[p_, g_, D_] := Abs[p (g + D) - D];
r[g_, D_, de_] := g^2 (D - de)^2/(2 (g + D) (g + de)^2);
unitBelow = Simplify[Integrate[D - p (g + D), {p, de/(g + de), D/(g + D)}] == r[g, D, de], g > 0 && D > de > 0];
unitAbove = Simplify[Integrate[p (g + D) - D, {p, D/(g + D), de/(g + de)}] == r[g, D, de], g > 0 && de > D > 0];
(* a reviewer's counter-example: g = D = 1, delta = 2, density f(p) = 2 p on [0, 1] *)
actual = Integrate[q[p, 1, 1] 2 p, {p, 1/2, 2/3}];
closed = r[1, 1, 2];
(* bound (ii) with M = sup f on I = 4/3 still holds *)
bound = (4/3) r[1, 1, 2];
res = <|"version" -> $Version,
  "unit_density_integral_equals_closed_form_D_gt_delta" -> unitBelow,
  "unit_density_integral_equals_closed_form_D_lt_delta" -> unitAbove,
  "example_actual_regret" -> ToString[actual, InputForm], "example_closed_form" -> ToString[closed, InputForm],
  "example_differs" -> (actual =!= closed), "example_bound_M_times_closed" -> ToString[bound, InputForm],
  "example_bound_holds" -> (actual <= bound)|>;
Export[dir <> "regret-check.json", res, "JSON"];
Print[ExportString[res, "JSON"]];
