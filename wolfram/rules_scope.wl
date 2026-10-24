(* HONZA-133 (round-5 referee Sol #10): Theorem S(rules)(c)'s last sentence needs s = ess sup(pg) < Infinity.
   Counterexample family: p = 1/2, g ~ Pareto(scale 1, shape 3/2). Mean stake finite, s infinite; already one
   consumable challenge (take the largest of t i.i.d. stakes, a lower bound on its optimal value via the prophet bound
   E max/2) grows like t^(2/3), faster than the sqrt(t) of one retained challenge. *)
Print[$Version];
dist = ParetoDistribution[1, 3/2];
Print["E[g] = ", Mean[dist]];
emax[t_] = Integrate[x D[CDF[dist, x]^t, x], {x, 1, Infinity}, Assumptions -> t > 1];
Print["E[max of t stakes] = ", FullSimplify[emax[t], t > 1]];
Print["lim E[max]/t^(2/3) = ", Limit[FullSimplify[emax[t], t > 1]/t^(2/3), t -> Infinity]];
(* any stopping rule gets at least half of E[max] (prophet inequality), and pg = g/2 here, so V_cons_t(1) >= E[max]/4 ->
   Infinity faster than sqrt(t): the finite-s condition cannot be dropped from "eventually beats any fixed allotment" *)
Print["lim (E[max]/4)/Sqrt[t] = ", Limit[FullSimplify[emax[t], t > 1]/(4 Sqrt[t]), t -> Infinity]];

(* HONZA-133 (Astra #7): under Student-t noise the posterior of a positive margin need not be monotone in the signal.
   Prior mass 1/2 at w = -1 and w = +1, noise t with 3 degrees of freedom. *)
post[s_] = PDF[StudentTDistribution[3], s - 1]/(PDF[StudentTDistribution[3], s - 1] + PDF[StudentTDistribution[3], s + 1]);
Print["posterior at s = 0, 2, 10: ", Simplify[{post[0], post[2], post[10]}], ";  limit s -> Infinity: ", Limit[post[s], s -> Infinity]];
