(* endpoint convention of I_t in Theorem 2(i) for laws with atoms.
   Rule rho_delta challenges iff p >= delta/(g+delta); optimum iff p >= D/(g+D). Q-gap |p(g+D) - D|. *)
regret[g_, D_, de_, p_] := If[(p >= de/(g + de)) =!= (p >= D/(g + D)), Abs[p (g + D) - D], 0];
r1 = regret[1, 1, 1/2, 1/3];   (* delta < D, atom at delta/(g+delta) = 1/3: rule challenges, optimum declines *)
r2 = regret[1, 1, 2, 2/3];     (* delta > D, atom at delta/(g+delta) = 2/3: both challenge *)
Print[$Version]; Print[{r1, r2}];
