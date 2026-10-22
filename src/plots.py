"""Figure 1: optimal threshold p* vs inning x challenges left (at the median challenged-pitch stake), with the
threshold teams reveal (fitted kappa rule mapped to perceived confidence). Figure 2: wins per team-season by
policy (bootstrap 95% CI). Also: role table."""
import json, pickle, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "src")
from sim import load, draw, monotone_calibration
from joint import unpack
d = pickle.load(open("data/sim_main.pkl", "rb")); r = json.load(open("data/res_main.json"))
g = r["g_ref"]
# revealed thresholds: perceived confidence at the signal cut-off kappa(state)
m, P = load({}); m = m[m.w.abs() < 8].reset_index(drop=True)
rng = np.random.default_rng(3); W, s, pt = draw(m, rng, {})
cal = {}
for side in ("off", "def"):
    k = (m.side == side).values
    b = np.clip(np.floor((s[k] + 15) / 0.1).astype(int), 0, 300)
    num = np.bincount(b, weights=W[k], minlength=301); den = np.bincount(b, minlength=301)
    cal[side] = monotone_calibration(num, den)
def revealed(side, inning, left1, strikes=1, balls=1):
    beta = P["off" if side == "off" else "deff"]["beta"]
    x = np.array([1, np.log(g), left1, min(inning, 10) / 9, strikes == 2, balls == 3], float)
    kap = x @ beta; return cal[side][int(np.clip(np.floor((kap + 15) / 0.1), 0, 300))]
fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
inn = np.arange(1, 10)
for j, (side, lab) in enumerate((("def", "fielding (catcher)"), ("off", "batting (batter)"))):
    for c, col in ((2, "#2a6fdb"), (1, "#d1495b")):
        ps = []
        for i in inn:
            vals = []
            for Dh, Da in d["Ds"]:
                # home fields in the top (slot 2i-2), bats in the bottom (2i-1); away the reverse
                sl_h = 2 * (i - 1) + (1 if side == "off" else 0); sl_a = 2 * (i - 1) + (0 if side == "off" else 1)
                for D, sl in ((Dh, sl_h), (Da, sl_a)):
                    vals.append(D[sl, 0, c] / (g + D[sl, 0, c]))
            ps.append(np.mean(vals))
        ax[j].plot(inn, ps, "-o", color=col, label=f"optimal, {c} left")
        ax[j].plot(inn, [revealed(side, i, float(c == 1)) for i in inn], "--", color=col, alpha=.7, label=f"teams (revealed), {c} left")
    ax[j].axhline(0.2 / (0.2 + 0.1), color="grey", lw=1, ls=":", label="Savant 0.2 rule (RV = 0.1 run)")
    ax[j].set_title(f"{lab}"); ax[j].set_xlabel("inning (start)"); ax[j].set_xticks(inn)
ax[0].set_ylabel(f"confidence needed to challenge\n(stake g = {g:.4f} WP)"); ax[1].legend(fontsize=7, loc="upper right")
fig.suptitle("ABS 2026: optimal challenge threshold vs inning and challenges left"); fig.tight_layout()
fig.savefig("figures/fig1_threshold.png", dpi=150)
# Figure 2
rows = [("Savant 0.2 rule", "savant"), ("observed 2026", "observed"), ("teams' fitted rule (sim)", "obs_rule_sim"),
        ("best stake-only WP rule", "best_stake"), ("optimal (DP)", "opt")]
fig, ax = plt.subplots(figsize=(6.5, 3.2))
for y, (lab, k) in enumerate(rows):
    e, lo, hi = r[k]; ax.errorbar(e, y, xerr=[[e - lo], [hi - e]], fmt="o", color="#2a6fdb" if k != "observed" else "#d1495b", capsize=3)
    ax.text(hi + 0.05, y, f"{e:.2f}", va="center", fontsize=8)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([a for a, _ in rows]); ax.set_xlabel("wins from challenges per team-season (95% CI, game bootstrap)")
ax.set_xlim(2.8, 4.9); fig.tight_layout(); fig.savefig("figures/fig2_wins.png", dpi=150)
# role table (observed realized)
mm, _ = load({}); c = mm[mm.ch]
tab = c.groupby("role").agg(n=("ch", "size"), overturn=("ch_over", "mean"), wins_gained_per_team=("g", lambda x: (x * c.loc[x.index, "ch_over"]).sum() / 30),
                             mean_g=("g", "mean"), mean_p_model=("p_model", "mean")).round(4)
print(tab); tab.to_csv("figures/roles_observed.csv")
