"""Figures for the DRS replication: optimal lbw review threshold by wickets down and reviews left (left), and the
option value of the last review against expected out decisions left with sqrt / linear fits (right).
usage: python src/drs/plots.py [out.png]  ->  figures/drs_fig1.png (default)"""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = {"Test": "#1f5f8b", "ODI": "#c1502e", "T20": "#5a8f29"}
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for fmt, col in COL.items():
    m = json.load(open(f"results/drs/dp_{fmt}.json"))["main"]
    T = np.array(m["thresholds"]["lbw"]) * 100
    ax[0].plot(range(10), T[:, 0], "-o", color=col, ms=3, label=f"{fmt}, 1 left")
    ax[0].plot(range(10), T[:, 1], "--", color=col, label=f"{fmt}, 2 left")
    sl = m["sqrt_law"]; N, y = np.array(sl["N"]), np.array(sl["delta1"]) * 100
    ax[1].plot(N, y, "o", color=col, label=fmt)
    xs = np.linspace(N.min(), N.max(), 50)
    for f, ls in ((np.sqrt, ":"), (lambda x: x, "-")):
        b = np.polyfit(f(N), y, 1); ax[1].plot(xs, np.polyval(b, f(xs)), ls, color=col, lw=1)
ax[0].set(xlabel="wickets down", ylabel="optimal lbw review threshold p* (%)", title="Optimal threshold at the median stake")
ax[1].set(xlabel="expected out decisions left in the innings", ylabel="value of the last review (% of a match)",
          title="Option value: linear (solid) vs sqrt (dotted) fit")
ax[0].legend(fontsize=7, ncol=2, frameon=False); ax[1].legend(fontsize=8, frameon=False)
for a in ax: a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(sys.argv[1] if len(sys.argv) > 1 else "figures/drs_fig1.png", dpi=150)
