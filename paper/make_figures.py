"""HONZA-77: manuscript figures from results/ (and figures/ for plots that need data/) -> paper/figures/.

fig_policies.pdf is drawn here from results/headline.json; the threshold, sqrt-n and DRS figures are produced by the
pipeline (src/refresh_thresholds.py, src/drs/plots.py) and copied.
usage (from the repo root): python paper/make_figures.py"""
import json, os, shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "paper/figures"
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
LABEL = {"bench": "Pooled-model benchmark", "opt_score": "Benchmark, lapses removed", "runs_dp": "Runs-objective rule", "opt": "Pooled DP (no score)",
         "stake_0.008": "One-line rule, δ = 0.008", "stake_0.01": "One-line rule, δ = 0.01",
         "card_pub": "48-cell card (published)", "card_refit": "48-cell card (refit)", "observed": "Observed 2026",
         "obs_rule": "Fitted behavioural rule", "savant": "Fixed 0.2-run rule",
         "savant_tuned": "Savant form, tuned run cost", "clairvoyant": "Clairvoyant comparator"}


def policies():
    h = json.load(open("results/headline.json"))
    P = sorted(h["policies"], key=lambda p: p["wins"])
    plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": INK})
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    for i, p in enumerate(P):
        col = ORANGE if p["policy"] == "observed" else BLUE
        ax.plot(p["wins_ci"], [i, i], color=col, lw=2, solid_capstyle="round")
        ax.plot(p["wins"], i, "o", ms=6, color=col, mec="white", mew=1.5)
        ax.text(p["wins_ci"][1] + 0.05, i, f"{p['wins']:.2f}  ({100 * p['share']:.0f} %)", va="center",
                fontsize=8, color=MUTED)
    ax.set_yticks(range(len(P)), [LABEL[p["policy"]] for p in P])
    ax.set_xlabel("Wins gained from challenges per team-season")
    ax.set_xlim(2.0, 4.9)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_policies.pdf"); plt.close(fig)


def sr_breakeven():
    """HONZA-91: wins lost against a misestimated sigma_r (results/sr_breakeven.json), with the profile log-likelihood
    of each side's sigma_r on the real data (results/identification.json) on the same multiplier axis."""
    br = sorted((float(k), v["lost"]) for k, v in json.load(open("results/sr_breakeven.json")).items())
    idn = json.load(open("results/identification.json"))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.4, 2.8))
    ax.plot([x for x, _ in br], [l for _, l in br], "o-", color=BLUE, ms=4)
    ax.axhline(0.25, color=MUTED, lw=0.8, ls="--"); ax.axhline(0, color=GRID, lw=0.8)
    ax.axvline(1, color=GRID, lw=0.8)
    ax.text(0.79, 0.29, "0.25", fontsize=8, color=MUTED)
    ax.set_xlabel(r"$\sigma_r$ / estimate"); ax.set_ylabel("Wins lost per team-season")
    for side, lab, col in (("off", "batter", BLUE), ("def", "catcher", ORANGE)):
        s0 = idn["truth"][f"{side}_sr"]
        P = sorted((x / s0, ll) for x, ll in idn[f"profile_{side}"])
        bx.plot([x for x, _ in P], [ll for _, ll in P], "o-", color=col, ms=4, label=lab)
    bx.axvline(1, color=GRID, lw=0.8)
    bx.set_xlabel(r"$\sigma_r$ / estimate"); bx.set_ylabel("Profile log-likelihood loss")
    bx.legend(frameon=False, fontsize=8)
    for a in (ax, bx):
        for s_ in ("top", "right"):
            a.spines[s_].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_sr_breakeven.pdf"); plt.close(fig)


def isotonic():
    """HONZA-106: nonparametric (isotonic) challenge curve against the fitted probit, by side and zone region
    (results/isotonic_check.json, src/isotonic_check.py)."""
    d = json.load(open("results/isotonic_check.json")); x = d["grid"]
    fig, axs = plt.subplots(2, 4, figsize=(6.6, 3.6), sharex=True)
    for i, (side, lab) in enumerate((("off", "Batters"), ("def", "Catchers"))):
        for j, reg in enumerate(("bottom", "top", "inside", "outside")):
            p = d["panels"][f"{side}_{reg}"]; ax = axs[i, j]
            ax.fill_between(x, p["lo"], p["hi"], color=GRID, lw=0)
            ax.step(x, p["empirical"], where="mid", color=INK, lw=1)
            ax.plot(x, p["model"], color=BLUE, lw=1.2)
            ax.set_title(f"{lab}, {reg}", fontsize=8)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
            if i == 1: ax.set_xlabel("$w$ (in)")
            if j == 0: ax.set_ylabel("P(challenge)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_isotonic.pdf"); plt.close(fig)


def copies():
    for src, dst in (("figures/fig1_threshold_score.pdf", "fig_threshold.pdf"), ("figures/fig_delta_sqrt.png", "fig_delta.png"),
                     ("figures/drs_fig1.png", "fig_drs.png")):
        if os.path.exists(src):
            shutil.copy(src, f"{OUT}/{dst}")
        else:
            print(f"missing {src}: run the pipeline step that writes it (see README)")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    policies()
    sr_breakeven()
    isotonic()
    copies()
