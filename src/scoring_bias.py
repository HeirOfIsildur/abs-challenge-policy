"""HONZA-134 (round-1 referees on Lemma 3: is the in-fold scoring contrast unbiased?): on S random 5-fold splits,
the signed contrast of the engine fitted to a fold alone (gv, the scorer of the full policy cross-fit) against the
full-season engine's signed contrast on the same pitches -- over all called pitches, the challenged pitches, and
pitches by decile of the full-season stake. A ratio of means near one says the fold engine neither shrinks nor
inflates the contrasts it scores; the full-season engine is the reference, not the truth.
usage: python src/scoring_bias.py [S]  -> results/scoring_bias.json"""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")


def main(S=3):
    import crossfit as CF
    m = pd.read_parquet("data/called_2026_w.parquet"); sc = CF.statcast()
    rows = []
    for s in range(S):
        mb = CF.crossfit_frame(m, sc, np.random.default_rng(5000 + s))
        g0, gv = m.g_signed.values, mb.gv.values
        ch = m.ch.values
        dec = pd.qcut(np.abs(g0), 10, labels=False, duplicates="drop")
        rows.append(dict(split=s, ratio_all=float(gv.sum() / g0.sum()), ratio_challenged=float(gv[ch].sum() / g0[ch].sum()),
                         ratio_overturned=float(gv[ch & m.ch_over.values].sum() / g0[ch & m.ch_over.values].sum()),
                         corr=float(np.corrcoef(gv, g0)[0, 1]),
                         ratio_by_decile=[float(gv[dec == d].sum() / g0[dec == d].sum()) for d in range(int(dec.max()) + 1)]))
        print(rows[-1], flush=True)
    keys = ["ratio_all", "ratio_challenged", "ratio_overturned", "corr"]
    out = dict(S=S, mean={k: float(np.mean([r[k] for r in rows])) for k in keys},
               ratio_by_decile=np.mean([r["ratio_by_decile"] for r in rows], axis=0).tolist(), rows=rows)
    json.dump(out, open("results/scoring_bias.json", "w"), indent=1)
    print(json.dumps(out["mean"], indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
