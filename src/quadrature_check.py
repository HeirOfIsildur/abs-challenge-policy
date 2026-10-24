"""accuracy of the primary model's 11 x 11 Gauss-Hermite integration. The negative
log-likelihood at the fitted parameters (data/th_re_lapse_gh11.npy) with k x k nodes for k in 7, 11, 21, 31; no refit.
usage: python src/quadrature_check.py  -> results/quadrature_primary.json"""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")


def main():
    import lik, jax.numpy as jnp
    name = "re_lapse_gh11"; spec = lik.spec_of(name); th = np.load(f"data/th_{name}.npy")
    data = lik.prepare(pd.read_parquet("data/called_2026_w.parquet"), players=True)
    out = {}
    for k in (7, 11, 21, 31):
        out[str(k)] = float(lik.make_nll(data, dict(spec, gh=k))(jnp.asarray(th)))
        print(k, out[str(k)], flush=True)
    res = dict(model=name, nll_by_nodes=out, diff_11_vs_31=out["11"] - out["31"])
    json.dump(res, open("results/quadrature_primary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
