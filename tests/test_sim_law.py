"""the simulated perception noise follows the fitted law (unit variance), as in lik.cdf_unit."""
import sys
from pathlib import Path
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import sim


def test_unit_noise_laws():
    rng = np.random.default_rng(0)
    for law, ref in (("logistic", stats.logistic(scale=np.sqrt(3) / np.pi)),
                     ("t5", stats.t(5, scale=np.sqrt(3 / 5))), ("t10", stats.t(10, scale=np.sqrt(8 / 10)))):
        x = sim.unit_noise(rng, 200_000, law)
        assert abs(x.std() - 1) < 0.02, law
        assert stats.kstest(x, ref.cdf).pvalue > 1e-3, law


def test_normal_law_keeps_the_generator_stream():
    a, b = np.random.default_rng(3), np.random.default_rng(3)
    assert np.array_equal(sim.unit_noise(a, 1000), b.standard_normal(1000))


def test_draw_reads_the_law_from_the_spec():
    assert sim.law_of({"spec": {"law": "t5"}}) == "t5"
    assert sim.law_of({"spec": {}}) == "normal" and sim.law_of({}) == "normal"
