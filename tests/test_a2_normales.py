"""DoD issue #6: sistema normal, betas cerradas y polyfit coinciden."""
import numpy as np

from src import normales as nr


def _datos(n=200, seed=2026):
    rng = np.random.default_rng(seed)
    x = rng.uniform(200.0, 20000.0, n)
    y = 127.3 - 8.55e-4 * x + rng.normal(0.0, 3.0, n)
    return x, y


def test_sumas_normales():
    x, y = _datos()
    s = nr.sumas_normales(x, y)
    assert s["n"] == len(x)
    assert np.isclose(s["sum_x"], x.sum())
    assert np.isclose(s["sum_y"], y.sum())
    assert np.isclose(s["sum_x2"], (x ** 2).sum())
    assert np.isclose(s["sum_xy"], (x * y).sum())


def test_sistema_normal_coincide_con_formulas_cerradas_y_polyfit():
    x, y = _datos()
    c = nr.contraste(x, y)
    assert np.allclose(c["normal"], c["cerrada"], rtol=0.0, atol=1e-10)
    assert np.allclose(c["normal"], c["polyfit"], rtol=0.0, atol=1e-10)
    assert np.allclose(c["gradiente_en_normal"], 0.0, rtol=0.0, atol=1e-7)


def test_gradiente_fuera_del_optimo_es_no_nulo():
    x, y = _datos()
    b0, b1 = nr.resolver_normal(x, y)
    g = nr.gradiente(x, y, b0 + 1.0, b1)
    assert g[0] > 0.0


def test_vectores_de_distinto_tamano_falla():
    try:
        nr.sumas_normales([1.0, 2.0], [1.0])
    except ValueError:
        return
    raise AssertionError("se esperaba ValueError")
