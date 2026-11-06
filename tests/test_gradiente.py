"""DoD issue #11: descenso por gradiente converge y coincide con la exacta."""
import numpy as np

from src.gradiente import (betas_a_original, descenso_gradiente,
                           estandarizar_features, paso_maximo_estable)
from src.multiple import betas


def _datos(n=500, seed=2026):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(200.0, 20000.0, n)
    x2 = rng.uniform(0.0, 22.0, n)
    x3 = rng.uniform(0.02, 0.30, n)
    x4 = rng.uniform(4e-4, 6e-2, n)
    y = (137.0 - 1.2e-3 * x1 - 0.35 * x2 - 33.9 * x3 - 164.0 * x4
         + rng.normal(0.0, 3.0, n))
    X = np.column_stack([np.ones(n), x1, x2, x3, x4])
    return X, y


def test_estandarizacion_de_features():
    X, _ = _datos()
    Xs, medias, desvios = estandarizar_features(X)
    assert np.allclose(Xs[:, 1:].mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(Xs[:, 1:].std(axis=0), 1.0, atol=1e-12)
    assert np.all(desvios > 0)
    assert np.allclose(medias, X[:, 1:].mean(axis=0))


def test_descenso_converge_y_coincide_con_betas_exactas():
    X, y = _datos()
    Xs, medias, desvios = estandarizar_features(X)
    _, alpha_max = paso_maximo_estable(Xs)

    res = descenso_gradiente(Xs, y, 0.5 * alpha_max, max_iter=5000,
                             tol=1e-9)
    assert res["estado"] == "convergio"

    b_original = betas_a_original(res["b"], medias, desvios)
    assert np.allclose(b_original, betas(X, y), rtol=1e-4, atol=1e-4)


def test_alpha_inestable_diverge():
    X, y = _datos()
    Xs, _, _ = estandarizar_features(X)
    _, alpha_max = paso_maximo_estable(Xs)

    res = descenso_gradiente(Xs, y, 1.1 * alpha_max, max_iter=1000)
    assert res["estado"] == "diverge"
