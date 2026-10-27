"""DoD issue #5: ajuste cuadrático, derivada y punto crítico reproducibles."""
import numpy as np

from src.cuadratico import (ajustar_cuadratico, comparar_ajustes, derivada,
                            evaluar_cuadratico, punto_critico)


def _datos(seed=2026, n=300):
    rng = np.random.default_rng(seed)
    x = rng.uniform(200.0, 20000.0, n)
    y = 130.0 - 1.0e-3 * x + 5.0e-8 * x ** 2 + rng.normal(0.0, 0.1, n)
    return x, y


def test_ajuste_recupera_coeficientes():
    x = np.linspace(200.0, 20000.0, 200)
    y = 130.0 - 1.0e-3 * x + 5.0e-8 * x ** 2
    b0, b1, b2 = ajustar_cuadratico(x, y)
    assert np.isclose(b0, 130.0, atol=1e-8)
    assert np.isclose(b1, -1.0e-3, atol=1e-12)
    assert np.isclose(b2, 5.0e-8, atol=1e-16)


def test_derivada_y_evaluacion():
    x = np.array([1000.0, 10000.0])
    b0, b1, b2 = 130.0, -1.0e-3, 5.0e-8
    yhat = evaluar_cuadratico(b0, b1, b2, x)
    assert np.allclose(yhat, b0 + b1 * x + b2 * x ** 2)
    assert np.allclose(derivada(b1, b2, x), b1 + 2.0 * b2 * x)


def test_punto_critico_dentro_y_minimo():
    crit = punto_critico(130.0, -1.0e-3, 5.0e-8)
    assert np.isclose(crit["x"], 10000.0)
    assert np.isclose(crit["y"], 125.0)
    assert crit["segunda"] > 0
    assert crit["tipo"] == "minimo"
    assert crit["dentro"] is True


def test_punto_critico_fuera_del_rango():
    crit = punto_critico(130.0, -1.0e-3, 1.0e-8)
    assert crit["x"] > 20000.0
    assert crit["dentro"] is False


def test_comparacion_lineal_vs_cuadratico():
    x, y = _datos()
    res = comparar_ajustes(x, y)
    assert res["r2_cuadratico"] > res["r2_lineal"]
    assert res["sce_cuadratico"] < res["sce_lineal"]
    assert res["r2_cuadratico"] > 0.99
