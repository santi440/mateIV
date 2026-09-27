"""Chequeos de reproducibilidad (correr con pytest o python)."""
import numpy as np


def test_stc_igual_scr_mas_sce():
    from src.metrics import stc, sce, scr
    rng = np.random.default_rng(0)
    x = rng.normal(3000, 2000, 50)
    y = 130 - 0.002 * x + rng.normal(0, 2, 50)
    from src.metrics import Sxx, Sxy, beta1, beta0, xbar, ybar
    b1 = beta1(Sxy(x, y), Sxx(x))
    b0 = beta0(ybar(y), b1, xbar(x))
    assert abs(stc(y) - (scr(stc(y), sce(x, y, b0, b1)) + sce(x, y, b0, b1))) < 1e-6


def test_beta_manual_vs_polyfit():
    from src.metrics import Sxx, Sxy, beta1, beta0, xbar, ybar
    rng = np.random.default_rng(1)
    x = rng.normal(3000, 2000, 50)
    y = 130 - 0.002 * x + rng.normal(0, 2, 50)
    b1 = beta1(Sxy(x, y), Sxx(x))
    b0 = beta0(ybar(y), b1, xbar(x))
    p = np.polyfit(x, y, 1)
    assert abs(b1 - p[0]) < 1e-8 and abs(b0 - p[1]) < 1e-8
