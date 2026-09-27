"""DoD issue #2: el contrato importa en verde y es autoconsistente."""
import numpy as np

from src import metrics as m
from src.config import SEED

CONTRACT = ["xbar", "ybar", "Sxx", "Sxy", "beta1", "beta0", "sce",
            "stc", "scr", "sigma2", "r2", "ic_beta", "icm_ip"]


def _data():
    rng = np.random.default_rng(SEED)
    x = rng.normal(3000, 2000, 200)
    y = 130 - 0.002 * x + rng.normal(0, 2, 200)
    return x, y


def test_contrato_expone_nombres():
    for name in CONTRACT:
        assert callable(getattr(m, name)), name


def test_beta_manual_vs_polyfit():
    x, y = _data()
    b1 = m.beta1(m.Sxy(x, y), m.Sxx(x))
    b0 = m.beta0(m.ybar(y), b1, m.xbar(x))
    p1, p0 = np.polyfit(x, y, 1)
    assert abs(b1 - p1) < 1e-8 and abs(b0 - p0) < 1e-8


def test_descomposicion_y_r():
    x, y = _data()
    b1 = m.beta1(m.Sxy(x, y), m.Sxx(x))
    b0 = m.beta0(m.ybar(y), b1, m.xbar(x))
    n = len(x)
    stc, sce = m.stc(y), m.sce(x, y, b0, b1)
    assert abs(stc - (m.scr(stc, sce) + sce)) < 1e-6
    r2 = m.r2(sce, stc)
    syy = ((y - y.mean()) ** 2).sum()
    assert abs(m.r_corr(m.Sxy(x, y), m.Sxx(x), syy) ** 2 - r2) < 1e-9
    assert 0 <= r2 <= 1


def test_intervalos_coherentes():
    x, y = _data()
    n = len(x)
    xb, sxx = m.xbar(x), m.Sxx(x)
    b1 = m.beta1(m.Sxy(x, y), sxx)
    b0 = m.beta0(m.ybar(y), b1, xb)
    s2 = m.sigma2(m.sce(x, y, b0, b1), n)
    t = m.t_crit_95(n - 2)
    assert abs(t - 1.96) < 0.01
    lo, hi = m.ic_beta(b1, m.se_beta1(s2, sxx), t)
    assert lo < b1 < hi
    icm_lo, icm_hi, ip_lo, ip_hi = m.icm_ip(2000.0, b0, b1, s2, n, xb, sxx, t)
    assert (icm_hi - icm_lo) < (ip_hi - ip_lo)  # ICM mas angosto que IP
