"""Contrato numérico único del TP (Esquema T, issue t0-2).

T1 lo implementa primero; T2/T3 lo IMPORTAN, no lo duplican.
Todo en numpy/pandas. Funciones puras x,y como vectores 1-D.
"""
import numpy as np


def xbar(x):
    return float(np.mean(x))


def ybar(y):
    return float(np.mean(y))


def Sxx(x):
    x = np.asarray(x, float)
    return float(((x - x.mean()) ** 2).sum())


def Sxy(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    return float(((x - x.mean()) * (y - y.mean())).sum())


def beta1(beta_sxy, beta_sxx):
    return float(beta_sxy / beta_sxx)


def beta0(yb, b1, xb):
    return float(yb - b1 * xb)


def sce(x, y, b0, b1):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    return float((((y - (b0 + b1 * x))) ** 2).sum())


def stc(y):
    y = np.asarray(y, float)
    return float(((y - y.mean()) ** 2).sum())


def scr(stc_v, sce_v):
    return float(stc_v - sce_v)


def sigma2(sce_v, n):
    return float(sce_v / (n - 2))


def r2(sce_v, stc_v):
    return float(1 - sce_v / stc_v)


def t_stat(b1, s2, sxx_v):
    return float(b1 / np.sqrt(s2 / sxx_v))


def ic_beta(b, s2, sxx_v, tcrit, xb=None, x=None):
    """IC pendiente: b1 ± t*sqrt(s2/Sxx). IC ordenada: b0 ± t*sqrt(s2*(1/n + xb²/Sxx))."""
    import numpy as np
    if xb is None:  # pendiente
        se = np.sqrt(s2 / sxx_v)
    else:  # ordenada
        n = len(np.asarray(x))
        se = np.sqrt(s2 * (1 / n + xb ** 2 / sxx_v))
    return (float(b - tcrit * se), float(b + tcrit * se))


def icm_ip(x_star, b0, b1, s2, n, xb, sxx_v, tcrit):
    """Devuelve (icm_lo, icm_hi, ip_lo, ip_hi) al 95% para x_star."""
    import numpy as np
    yhat = b0 + b1 * x_star
    se_mean = np.sqrt(s2 * (1 / n + (x_star - xb) ** 2 / sxx_v))
    se_pred = np.sqrt(s2 * (1 + 1 / n + (x_star - xb) ** 2 / sxx_v))
    return (float(yhat - tcrit * se_mean), float(yhat + tcrit * se_mean),
            float(yhat - tcrit * se_pred), float(yhat + tcrit * se_pred))
