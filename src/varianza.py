"""B.1 y B.4: descomposición de la variabilidad e intervalos en x*.

Modelo: recta de mínimos cuadrados de sound_level (dB) sobre frequency (Hz).

B.1  STC = SCR + SCE
     STC = Σ(yᵢ - ȳ)²     variabilidad total
     SCR = Σ(ŷᵢ - ȳ)²     explicada por la regresión
     SCE = Σ(yᵢ - ŷᵢ)²    no explicada (residuos)
     σ̂² = SCE / (n - 2)
     R²  = 1 - SCE / STC

B.4  ICM: ŷ* ± t(α/2, n-2) · √( σ̂² (1/n + (x* - x̄)²/Sxx) )
     IP : ŷ* ± t(α/2, n-2) · √( σ̂² (1 + 1/n + (x* - x̄)²/Sxx) )

Las tres sumas se calculan cada una con su propia fórmula, para que la
igualdad STC = SCR + SCE sea una verificación real. Por eso NO se usa
metrics.scr, que obtiene SCR como STC − SCE.
"""
import numpy as np

from src import metrics as m

X_STAR = 2000.0  # Hz, valor fijo de la consigna (B.4)


def _vectores(x, y):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError("x e y con distinto n")
    if x.size <= 2:
        raise ValueError("se necesitan al menos 3 observaciones")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("x o y con NaN/inf")
    return x, y


def recta(x, y):
    """(b0, b1) de mínimos cuadrados usando el contrato src.metrics."""
    x, y = _vectores(x, y)
    b1 = m.beta1(m.Sxy(x, y), m.Sxx(x))
    b0 = m.beta0(m.ybar(y), b1, m.xbar(x))
    return b0, b1


def descomposicion(x, y):
    """B.1: STC, SCR y SCE calculadas por separado y su verificación."""
    x, y = _vectores(x, y)
    b0, b1 = recta(x, y)
    yhat = m.yhat(b0, b1, x)
    ybar = m.ybar(y)

    stc = float(((y - ybar) ** 2).sum())     # Σ(yᵢ − ȳ)²
    scr = float(((yhat - ybar) ** 2).sum())  # Σ(ŷᵢ − ȳ)²
    sce = float(((y - yhat) ** 2).sum())     # Σ(yᵢ − ŷᵢ)²

    n = int(x.size)
    return {
        "n": n,
        "b0": b0,
        "b1": b1,
        "STC": stc,
        "SCR": scr,
        "SCE": sce,
        "SCR+SCE": scr + sce,
        "diferencia": stc - (scr + sce),
        "gl": {"SCR": 1, "SCE": n - 2, "STC": n - 1},
        "suma_residuos": float((y - yhat).sum()),
    }


def varianza_y_r2(x, y):
    """B.1: σ̂², σ̂ y R² (por tres caminos que deben coincidir)."""
    x, y = _vectores(x, y)
    d = descomposicion(x, y)
    s2 = m.sigma2(d["SCE"], d["n"])
    r = m.r_corr(m.Sxy(x, y), m.Sxx(x), d["STC"])
    return {
        "sigma2": s2,
        "sigma": float(np.sqrt(s2)),
        "R2": m.r2(d["SCE"], d["STC"]),   # 1 − SCE/STC
        "R2_SCR_STC": d["SCR"] / d["STC"],  # SCR/STC
        "r": r,
        "r2_corr": r ** 2,                 # r² (regresión simple)
    }


def intervalos(x, y, x_star=X_STAR, tcrit=None):
    """B.4: ICM e IP al 95 % en x* con las fórmulas de la consigna."""
    x, y = _vectores(x, y)
    n = int(x.size)
    b0, b1 = recta(x, y)
    sce = m.sce(x, y, b0, b1)
    s2 = m.sigma2(sce, n)
    xb = m.xbar(x)
    sxx = m.Sxx(x)
    t = m.t_crit_95(n - 2) if tcrit is None else float(tcrit)

    icm_lo, icm_hi, ip_lo, ip_hi = m.icm_ip(x_star, b0, b1, s2, n, xb, sxx, t)
    y_star = float(b0 + b1 * x_star)
    return {
        "x_star": float(x_star),
        "y_star": y_star,
        "t": t,
        "ICM": (icm_lo, icm_hi),
        "IP": (ip_lo, ip_hi),
        "semiancho_ICM": (icm_hi - icm_lo) / 2,
        "semiancho_IP": (ip_hi - ip_lo) / 2,
        "cociente_anchos": (ip_hi - ip_lo) / (icm_hi - icm_lo),
    }