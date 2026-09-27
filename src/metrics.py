"""Contrato numerico unico del TP (Esquema T, issue #2 CONGELADO).

T1/T2/T3 lo IMPORTAN, no lo duplican. Todo en numpy (sin sklearn/scipy).
Funciones puras sobre vectores 1-D. Nombres congelados:
Sxx, Sxy, xbar, ybar, beta1, beta0, sce, stc, scr, sigma2, r2,
ic_beta, icm_ip (+ helpers se_beta1/se_beta0, t_stat, r_corr, r2_adj,
t_crit_95, yhat).
"""
import numpy as np

__all__ = ["xbar", "ybar", "Sxx", "Sxy", "beta1", "beta0", "yhat",
           "sce", "stc", "scr", "sigma2", "r2", "r_corr", "r2_adj",
           "t_stat", "se_beta1", "se_beta0", "ic_beta", "icm_ip",
           "t_crit_95"]


def _v(a, name="x"):
    a = np.asarray(a, dtype=float).ravel()
    if a.size == 0:
        raise ValueError(f"{name} vacio")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} con NaN/inf")
    return a


def xbar(x):
    return float(_v(x).mean())   # x̄ = (1/n)·Σᵢxᵢ


def ybar(y):
    return float(_v(y, "y").mean())   # ȳ = (1/n)·Σᵢyᵢ


def Sxx(x):
    x = _v(x)
    return float(((x - x.mean()) ** 2).sum())   # Sxx = Σᵢ(xᵢ − x̄)²


def Sxy(x, y):
    x = _v(x)
    y = _v(y, "y")
    if x.size != y.size:
        raise ValueError("x e y con distinto n")
    return float(((x - x.mean()) * (y - y.mean())).sum())   # Sxy = Σᵢ(xᵢ − x̄)(yᵢ − ȳ)


def beta1(sxy, sxx):
    if sxx == 0:
        raise ValueError("Sxx=0 (x constante)")
    return float(sxy / sxx)   # β̂₁ = Sxy / Sxx


def beta0(yb, b1, xb):
    return float(yb - b1 * xb)   # β̂₀ = ȳ − β̂₁·x̄


def yhat(b0, b1, x):
    return b0 + np.asarray(x, dtype=float) * b1   # ŷ = β̂₀ + β̂₁·x


def sce(x, y, b0, b1):
    x = _v(x)
    y = _v(y, "y")
    return float(((y - (b0 + b1 * x)) ** 2).sum())   # SCE = Σᵢ[yᵢ − (β̂₀ + β̂₁xᵢ)]²


def stc(y):
    y = _v(y, "y")
    return float(((y - y.mean()) ** 2).sum())   # STC = Σᵢ(yᵢ − ȳ)²


def scr(stc_v, sce_v):
    return float(stc_v - sce_v)   # SCR = STC − SCE  (STC = SCR + SCE)


def sigma2(sce_v, n):
    if n <= 2:
        raise ValueError("n debe ser > 2")
    return float(sce_v / (n - 2))   # σ̂² = SCE / (n − 2)


def r2(sce_v, stc_v):
    if stc_v == 0:
        raise ValueError("STC=0 (y constante)")
    return float(1 - sce_v / stc_v)   # R² = 1 − SCE/STC


def r_corr(sxy, sxx, syy):
    if sxx <= 0 or syy <= 0:
        raise ValueError("Sxx/Syy no positivos")
    return float(sxy / np.sqrt(sxx * syy))   # r = Sxy / √(Sxx·Syy)


def r2_adj(r2v, n, k):
    """R2 ajustado = 1-(1-R2)(n-k)/(n-k-1); k = nro de parametros con intercepto."""
    if n - k - 1 <= 0:
        raise ValueError("n-k-1 debe ser > 0")
    return float(1 - (1 - r2v) * (n - k) / (n - k - 1))   # R²aj = 1 − (1−R²)(n−k)/(n−k−1)


def t_stat(b1, s2, sxx_v):
    return float(b1 / np.sqrt(s2 / sxx_v))   # T = β̂₁ / √(σ̂²/Sxx)  ~ t(n−2) bajo H₀


def se_beta1(s2, sxx_v):
    return float(np.sqrt(s2 / sxx_v))   # SE(β̂₁) = √(σ̂²/Sxx)


def se_beta0(s2, sxx_v, xb, n):
    return float(np.sqrt(s2 * (1 / n + xb ** 2 / sxx_v)))   # SE(β̂₀) = √[σ̂²(1/n + x̄²/Sxx)]


def ic_beta(b, se, tcrit):
    """IC generico: b +/- tcrit*se (vale para beta0 y beta1)."""
    return (float(b - tcrit * se), float(b + tcrit * se))   # IC = β̂ ± t(α/2,n−2)·SE


def icm_ip(x_star, b0, b1, s2, n, xb, sxx_v, tcrit):
    """(icm_lo, icm_hi, ip_lo, ip_hi) al 95% para x_star (formulas consigna B.4)."""
    yh = b0 + b1 * x_star   # ŷ* = β̂₀ + β̂₁·x*
    se_mean = np.sqrt(s2 * (1 / n + (x_star - xb) ** 2 / sxx_v))   # SE_media = √[σ̂²(1/n + (x*−x̄)²/Sxx)]
    se_pred = np.sqrt(s2 * (1 + 1 / n + (x_star - xb) ** 2 / sxx_v))   # SE_pred = √[σ̂²(1 + 1/n + (x*−x̄)²/Sxx)]
    return (float(yh - tcrit * se_mean), float(yh + tcrit * se_mean),   # ICM = ŷ* ± t·SE_media
            float(yh - tcrit * se_pred), float(yh + tcrit * se_pred))   # IP  = ŷ* ± t·SE_pred


# t(.025, df) tabla bilateral 95% (sin scipy): valores exactos usuales;
# df>=120 -> 1.96 (para n=1503, df=1501, error < 0.002 vs 1.9616).
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
         7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086,
         30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}


def t_crit_95(df):
    df = int(df)
    if df < 1:
        raise ValueError("df debe ser >= 1")
    if df in _T975:
        return _T975[df]
    below = [d for d in _T975 if d < df]
    if not below:
        return _T975[1]
    nearest = max(below)
    if nearest >= 120 or df >= 120:
        return 1.96
    return _T975[nearest]
