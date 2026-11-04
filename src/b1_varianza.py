"""B.1: Varianza residual, descomposición ANOVA (STC = SCR + SCE), sigma² y R²."""
import numpy as np
from src import metrics as m

def analizar_varianza(x, y):
    """Calcula Sxx, Sxy, beta0, beta1, STC, SCE, SCR, sigma2 y R2."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = len(x)
    xb = m.xbar(x)
    yb = m.ybar(y)
    sxx = m.Sxx(x)
    sxy = m.Sxy(x, y)
    b1 = m.beta1(sxy, sxx)
    b0 = m.beta0(yb, b1, xb)
    
    stc_v = m.stc(y)
    sce_v = m.sce(x, y, b0, b1)
    scr_v = m.scr(stc_v, sce_v)
    s2 = m.sigma2(sce_v, n)
    r2_v = m.r2(sce_v, stc_v)
    
    return {
        "n": n,
        "xbar": xb,
        "ybar": yb,
        "Sxx": sxx,
        "Sxy": sxy,
        "beta0": b0,
        "beta1": b1,
        "stc": stc_v,
        "sce": sce_v,
        "scr": scr_v,
        "sigma2": s2,
        "sigma": np.sqrt(s2),
        "r2": r2_v,
    }
