"""B.4: Intervalos de confianza para la media (ICM) y de predicción (IP) al 95% en x* = 2000 Hz."""
import numpy as np
from src import metrics as m

def calcular_icm_ip_2000(df, x_star=2000.0):
    """Calcula ICM e IP al 95% para x* = 2000 Hz en la regresión frequency -> sound_level."""
    x = df["frequency"].values
    y = df["sound_level"].values
    n = len(df)
    xb = m.xbar(x)
    yb = m.ybar(y)
    sxx = m.Sxx(x)
    sxy = m.Sxy(x, y)
    b1 = m.beta1(sxy, sxx)
    b0 = m.beta0(yb, b1, xb)
    
    sce_v = m.sce(x, y, b0, b1)
    s2 = m.sigma2(sce_v, n)
    tcrit = m.t_crit_95(n - 2)
    
    yhat_star = m.yhat(b0, b1, x_star)
    icm_lo, icm_hi, ip_lo, ip_hi = m.icm_ip(x_star, b0, b1, s2, n, xb, sxx, tcrit)
    
    return {
        "x_star": x_star,
        "yhat_star": yhat_star,
        "icm_lo": icm_lo,
        "icm_hi": icm_hi,
        "ip_lo": ip_lo,
        "ip_hi": ip_hi,
        "tcrit": tcrit,
        "sigma2": s2,
    }
