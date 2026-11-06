"""T2 - Regresiones simples de sound_level (velocity excluida)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data_loader import load_airfoil
from src.metrics import (
    xbar, ybar, Sxx, Sxy, beta1, beta0,
    sce, stc, scr, sigma2, r2, r_corr,
    se_beta1, se_beta0, ic_beta, icm_ip, t_crit_95, t_stat,
)

DATA_PATH = "data/airfoil_self_noise.dat"
VARIABLES = ["frequency", "angle", "chord_length", "displacement"]  
Y_COL = "sound_level"
ALPHA = 0.05  

df = load_airfoil(DATA_PATH)
print(df.shape)   
print(df.head())

x = df["frequency"]
y = df[Y_COL]
n = len(x)

xb, yb = xbar(x), ybar(y)
sxx, sxy, syy = Sxx(x), Sxy(x, y), stc(y)

b1 = beta1(sxy, sxx)
b0 = beta0(yb, b1, xb)
e = sce(x, y, b0, b1)
reg = scr(syy, e)
s2 = sigma2(e, n)
R2 = r2(e, syy)
r = r_corr(sxy, sxx, syy)

tc = t_crit_95(n - 2)
se_b1 = se_beta1(s2, sxx)
ic_b1 = ic_beta(b1, se_b1, tc)
se_b0 = se_beta0(s2, sxx, xb, n)
ic_b0 = ic_beta(b0, se_b0, tc)
icm_lo, icm_hi, ip_lo, ip_hi = icm_ip(2000, b0, b1, s2, n, xb, sxx, tc)


XSTAR = {"frequency": 2000.0}


def ajustar(var):
    x = df[var]
    y = df[Y_COL]
    n = len(x)

    xb, yb = xbar(x), ybar(y)
    sxx, sxy, syy = Sxx(x), Sxy(x, y), stc(y)

    b1 = beta1(sxy, sxx)
    b0 = beta0(yb, b1, xb)
    e = sce(x, y, b0, b1)
    s2 = sigma2(e, n)
    R2 = r2(e, syy)
    r = r_corr(sxy, sxx, syy)
    T = t_stat(b1, s2, sxx)

    tc = t_crit_95(n - 2)
    ic_b1 = ic_beta(b1, se_beta1(s2, sxx), tc)
    ic_b0 = ic_beta(b0, se_beta0(s2, sxx, xb, n), tc)

    x_star = XSTAR.get(var, xb)
    icm_lo, icm_hi, ip_lo, ip_hi = icm_ip(x_star, b0, b1, s2, n, xb, sxx, tc)

    return {
        "variable": var,
        "b1": b1,
        "b0": b0,
        "sigma2": s2,
        "R2": R2,
        "r": r,
        "T": T,
        "IC_b1_lo": ic_b1[0],
        "IC_b1_hi": ic_b1[1],
        "IC_b0_lo": ic_b0[0],
        "IC_b0_hi": ic_b0[1],
        "x_star": x_star,
        "ICM_lo": icm_lo,
        "ICM_hi": icm_hi,
        "IP_lo": ip_lo,
        "IP_hi": ip_hi,
    }


filas = [ajustar(v) for v in VARIABLES]
tabla = pd.DataFrame(filas).set_index("variable")
pd.set_option("display.float_format", "{:.6g}".format)
print(tabla.T)


elegida = tabla["R2"].idxmax()   # mayor R2 (y menor sigma2, mayor |T|)
T_elegida = tabla.loc[elegida, "T"]
rechaza_h0 = abs(T_elegida) > tc   # tc = t_crit_95(n - 2), calculado arriba

print(f"\nB.3 - variable elegida: {elegida}")
print(f"T = {T_elegida:.4f}   t(0.025, {n - 2}) = {tc}")
print(f"|T| > t critico -> {'se rechaza H0' if rechaza_h0 else 'no se rechaza H0'}")


b1_np, b0_np = np.polyfit(x, y, 1)
print("b1:", b1, b1_np, np.isclose(b1, b1_np))
print("b0:", b0, b0_np, np.isclose(b0, b0_np))

xs = np.linspace(x.min(), x.max(), 200)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(x, y, s=8, alpha=0.4, label="datos")
ax.plot(xs, b0 + b1 * xs, color="red", label="recta ajustada")
ax.set_xlabel("frequency (Hz)")
ax.set_ylabel("sound_level (dB)")
ax.legend()
fig.tight_layout()
fig.savefig("docs/figs/b2_dispersion_frequency.png", dpi=150)
plt.show()