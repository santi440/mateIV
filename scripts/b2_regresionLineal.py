"""T2 - Regresiones simples de sound_level (velocity excluida)."""
import numpy as np
from src.data_loader import load_airfoil
from src.metrics import (
    xbar, ybar, Sxx, Sxy, beta1, beta0, yhat,
    sce, stc, scr, sigma2, r2, r_corr,
    se_beta1, se_beta0, ic_beta, icm_ip, t_crit_95,
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

# Paso 3: loop sobre VARIABLES y tabla final
#prueba mail
# Paso 4: elección de la mejor variable