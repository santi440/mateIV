"""A.1: ajuste cuadratico sound_level(frequency), derivada, punto critico + figura.

Usa src/metrics.py para R2/SCE/STC y
src/data_loader.load_airfoil.
Salida: docs/figs/a1_aproximacion.png + resumen numerico por consola.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import metrics as m
from src.data_loader import load_airfoil

df = load_airfoil("data/airfoil_self_noise.dat")
x = df["frequency"].to_numpy(float)
y = df["sound_level"].to_numpy(float)

# 1. Ajustes por minimos cuadrados:  min  Σᵢ [yᵢ − ŷ(xᵢ)]²
b2, b1, b0 = np.polyfit(x, y, 2)   # ŷ(x) = β₀ + β₁·x + β₂·x²
bl1, bl0 = np.polyfit(x, y, 1)     # ŷ(x) = β₀ + β₁·x  (recta A.2/B.1)

# 2. Derivada del cuadratico en dos x del rango:  dŷ/dx = β₁ + 2·β₂·x
EVAL = (1000.0, 10000.0)
der = {xs: b1 + 2 * b2 * xs for xs in EVAL}   # dŷ/dx│ₓ = β₁ + 2β₂x

# 3. Punto critico (dŷ/dx = 0) + clasificacion con 2da derivada
xc = -b1 / (2 * b2)        # x* = −β₁ / (2β₂)
segunda = 2 * b2           # d²ŷ/dx² = 2β₂  (>0 → mínimo, <0 → máximo)
yc = b0 + b1 * xc + b2 * xc ** 2   # y* = ŷ(x*) = β₀ + β₁x* + β₂x*²
dentro = 200.0 <= xc <= 20000.0    # ¿x* ∈ [200, 20000] observado?

# 4. Bondad de ajuste (contrato):  SCE = Σᵢ(yᵢ−ŷᵢ)²,  STC = Σᵢ(yᵢ−ȳ)²,  R² = 1 − SCE/STC
def r2_ajuste(yh):
    sce = float(((y - yh) ** 2).sum())   # SCE = Σ(yᵢ − ŷᵢ)²
    return m.r2(sce, m.stc(y)), sce      # R² = 1 − SCE/STC

xg = np.linspace(200, 20000, 400)
yh_q = b0 + b1 * xg + b2 * xg ** 2
yh_l = bl0 + bl1 * xg
r2q, sceq = r2_ajuste(b0 + b1 * x + b2 * x ** 2)
r2l, scel = r2_ajuste(bl0 + bl1 * x)

print(f"cuadratico: b0={b0:.4f} dB  b1={b1:.6e} dB/Hz  b2={b2:.6e} dB/Hz2")
print(f"lineal:     bl0={bl0:.4f} dB  bl1={bl1:.6e} dB/Hz")
for xs, d in der.items():
    print(f"dy/dx en x={xs:.0f} Hz = {d:.6e} dB/Hz  (~{d*100:+.3f} dB cada 100 Hz)")
print(f"critico x*={xc:.1f} Hz  y*={yc:.2f} dB  2da deriv.={segunda:.3e} "
      f"({'minimo' if segunda > 0 else 'maximo'})  dentro [200,20000]: {dentro}")
print(f"R2 cuadr.={r2q:.4f} (SCE={sceq:.1f})  R2 lin.={r2l:.4f} (SCE={scel:.1f})")

# 5. Figura: ajuste + residuos
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                               gridspec_kw={"hspace": 0.08})
ax1.scatter(x, y, s=8, alpha=0.35, label="mediciones (n=1503)")
ax1.plot(xg, yh_q, lw=2, label=f"cuadratico (R2={r2q:.3f})")
ax1.plot(xg, yh_l, lw=2, ls="--", label=f"lineal (R2={r2l:.3f})")
ax1.axvline(xc, color="k", ls=":", lw=1.5)
ax1.plot([xc], [yc], "ko", ms=6)
ax1.annotate(f"minimo x*={xc:.0f} Hz", (xc, yc),
             textcoords="offset points", xytext=(8, 12), fontsize=9)
ax1.set_ylabel("sound_level (dB)")
ax1.legend(fontsize=9)
ax1.set_title("A.1 — Aproximacion cuadratica vs lineal del ruido (dB) en funcion de la frecuencia (Hz)")
res_q = y - (b0 + b1 * x + b2 * x ** 2)   # eᵢ = yᵢ − ŷ(xᵢ), con ŷ(x) = β₀ + β₁x + β₂x²
ax2.scatter(x, res_q, s=8, alpha=0.35)
ax2.axhline(0, color="k", lw=1)
ax2.set_xlabel("frequency (Hz)")
ax2.set_ylabel("residuo cuadr. (dB)")
fig.savefig("docs/figs/a1_aproximacion.png", dpi=130, bbox_inches="tight")
print("figura: docs/figs/a1_aproximacion.png")
