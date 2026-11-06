"""A.1: ajuste cuadratico sound_level(frequency), derivada, punto critico + figura.

Usa src/cuadratico.py para el ajuste y src/metrics.py para R2/SCE/STC.
Salida: docs/figs/a1_aproximacion.png + resumen numerico por consola.

Uso (desde la raíz del repo):
    python -m scripts.a1_cuadratico
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.cuadratico import (EVAL, RANGO, comparar_ajustes,
                            evaluar_cuadratico)
from src.data_loader import load_airfoil


def main():
    df = load_airfoil("data/airfoil_self_noise.dat")
    x = df["frequency"].to_numpy(float)
    y = df["sound_level"].to_numpy(float)

    res = comparar_ajustes(x, y, eval_x=EVAL, rango=RANGO)
    c = res["cuadratico"]
    lin = res["lineal"]
    crit = res["critico"]
    b0, b1, b2 = c["b0"], c["b1"], c["b2"]
    bl0, bl1 = lin["b0"], lin["b1"]

    print(f"cuadratico: b0={b0:.4f} dB  b1={b1:.6e} dB/Hz  b2={b2:.6e} dB/Hz2")
    print(f"lineal:     bl0={bl0:.4f} dB  bl1={bl1:.6e} dB/Hz")
    for xs, d in res["derivadas"].items():
        print(f"dy/dx en x={xs:.0f} Hz = {d:.6e} dB/Hz  (~{d*100:+.3f} dB cada 100 Hz)")
    print(f"critico x*={crit['x']:.1f} Hz  y*={crit['y']:.2f} dB  "
          f"2da deriv.={crit['segunda']:.3e}  ({crit['tipo']})  "
          f"dentro [200,20000]: {crit['dentro']}")
    print(f"R2 cuadr.={res['r2_cuadratico']:.4f} "
          f"(SCE={res['sce_cuadratico']:.1f})  "
          f"R2 lin.={res['r2_lineal']:.4f} "
          f"(SCE={res['sce_lineal']:.1f})")

    xg = np.linspace(RANGO[0], RANGO[1], 400)
    yh_q = evaluar_cuadratico(b0, b1, b2, xg)
    yh_l = bl0 + bl1 * xg

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=False,
                                   gridspec_kw={"hspace": 0.3})
    ax1.scatter(x, y, s=8, alpha=0.35, label="mediciones (n=1503)")
    ax1.plot(xg, yh_q, lw=2,
             label=f"cuadratico (R2={res['r2_cuadratico']:.3f})")
    ax1.plot(xg, yh_l, lw=2, ls="--",
             label=f"lineal (R2={res['r2_lineal']:.3f})")
    ax1.axvline(crit["x"], color="k", ls=":", lw=1.5)
    ax1.plot([crit["x"]], [crit["y"]], "ko", ms=6)
    ax1.annotate(f"minimo x*={crit['x']:.0f} Hz", (crit["x"], crit["y"]),
                 textcoords="offset points", xytext=(8, 12), fontsize=9)
    ax1.set_xlabel("frequency (Hz)")
    ax1.set_ylabel("sound_level (dB)")
    ax1.legend(fontsize=9)
    ax1.set_title("A.1 — Aproximacion cuadratica vs lineal del ruido (dB) "
                  "en funcion de la frecuencia (Hz)")
    res_q = y - evaluar_cuadratico(b0, b1, b2, x)
    ax2.scatter(x, res_q, s=8, alpha=0.35)
    ax2.axhline(0, color="k", lw=1)
    ax2.set_xlabel("frequency (Hz)")
    ax2.set_ylabel("residuo cuadr. (dB)")
    fig.savefig("docs/figs/a1_aproximacion.png", dpi=130, bbox_inches="tight")
    print("figura: docs/figs/a1_aproximacion.png")


if __name__ == "__main__":
    main()
