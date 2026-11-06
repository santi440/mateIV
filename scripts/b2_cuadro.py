"""B.2: cuadro de las cuatro regresiones simples como imagen, para el PDF.

El cuadro que imprime scripts/b2_regresionLineal.py es más ancho que la página
del PDF y sus filas se parten al exportar. Este script toma la misma tabla que
calcula ese script (sin recalcular ni copiar números) y la dibuja como imagen.

Uso (desde la raíz del repo):
    python -m scripts.b2_cuadro

Salida: docs/figs/b2_cuadro.png
"""
import contextlib
import io
import runpy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SALIDA = "docs/figs/b2_cuadro.png"


def _intervalo(lo, hi, fmt):
    return f"[{lo:{fmt}} ; {hi:{fmt}}]".replace("-", "−")


def main():
    # Ejecuta el script del B.2 sin mostrar su salida y toma su tabla de resultados
    with contextlib.redirect_stdout(io.StringIO()):
        g = runpy.run_module("scripts.b2_regresionLineal")
    tabla, variables = g["tabla"], g["VARIABLES"]
    plt.close("all")

    columnas = [r"$X_i$", r"$\hat y=\hat\beta_1x+\hat\beta_0$", r"$\hat\sigma^2$ (dB²)",
                r"$R^2$", r"$r$", r"IC($\beta_1$)", r"IC($\beta_0$) (dB)",
                r"ICM($Y|x^*$) (dB)", r"IP($Y|x^*$) (dB)"]
    filas = []
    for v in variables:
        t = tabla.loc[v]
        signo = "+" if t.b0 >= 0 else "−"
        filas.append([
            v,
            f"{t.b1:.4g}·x {signo} {abs(t.b0):.2f}".replace("-", "−"),
            f"{t.sigma2:.2f}",
            f"{t.R2:.4f}",
            f"{t.r:.3f}".replace("-", "−"),
            _intervalo(t.IC_b1_lo, t.IC_b1_hi, ".4g"),
            _intervalo(t.IC_b0_lo, t.IC_b0_hi, ".2f"),
            _intervalo(t.ICM_lo, t.ICM_hi, ".2f"),
            _intervalo(t.IP_lo, t.IP_hi, ".2f"),
        ])

    fig, ax = plt.subplots(figsize=(12, 2.05), dpi=200)
    ax.axis("off")
    tb = ax.table(cellText=filas, colLabels=columnas, cellLoc="center", loc="center",
                  colWidths=[0.105, 0.165, 0.072, 0.062, 0.058, 0.20, 0.118, 0.118, 0.118])
    tb.auto_set_font_size(False)
    tb.set_fontsize(8.6)
    tb.scale(1, 1.75)
    for (r, c), celda in tb.get_celld().items():
        celda.set_edgecolor("#b0b0b0")
        if r == 0:
            celda.set_facecolor("#e8eef5")
            celda.set_text_props(weight="bold")
        elif c == 0:
            celda.set_text_props(family="monospace")
    ax.set_title("B.2 — Regresiones lineales simples de sound_level (intervalos al 95 %)",
                 fontsize=10.5, pad=4)
    fig.text(0.5, 0.04,
             "x* = 2000 Hz para frequency; x* = x̄ para angle, chord_length y displacement.   "
             "Unidades de β̂₁ e IC(β₁): dB/Hz, dB/grado, dB/m y dB/m.",
             ha="center", fontsize=8, color="#444")
    fig.savefig(SALIDA, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Figura: {SALIDA}")


if __name__ == "__main__":
    main()
