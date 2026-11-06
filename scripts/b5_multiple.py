"""B.5: modelo múltiple + figura observado vs predicho para el informe."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data_loader import load_airfoil
from src.multiple import armar_X_y, betas, r2_ajustado, r2_multiple
from src.varianza import descomposicion


def main():
    df = load_airfoil("data/airfoil_self_noise.dat")
    X, y = armar_X_y(df)
    b = betas(X, y)
    yhat = X @ b
    r2 = r2_multiple(X, y, b)
    r2a = r2_ajustado(r2, len(y), 4)

    nombres = ["intercepto", "frequency", "angle", "chord_length", "displacement"]
    unidades = ["dB", "dB/Hz", "dB/grado", "dB/m", "dB/m"]
    print("B.5 — Regresión múltiple (velocity excluida)")
    print(f"{'coeficiente':15s} {'valor':>14s}  unidad")
    for nombre, valor, unidad in zip(nombres, b, unidades):
        print(f"{nombre:15s} {valor:14.6e}  {unidad}")
    print(f"R2 = {r2:.4f}   R2 ajustado = {r2a:.4f}")

    # Comparacion con el mejor modelo simple (B.2: frequency), con la misma formula
    r2s = descomposicion(df["frequency"], df["sound_level"])["R2"]
    r2s_a = r2_ajustado(r2s, len(y), 1)
    print(f"Simple (frequency): R2 = {r2s:.4f}   R2 ajustado = {r2s_a:.4f}")

    lims = [min(y.min(), yhat.min()), max(y.max(), yhat.max())]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y, yhat, s=10, alpha=0.35, label="mediciones")
    ax.plot(lims, lims, "k--", lw=1.2, label="identidad 1:1")
    ax.set_xlabel("sound_level observado (dB)")
    ax.set_ylabel("sound_level predicho (dB)")
    ax.set_title(f"B.5 — Observado vs predicho (R²={r2:.3f}, R²aj={r2a:.3f})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig("docs/figs/b5_predicho_observado.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)
    print("figura: docs/figs/b5_predicho_observado.png")


if __name__ == "__main__":
    main()
