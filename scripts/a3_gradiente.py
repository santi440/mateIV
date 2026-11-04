"""A.3: descenso por gradiente para el modelo múltiple de B.5.

Uso (desde la raíz del repo):
    python -m scripts.a3_gradiente

Salida: docs/figs/a3_convergencia.png + tabla numérica por consola.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data_loader import load_airfoil
from src.gradiente import (betas_a_original, descenso_gradiente,
                           estandarizar_features, paso_maximo_estable)
from src.multiple import armar_X_y, betas, r2_multiple


def figura_convergencia(resultados, f_exacta, path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for nombre, r in resultados.items():
        k = np.arange(1, r["hist_f"].size + 1)
        ax.plot(k, r["hist_f"], lw=1.8,
                label=f"{nombre}: α={r['alpha']:.4f} ({r['estado']})")
    ax.axhline(f_exacta, color="k", ls="--", lw=1.2,
               label=f"exacta: f={f_exacta:.4f} dB²")
    ax.set_yscale("log")
    ax.set_xlabel("iteración k")
    ax.set_ylabel("f(β⁽ᵏ⁾) = SCE/(2n) (dB²)")
    ax.set_title("A.3 — Descenso por gradiente: convergencia según α")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    df = load_airfoil("data/airfoil_self_noise.dat")
    X, y = armar_X_y(df)
    Xs, medias, desvios = estandarizar_features(X)
    L, alpha_max = paso_maximo_estable(Xs)

    b_exacta = betas(X, y)
    r2 = r2_multiple(X, y, b_exacta)
    f_exacta = float(((y - X @ b_exacta) ** 2).sum() / (2 * len(y)))

    alphas = {
        "chica": 0.05 * alpha_max,
        "ok": 0.50 * alpha_max,
        "diverge": 1.10 * alpha_max,
    }
    #Dandome cuenta que deberia de haber cursado mate 4 antes que big data -.-
    resultados = {
        nombre: descenso_gradiente(Xs, y, alpha, max_iter=6000, tol=1e-10)
        for nombre, alpha in alphas.items()
    }

    print("A.3 — Descenso por gradiente (modelo B.5)")
    print("Features estandarizadas (media, desvío):")
    for nombre, mu, sd in zip(["frequency", "angle", "chord_length",
                               "displacement"], medias, desvios):
        print(f"  {nombre:13s} media={mu:.6e}  desvío={sd:.6e}")
    print(f"\nL = λmax(XᵀX/n) = {L:.6f}   α_max = 2/L = {alpha_max:.6f}")
    print(f"β exacta = {np.array2string(b_exacta, precision=6)}")
    print(f"R² exacto = {r2:.4f}   f exacta = {f_exacta:.6f} dB²\n")

    print(f"{'caso':8s} {'alpha':>10s} {'iter':>6s} {'estado':>10s} "
          f"{'f final':>12s} {'||grad||':>10s} {'||β-βex||':>12s}")
    for nombre, r in resultados.items():
        b_orig = betas_a_original(r["b"], medias, desvios)
        err = float(np.linalg.norm(b_orig - b_exacta))
        print(f"{nombre:8s} {r['alpha']:10.6f} {r['iteraciones']:6d} "
              f"{r['estado']:>10s} {r['hist_f'][-1]:12.4f} "
              f"{r['hist_grad'][-1]:10.2e} {err:12.3e}")

    path = figura_convergencia(resultados, f_exacta,
                               "docs/figs/a3_convergencia.png")
    print(f"\nFigura: {path}")


if __name__ == "__main__":
    main()
