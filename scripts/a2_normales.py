"""A.2: sistema normal, recta de mínimos cuadrados y contraste con polyfit.

Uso (desde la raíz del repo):
    python -m scripts.a2_normales
"""
import numpy as np

from src import normales as nr
from src.data_loader import load_airfoil


def main():
    df = load_airfoil("data/airfoil_self_noise.dat")
    x = df["frequency"].to_numpy(float)
    y = df["sound_level"].to_numpy(float)

    s = nr.sumas_normales(x, y)
    A, d = nr.sistema_normal(x, y)
    b0_normal, b1_normal = nr.resolver_normal(x, y)
    b0_cerrada, b1_cerrada = nr.betas_cerradas(x, y)
    b1_poly, b0_poly = np.polyfit(x, y, 1)
    c = nr.contraste(x, y)

    b1_sumas = ((s["n"] * s["sum_xy"] - s["sum_x"] * s["sum_y"])
                / (s["n"] * s["sum_x2"] - s["sum_x"] ** 2))
    b0_sumas = (s["sum_y"] - b1_sumas * s["sum_x"]) / s["n"]

    print("A.2 — Parciales y ecuaciones normales (frequency -> sound_level)")
    print(f"n = {s['n']}")
    print(f"Σx  = {s['sum_x']:.6e} Hz")
    print(f"Σy  = {s['sum_y']:.6e} dB")
    print(f"Σx² = {s['sum_x2']:.6e} Hz²")
    print(f"Σxy = {s['sum_xy']:.6e} Hz·dB")
    print()
    print("Sistema normal A·[b0, b1]ᵀ = d:")
    print(A)
    print("d =", d)
    print()
    print(f"Solución del sistema: b0 = {b0_normal:.6f} dB, b1 = {b1_normal:.6e} dB/Hz")
    print(f"Fórmulas cerradas:    b0 = {b0_cerrada:.6f} dB, b1 = {b1_cerrada:.6e} dB/Hz")
    print(f"Desde sumatorias:     b0 = {b0_sumas:.6f} dB, b1 = {b1_sumas:.6e} dB/Hz")
    print(f"numpy.polyfit:        b0 = {b0_poly:.6f} dB, b1 = {b1_poly:.6e} dB/Hz")
    print()
    dif_cerrada = c["dif_normal_cerrada"]
    dif_polyfit = c["dif_normal_polyfit"]
    print(f"|normal - cerrada| = ({dif_cerrada[0]:.3e}, {dif_cerrada[1]:.3e})")
    print(f"|normal - polyfit| = ({dif_polyfit[0]:.3e}, {dif_polyfit[1]:.3e})")
    print(f"Gradiente en el óptimo = {c['gradiente_en_normal']}")
    print()
    print(f"Recta: sound_level = {b0_normal:.4f} dB "
          f"{b1_normal:+.6e} dB/Hz · frequency")


if __name__ == "__main__":
    main()
