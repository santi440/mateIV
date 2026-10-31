"""B.1 (varianza) y B.4 (ICM e IP en x* = 2000 Hz).

Uso (desde la raíz del repo):
    python -m scripts.b1_b4_varianza
"""
from src import varianza as v
from src.data_loader import load_airfoil


def main():
    df = load_airfoil("data/airfoil_self_noise.dat")
    x = df["frequency"].to_numpy(float)
    y = df["sound_level"].to_numpy(float)

    d = v.descomposicion(x, y)
    i = v.intervalos(x, y)

    print("B.1 — Descomposición de la variabilidad (sound_level ~ frequency)")
    print(f"Recta: sound_level = {d['b0']:.4f} dB "
          f"{d['b1']:+.6e} dB/Hz · frequency   (n = {d['n']})")
    print()
    print(f"{'Fuente':<22}{'Suma de cuadrados (dB²)':>26}")
    print(f"{'Regresión (SCR)':<22}{d['SCR']:>26.4f}")
    print(f"{'Error (SCE)':<22}{d['SCE']:>26.4f}")
    print(f"{'Total (STC)':<22}{d['STC']:>26.4f}")
    print(f"{'SCR + SCE':<22}{d['SCR+SCE']:>26.4f}")
    print(f"{'STC − (SCR + SCE)':<22}{d['diferencia']:>26.2e}")
    print()
    print(f"σ̂² = SCE/(n−2) = {d['sigma2']:.4f} dB²")
    print(f"R² = 1 − SCE/STC = {d['R2']:.6f}  → frequency explica el "
          f"{100 * d['R2']:.1f} % de la variabilidad de sound_level.")
    print()
    print(f"B.4 — Intervalos al 95 % en x* = {i['x_star']:.0f} Hz "
          f"(t = {i['t']})")
    print(f"ŷ* = {i['y_star']:.4f} dB")
    print(f"ICM (respuesta media): ({i['ICM'][0]:.4f} ; {i['ICM'][1]:.4f}) dB"
          f"   ± {i['semiancho_ICM']:.4f} dB")
    print(f"IP  (medición futura): ({i['IP'][0]:.4f} ; {i['IP'][1]:.4f}) dB"
          f"   ± {i['semiancho_IP']:.4f} dB")
    print(f"El IP es {i['cociente_anchos']:.1f} veces más ancho que el ICM.")


if __name__ == "__main__":
    main()