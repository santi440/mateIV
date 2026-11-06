"""EDA del Airfoil Self-Noise: descriptivos, discretas e histogramas.

Uso (desde la raíz del repo):
    python -m scripts.eda
"""
import matplotlib
matplotlib.use("Agg")

from src.data_loader import load_airfoil
from src.eda import figura_histogramas, resumen_discretas, tabla_descriptivos


def main():
    df = load_airfoil("data/airfoil_self_noise.dat")

    print("EDA — Airfoil Self-Noise (n = 1503)")
    print("\nDescriptivos:")
    print(tabla_descriptivos(df).to_string(float_format=lambda v: f"{v:.6g}"))

    print("\nVariables con pocos valores distintos:")
    r = resumen_discretas(df)
    for col in ["frequency", "angle", "chord_length", "velocity"]:
        print(f"{col}: {r[col]['n_valores']} valores -> {r[col]['valores']}")
    print("velocity (conteos):",
          {k: int(v) for k, v in r["velocity"]["conteos"].items()})
    print(f"displacement: {r['displacement']['n_valores']} valores distintos, "
          f"rango [{r['displacement']['min']:.6e}, "
          f"{r['displacement']['max']:.6e}] m")

    print("\nNota: velocity se usa solo en el EDA; B.2 y B.5 la excluyen.")

    path = figura_histogramas(df, "docs/figs/eda_histogramas.png")
    print(f"\nFigura: {path}")


if __name__ == "__main__":
    main()
