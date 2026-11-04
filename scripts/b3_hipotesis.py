"""Script para B.3 (test H₀: β₁ = 0 y scatter).

Uso:
    python -m scripts.b3_hipotesis
"""
from src.data_loader import load_airfoil
from src.b3 import test_hipotesis_betas, figura_scatter_mejor


def main():
    df = load_airfoil("data/airfoil_self_noise.dat")
    tests = test_hipotesis_betas(df)

    print("B.3 — Test de Hipótesis H₀: β₁ = 0 (α = 0.05)")
    print(f"{'Variable':<15} | {'β₁':<14} | {'EE(β₁)':<12} | {'T-stat':<10} | {'t_crit':<8} | {'¿Rechaza H₀?':<12}")
    print("-" * 78)
    for t in tests:
        rechazo = "SÍ (p < 0.05)" if t["rechaza_H0"] else "NO"
        print(f"{t['variable']:<15} | {t['beta1']:13.4e} | {t['se_beta1']:11.4e} | {t['t_stat']:9.2f} | {t['t_crit']:8.2f} | {rechazo:<12}")

    path_fig = figura_scatter_mejor(df)
    print(f"\nFigura scatter guardada en: {path_fig}")


if __name__ == "__main__":
    main()
