"""Script para B.1 (varianza, ANOVA, SCE/SCR/STC, R²).

Uso:
    python -m scripts.b1_varianza
"""
from src.data_loader import load_airfoil
from src.b1_varianza import analizar_varianza


def main():
    df = load_airfoil("data/airfoil_self_noise.dat")
    x = df["frequency"]
    y = df["sound_level"]
    res = analizar_varianza(x, y)

    print("B.1 — Varianza y Descomposición ANOVA (frequency -> sound_level)")
    print(f"n = {res['n']}")
    print(f"STC (Suma Total de Cuadrados)  = {res['stc']:.4f} dB²")
    print(f"SCR (Suma de Cuadrados Regres.) = {res['scr']:.4f} dB²")
    print(f"SCE (Suma de Cuadrados Error)   = {res['sce']:.4f} dB²")
    print(f"STC - (SCR + SCE) = {res['stc'] - (res['scr'] + res['sce']):.2e}")
    print(f"Varianza residual σ² = {res['sigma2']:.4f} dB²")
    print(f"Desvío residual σ    = {res['sigma']:.4f} dB")
    print(f"Coeficiente de determinación R² = {res['r2']:.4f} ({res['r2']*100:.2f}%)")


if __name__ == "__main__":
    main()
