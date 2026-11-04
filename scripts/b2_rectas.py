"""Script para B.2 (tabla de las 4 rectas simples).

Uso:
    python -m scripts.b2_rectas
"""
from src.data_loader import load_airfoil
from src.b2_rectas import ajustar_rectas_simples


def main():
    df = load_airfoil("data/airfoil_self_noise.dat")
    rectas = ajustar_rectas_simples(df)

    print("B.2 — Tabla de 4 Rectas Simples (sound_level ~ x)")
    print(f"{'Variable':<15} | {'β₀ (dB)':<12} | {'β₁ (Unidad)':<22} | {'SCE (dB²)':<12} | {'R²':<8} | {'r':<8}")
    print("-" * 88)
    for r in rectas:
        b1_str = f"{r['beta1']:.6e} {r['unidad_b1']}"
        print(f"{r['variable']:<15} | {r['beta0']:11.4f} | {b1_str:<22} | {r['sce']:11.2f} | {r['r2']:7.4f} | {r['r']:7.4f}")

    print("\nNota: velocity está excluida por la consigna.")


if __name__ == "__main__":
    main()
