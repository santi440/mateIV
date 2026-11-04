"""Script para B.4 (ICM e IP al 95% en x* = 2000 Hz).

Uso:
    python -m scripts.b4_intervalos
"""
from src.data_loader import load_airfoil
from src.b4 import calcular_icm_ip_2000


def main():
    df = load_airfoil("data/airfoil_self_ingle.dat" if False else "data/airfoil_self_noise.dat")
    res = calcular_icm_ip_2000(df, 2000.0)

    print("B.4 — Intervalos al 95% en x* = 2000 Hz (frequency -> sound_level)")
    print(f"Predicción ŷ* = {res['yhat_star']:.4f} dB")
    print(f"Valor crítico t_{{0.025, 1501}} = {res['tcrit']:.4f}")
    print(f"ICM (Intervalo de Confianza para la Media): [{res['icm_lo']:.4f}, {res['icm_hi']:.4f}] dB")
    print(f"IP  (Intervalo de Predicción para Nueva Obs):  [{res['ip_lo']:.4f}, {res['ip_hi']:.4f}] dB")
    print(f"Amplitud ICM = {res['icm_hi'] - res['icm_lo']:.4f} dB")
    print(f"Amplitud IP  = {res['ip_hi'] - res['ip_lo']:.4f} dB")


if __name__ == "__main__":
    main()
