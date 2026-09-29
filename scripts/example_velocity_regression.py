"""
Ejemplo de regresión lineal simple: Velocity vs Sound Level.
Demuestra uso de load_airfoil(), LinearRegression, y metrics.

Nota: velocity se excluye del modelo final (consigna), pero sirve como ejemplo.
"""

import sys
import os

# Agregar parent dir al path para importar src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import load_airfoil
from src.linear_regression import LinearRegression
from src import metrics


def main():
    # 1. Cargar dataset
    print("=" * 70)
    print("REGRESIÓN LINEAL: VELOCIDAD vs RUIDO AERODINÁMICO")
    print("=" * 70)
    print("\n[1] Cargando dataset...")
    df = load_airfoil('data/airfoil_self_noise.dat')
    print(f"    ✓ Dataset cargado: {df.shape[0]} filas × {df.shape[1]} columnas")

    # 2. Extraer variables
    print("\n[2] Extrayendo variables...")
    x = df['velocity'].values
    y = df['sound_level'].values
    n = len(x)
    print(f"    ✓ Variable independiente (x): velocity")
    print(f"    ✓ Variable dependiente (y): sound_level")
    print(f"    ✓ n = {n}")

    # 3. Ajustar modelo
    print("\n[3] Ajustando modelo de regresión lineal...")
    lr = LinearRegression()
    b0, b1 = lr.fit(x, y)
    print(f"    ✓ Modelo ajustado")

    # 4. Obtener estadísticos
    print("\n[4] Calculando estadísticos...")
    x_mean = lr.x_mean
    y_mean = lr.y_mean
    sxx = lr.sxx
    sxy = lr.sxy

    print(f"    Intercepto (β₀)             = {b0:.6f}")
    print(f"    Pendiente (β₁)              = {b1:.6f}")
    print(f"    Media de x (x̄)              = {x_mean:.6f}")
    print(f"    Media de y (ȳ)              = {y_mean:.6f}")
    print(f"    Suma de cuadrados Sxx       = {sxx:.2f}")
    print(f"    Suma de productos Sxy       = {sxy:.2f}")
    print(f"\n    Ecuación: ŷ = {b0:.4f} + {b1:.6f}·x")

    # 5. Calcular métricas adicionales
    print("\n[5] Métricas de bondad de ajuste...")
    y_pred = lr.predict(x)
    sce_val = metrics.sce(x, y, b0, b1)
    stc_val = metrics.stc(y)
    scr_val = metrics.scr(stc_val, sce_val)
    sigma2_val = metrics.sigma2(sce_val, n)
    r2_val = metrics.r2(sce_val, stc_val)
    r_val = metrics.r_corr(sxy, sxx, metrics.Sxx(y))  # syy = Sxx(y)

    print(f"    SCE (suma cuadrados error)  = {sce_val:.2f}")
    print(f"    STC (suma cuadrados total)  = {stc_val:.2f}")
    print(f"    SCR (suma cuadrados regr.)  = {scr_val:.2f}")
    print(f"    σ̂² (varianza residual)      = {sigma2_val:.4f}")
    print(f"    R² (coef. determinación)    = {r2_val:.6f}")
    print(f"    r (correlación)             = {r_val:.6f}")

    # 6. Generar gráfico
    print("\n[6] Generando gráfico...")
    plt.figure(figsize=(10, 6))

    # Puntos de datos
    plt.scatter(x, y, alpha=0.4, color='steelblue', edgecolors='gray', s=30, label='Datos observados')

    # Recta de regresión
    x_line = np.array([x.min(), x.max()])
    y_line = lr.predict(x_line)
    plt.plot(x_line, y_line, color='red', linewidth=2.5,
             label=f'ŷ = {b0:.2f} + {b1:.4f}·x')

    # Etiquetas y título
    plt.xlabel('Velocity (m/s)', fontsize=12, fontweight='bold')
    plt.ylabel('Sound Level (dB)', fontsize=12, fontweight='bold')
    plt.title('Regresión Lineal: Velocidad vs Ruido Aerodinámico', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')

    # Agregar texto con estadísticos
    textstr = f'n = {n}\nR² = {r2_val:.4f}\nr = {r_val:.4f}'
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # 7. Guardar gráfico
    output_path = 'plots/velocity_regression.png'
    os.makedirs('plots', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"    Gráfico guardado: {output_path}")

    # 8. Cerrar figura (comentar plt.show() en modo script)
    plt.close()

    print("\n" + "=" * 70)
    print("Script completado exitosamente")
    print("=" * 70)


if __name__ == "__main__":
    main()
