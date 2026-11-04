"""B.3: Test de hipótesis H₀: β₁ = 0 para las rectas simples y gráfico de dispersión."""
import numpy as np
from src import metrics as m
from src.b2_rectas import ajustar_rectas_simples
from src.data_loader import load_airfoil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

def test_hipotesis_betas(df):
    """Calcula t-estadístico y valor crítico para cada recta simple (alfa = 0.05)."""
    y = df["sound_level"].values
    n = len(df)
    df_err = n - 2
    tcrit = m.t_crit_95(df_err)
    rectas = ajustar_rectas_simples(df)
    
    resultados = []
    for r in rectas:
        var = r["variable"]
        x = df[var].values
        sxx = m.Sxx(x)
        s2 = m.sigma2(r["sce"], n)
        se = m.se_beta1(s2, sxx)
        t_val = m.t_stat(r["beta1"], s2, sxx)
        rechaza = abs(t_val) > tcrit
        
        resultados.append({
            "variable": var,
            "beta1": r["beta1"],
            "se_beta1": se,
            "t_stat": t_val,
            "t_crit": tcrit,
            "rechaza_H0": rechaza,
            "r2": r["r2"],
        })
    return resultados

def figura_scatter_mejor(df, path="docs/figs/b3_scatter_frequency.png"):
    """Scatter plot y recta de regresión para frequency vs sound_level."""
    x = df["frequency"].values
    y = df["sound_level"].values
    xb, yb = m.xbar(x), m.ybar(y)
    sxx = m.Sxx(x)
    sxy = m.Sxy(x, y)
    b1 = m.beta1(sxy, sxx)
    b0 = m.beta0(yb, b1, xb)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, alpha=0.3, color="blue", label="Datos (n = 1503)")
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = m.yhat(b0, b1, x_line)
    ax.plot(x_line, y_line, color="red", lw=2, label=f"Recta: ŷ = {b0:.2f} {b1:.4e}·x")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("sound_level (dB)")
    ax.set_title("B.3 — Regresión lineal simple: sound_level vs frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
