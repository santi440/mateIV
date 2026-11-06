"""EDA del dataset Airfoil Self-Noise (issue t0-3).

Descriptivos, conteo de valores distintos e histogramas con unidades.
`velocity` se describe solo en el EDA: la consigna la excluye de B.2 y B.5.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.plots import UNITS


def tabla_descriptivos(df):
    """Tabla count/mean/std/min/cuartiles/max para las 6 columnas."""
    return df.describe().T[["count", "mean", "std", "min", "25%", "50%",
                            "75%", "max"]]


def resumen_discretas(df):
    """Cantidad y valores de las columnas con pocos valores distintos."""
    return {
        "frequency": {
            "n_valores": int(df["frequency"].nunique()),
            "valores": sorted(df["frequency"].unique().tolist()),
        },
        "angle": {
            "n_valores": int(df["angle"].nunique()),
            "valores": sorted(df["angle"].unique().tolist()),
        },
        "chord_length": {
            "n_valores": int(df["chord_length"].nunique()),
            "valores": sorted(df["chord_length"].unique().tolist()),
        },
        "velocity": {
            "n_valores": int(df["velocity"].nunique()),
            "valores": sorted(df["velocity"].unique().tolist()),
            "conteos": df["velocity"].value_counts().sort_index().to_dict(),
        },
        "displacement": {
            "n_valores": int(df["displacement"].nunique()),
            "min": float(df["displacement"].min()),
            "max": float(df["displacement"].max()),
        },
    }


def _bins(datos):
    """Bordes centrados en cada valor distinto si hay pocos; si no, 40 bins."""
    valores = np.sort(datos.unique())
    if valores.size <= 40:
        if valores.size == 1:
            return np.array([valores[0] - 0.5, valores[0] + 0.5])
        pasos = np.diff(valores) / 2.0
        bordes = np.concatenate(([valores[0] - pasos[0]],
                                 valores[:-1] + pasos,
                                 [valores[-1] + pasos[-1]]))
        return bordes
    return 40


def figura_histogramas(df, path):
    """Histogramas 2x3 de las 6 columnas, con unidades en los ejes."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.ravel(), df.columns):
        datos = df[col]
        ax.hist(datos, bins=_bins(datos), edgecolor="white")
        ax.set_xlabel(f"{col} ({UNITS[col]})")
        ax.set_ylabel("frecuencia absoluta (mediciones)")
        ax.set_title(f"{col}: {datos.nunique()} valores distintos")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
