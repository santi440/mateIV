"""A.1: ajuste cuadrático sound_level(frequency), derivada y punto crítico.

El modelo es

    ŷ(x) = b0 + b1·x + b2·x²,

con x = frequency (Hz) e y = sound_level (dB). La derivada es

    dŷ/dx = b1 + 2·b2·x   (dB/Hz),

y el punto crítico se obtiene de dŷ/dx = 0:

    x* = -b1 / (2·b2).

La segunda derivada d²ŷ/dx² = 2·b2 decide si es mínimo (b2 > 0) o máximo
(b2 < 0). El ajuste se contrasta con la recta de A.2 usando R² y SCE del
contrato src.metrics.
"""
import numpy as np

from src import metrics as m

RANGO = (200.0, 20000.0)
EVAL = (1000.0, 10000.0)


def ajustar_cuadratico(x, y):
    """Devuelve (b0, b1, b2) de ŷ = b0 + b1·x + b2·x² por mínimos cuadrados."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    b2, b1, b0 = np.polyfit(x, y, 2)
    return float(b0), float(b1), float(b2)


def ajustar_lineal(x, y):
    """Devuelve (b0, b1) de la recta ŷ = b0 + b1·x."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    b1, b0 = np.polyfit(x, y, 1)
    return float(b0), float(b1)


def evaluar_cuadratico(b0, b1, b2, x):
    x = np.asarray(x, dtype=float)
    return b0 + b1 * x + b2 * x ** 2


def derivada(b1, b2, x):
    """dŷ/dx = b1 + 2·b2·x en dB/Hz."""
    return b1 + 2.0 * b2 * np.asarray(x, dtype=float)


def punto_critico(b0, b1, b2, rango=RANGO):
    """Resuelve dŷ/dx=0, clasifica con la 2da derivada y chequea el rango."""
    if b2 == 0:
        raise ValueError("b2=0: el ajuste no es cuadrático")
    xc = -b1 / (2.0 * b2)
    yc = float(evaluar_cuadratico(b0, b1, b2, xc))
    segunda = 2.0 * b2
    if segunda > 0:
        tipo = "minimo"
    elif segunda < 0:
        tipo = "maximo"
    else:
        tipo = "inflexion"
    return {
        "x": float(xc),
        "y": yc,
        "segunda": float(segunda),
        "tipo": tipo,
        "dentro": bool(rango[0] <= xc <= rango[1]),
    }


def r2_sce(y, yhat):
    """R² = 1 - SCE/STC y SCE usando el contrato src.metrics."""
    y = np.asarray(y, dtype=float).ravel()
    yhat = np.asarray(yhat, dtype=float).ravel()
    sce = float(((y - yhat) ** 2).sum())
    return m.r2(sce, m.stc(y)), sce


def comparar_ajustes(x, y, eval_x=EVAL, rango=RANGO):
    """Resume ajuste cuadrático, recta, derivadas, crítico y bondad."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    b0, b1, b2 = ajustar_cuadratico(x, y)
    bl0, bl1 = ajustar_lineal(x, y)

    yq = evaluar_cuadratico(b0, b1, b2, x)
    yl = bl0 + bl1 * x
    r2q, sceq = r2_sce(y, yq)
    r2l, scel = r2_sce(y, yl)

    return {
        "cuadratico": {"b0": b0, "b1": b1, "b2": b2},
        "lineal": {"b0": bl0, "b1": bl1},
        "derivadas": {float(xs): float(derivada(b1, b2, xs))
                      for xs in eval_x},
        "critico": punto_critico(b0, b1, b2, rango),
        "r2_cuadratico": r2q,
        "r2_lineal": r2l,
        "sce_cuadratico": sceq,
        "sce_lineal": scel,
    }
