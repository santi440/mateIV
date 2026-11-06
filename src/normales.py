"""A.2: derivadas parciales, sistema de ecuaciones normales y contraste.

La suma de cuadrados a minimizar es

    SCE(b0, b1) = Σᵢ (yᵢ - b0 - b1·xᵢ)²

Sus derivadas parciales son

    ∂SCE/∂b0 = -2·Σᵢ (yᵢ - b0 - b1·xᵢ)
    ∂SCE/∂b1 = -2·Σᵢ xᵢ·(yᵢ - b0 - b1·xᵢ)

Igualando a cero se obtiene el sistema normal

    n·b0 + (Σxᵢ)·b1   = Σyᵢ
    (Σxᵢ)·b0 + (Σxᵢ²)·b1 = Σxᵢyᵢ

cuya solución es la recta de mínimos cuadrados. Este módulo resuelve el
sistema en forma matricial y contrasta el resultado con las fórmulas
cerradas de src.metrics y con numpy.polyfit.
"""
import numpy as np

from src import metrics as m


def _vectores(x, y):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError("x e y con distinto n")
    if x.size == 0:
        raise ValueError("x e y vacios")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("x o y con NaN/inf")
    return x, y


def sumas_normales(x, y):
    """Devuelve n, Σx, Σy, Σx² y Σxy usados en el sistema normal."""
    x, y = _vectores(x, y)
    return {
        "n": int(x.size),
        "sum_x": float(x.sum()),
        "sum_y": float(y.sum()),
        "sum_x2": float((x ** 2).sum()),
        "sum_xy": float((x * y).sum()),
    }


def sistema_normal(x, y):
    """Arma A·[b0, b1]ᵀ = d a partir de las sumas normales."""
    s = sumas_normales(x, y)
    A = np.array([[s["n"], s["sum_x"]],
                  [s["sum_x"], s["sum_x2"]]], dtype=float)
    d = np.array([s["sum_y"], s["sum_xy"]], dtype=float)
    return A, d


def resolver_normal(x, y):
    """Resuelve el sistema normal y devuelve (b0, b1)."""
    A, d = sistema_normal(x, y)
    b0, b1 = np.linalg.solve(A, d)
    return float(b0), float(b1)


def gradiente(x, y, b0, b1):
    """Gradiente de SCE respecto de (b0, b1)."""
    x, y = _vectores(x, y)
    residuos = y - b0 - b1 * x
    d_b0 = -2.0 * residuos.sum()
    d_b1 = -2.0 * (x * residuos).sum()
    return np.array([d_b0, d_b1], dtype=float)


def betas_cerradas(x, y):
    """β̂1 = Sxy/Sxx y β̂0 = ȳ - β̂1·x̄ usando el contrato src.metrics."""
    x, y = _vectores(x, y)
    b1 = m.beta1(m.Sxy(x, y), m.Sxx(x))
    b0 = m.beta0(m.ybar(y), b1, m.xbar(x))
    return float(b0), float(b1)


def contraste(x, y):
    """Compara sistema normal, fórmulas cerradas y numpy.polyfit."""
    x, y = _vectores(x, y)
    b0_normal, b1_normal = resolver_normal(x, y)
    b0_cerrada, b1_cerrada = betas_cerradas(x, y)
    b1_poly, b0_poly = np.polyfit(x, y, 1)
    return {
        "normal": (b0_normal, b1_normal),
        "cerrada": (b0_cerrada, b1_cerrada),
        "polyfit": (float(b0_poly), float(b1_poly)),
        "dif_normal_cerrada": (abs(b0_normal - b0_cerrada),
                               abs(b1_normal - b1_cerrada)),
        "dif_normal_polyfit": (abs(b0_normal - float(b0_poly)),
                               abs(b1_normal - float(b1_poly))),
        "gradiente_en_normal": gradiente(x, y, b0_normal, b1_normal),
    }
