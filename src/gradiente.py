"""A.3: estimación de B.5 por descenso por gradiente.

El modelo múltiple es

    y = X·β + ε,

con X incluyendo la columna de unos. Se minimiza

    f(β) = SCE(β)/(2n) = ||y - X·β||²/(2n).

El gradiente es

    ∇f(β) = -Xᵀ(y - X·β)/n,

y la actualización es

    β⁽ᵏ⁺¹⁾ = β⁽ᵏ⁾ - α·∇f(β⁽ᵏ⁾).

La dirección -∇f es de descenso porque la derivada direccional de f en esa
dirección vale -||∇f||² < 0. Las features se estandarizan antes de iterar
para que frequency (Hz, ~10⁴) y displacement (m, ~10⁻³) no impongan escalas
incompatibles en el mismo gradiente.
"""
import numpy as np


def estandarizar_features(X):
    """Estandariza las columnas de X salvo el intercepto (columna 0)."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError("X debe ser 2-D con intercepto y al menos una feature")
    medias = X[:, 1:].mean(axis=0)
    desvios = X[:, 1:].std(axis=0, ddof=0)
    if np.any(desvios == 0):
        raise ValueError("alguna feature tiene desvío cero")
    Xs = X.copy()
    Xs[:, 1:] = (X[:, 1:] - medias) / desvios
    return Xs, medias, desvios


def paso_maximo_estable(X):
    """Devuelve L = λmax(XᵀX/n) y α_max = 2/L para gradiente descendente."""
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    L = float(np.linalg.eigvalsh(X.T @ X / n).max())
    if L <= 0:
        raise ValueError("XᵀX/n no definida positiva")
    return L, 2.0 / L


def betas_a_original(b_est, medias, desvios):
    """Vuelve de coeficientes estandarizados a la escala original."""
    b_est = np.asarray(b_est, dtype=float)
    medias = np.asarray(medias, dtype=float)
    desvios = np.asarray(desvios, dtype=float)
    b0 = b_est[0] - np.sum(b_est[1:] * medias / desvios)
    return np.concatenate(([b0], b_est[1:] / desvios))


def descenso_gradiente(X, y, alpha, max_iter=10000, tol=1e-10, f_max=1e12):
    """Corre β -= α·∇f y devuelve coeficientes, historial y estado."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim != 2 or X.shape[0] != y.size:
        raise ValueError("X e y incompatibles")
    if alpha <= 0:
        raise ValueError("alpha debe ser > 0")

    n, p = X.shape
    b = np.zeros(p)
    hist_f = []
    hist_grad = []

    for k in range(max_iter):
        residuos = y - X @ b
        f = float(residuos @ residuos / (2.0 * n))
        grad = -(X.T @ residuos) / n
        norma_grad = float(np.linalg.norm(grad))
        hist_f.append(f)
        hist_grad.append(norma_grad)

        if norma_grad < tol:
            return {
                "b": b,
                "hist_f": np.array(hist_f),
                "hist_grad": np.array(hist_grad),
                "iteraciones": k + 1,
                "estado": "convergio",
                "alpha": float(alpha),
            }

        b = b - alpha * grad
        if not np.isfinite(b).all() or f > f_max:
            return {
                "b": b,
                "hist_f": np.array(hist_f),
                "hist_grad": np.array(hist_grad),
                "iteraciones": k + 1,
                "estado": "diverge",
                "alpha": float(alpha),
            }

    return {
        "b": b,
        "hist_f": np.array(hist_f),
        "hist_grad": np.array(hist_grad),
        "iteraciones": max_iter,
        "estado": "max_iter",
        "alpha": float(alpha),
    }
