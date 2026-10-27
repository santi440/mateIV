"""B.5 Regresión múltiple matricial (issue t3-1)."""
import numpy as np

VARIABLES = ["frequency", "angle", "chord_length", "displacement"] #la de velocity es la única q no pide

def armar_X_y(df):
    """Devuelve X (n x 5, con columna de unos) e y (n,)."""
    n = len(df) #1503
    unos = np.ones(n)
    X = np.column_stack([unos] + [df[v].to_numpy(float) for v in VARIABLES])
    y = df["sound_level"].to_numpy(float)
    return X, y

def betas(X, y):
    """beta = (X^T X)^-1 X^T y (ecuaciones normales, forma matricial)."""
    XtX = X.T @ X
    Xty = X.T @ y
    return np.linalg.inv(XtX) @ Xty


def r2_multiple(X, y, b):
    """R^2 = 1 - SCE/STC usando yhat = X @ b."""
    yhat = X @ b
    sce = ((y - yhat) ** 2).sum()
    stc = ((y - y.mean()) ** 2).sum()
    return float(1 - sce / stc)

def r2_ajustado(r2, n, k):
    """Fórmula de la consigna: 1 - (1 - R^2)(n - k)/(n - k - 1)."""
    return float(1 - (1 - r2) * (n - k) / (n - k - 1))


if __name__ == "__main__":
    from src.data_loader import load_airfoil

    df = load_airfoil("data/airfoil_self_noise.dat")
    X, y = armar_X_y(df)
    b = betas(X, y)
    r2 = r2_multiple(X, y, b)
    print("Betas [b0, frequency, angle, chord_length, displacement]:")
    print(b)
    print("R2   =", round(r2, 4))
    print("R2aj =", round(r2_ajustado(r2, len(y), len(VARIABLES)), 4))