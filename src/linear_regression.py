import numpy as np
from src import metrics


class LinearRegression:
    """Regresión lineal simple por mínimos cuadrados.

    Wrapper que reutiliza las funciones congeladas de src.metrics.
    Atributos:
        b0, b1: coeficientes estimados.
        x_mean, y_mean: medias de x e y en el entrenamiento.
        sxx, sxy: sumas de cuadrados.
    """

    def __init__(self):
        self.b0 = None
        self.b1 = None
        self.x_mean = None
        self.y_mean = None
        self.sxx = None
        self.sxy = None

    def calculate_sums(self, x, y):
        """Calcula Sxx, Sxy, x̄, ȳ usando metrics (contrato T1/T2/T3).

        Parameters
        ----------
        x : array-like
            Variable independiente.
        y : array-like
            Variable dependiente.

        Returns
        -------
        dict
            {'Sxx': float, 'Sxy': float, 'xbar': float, 'ybar': float}
        """
        self.x_mean = metrics.xbar(x)
        self.y_mean = metrics.ybar(y)
        self.sxx = metrics.Sxx(x)
        self.sxy = metrics.Sxy(x, y)

        return {
            'Sxx': self.sxx,
            'Sxy': self.sxy,
            'xbar': self.x_mean,
            'ybar': self.y_mean
        }

    def fit(self, x, y):
        """Calcula β₀ y β₁ mediante mínimos cuadrados.

        Parameters
        ----------
        x : array-like
            Variable independiente.
        y : array-like
            Variable dependiente.

        Returns
        -------
        tuple
            (β₀, β₁)
        """
        self.calculate_sums(x, y)
        self.b1 = metrics.beta1(self.sxy, self.sxx)
        self.b0 = metrics.beta0(self.y_mean, self.b1, self.x_mean)
        return self.b0, self.b1

    def predict(self, x):
        """Predice ŷ = β̂₀ + β̂₁·x.

        Parameters
        ----------
        x : array-like
            Valores de la variable independiente.

        Returns
        -------
        np.ndarray
            Predicciones ŷ.

        Raises
        ------
        ValueError
            Si el modelo no ha sido ajustado aún (fit() no llamado).
        """
        if self.b0 is None or self.b1 is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return metrics.yhat(self.b0, self.b1, x)
    