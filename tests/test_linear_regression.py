import numpy as np
import pytest
from src.linear_regression import LinearRegression
from src import metrics


class TestLinearRegression:
    """Suite de pruebas para LinearRegression."""

    def test_init(self):
        """El modelo no entrenado tiene coeficientes None."""
        lr = LinearRegression()
        assert lr.b0 is None
        assert lr.b1 is None
        assert lr.x_mean is None
        assert lr.y_mean is None

    def test_calculate_sums(self):
        """calculate_sums retorna dict con Sxx, Sxy, xbar, ybar."""
        lr = LinearRegression()
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 5, 4, 5])

        result = lr.calculate_sums(x, y)

        assert isinstance(result, dict)
        assert set(result.keys()) == {'Sxx', 'Sxy', 'xbar', 'ybar'}
        assert result['xbar'] == metrics.xbar(x)
        assert result['ybar'] == metrics.ybar(y)
        assert result['Sxx'] == metrics.Sxx(x)
        assert result['Sxy'] == metrics.Sxy(x, y)

    def test_fit_perfect_line(self):
        """fit() calcula coeficientes correctos para una recta perfecta."""
        lr = LinearRegression()
        x = np.array([1, 2, 3, 4, 5])
        y = 2 * x + 3  # y = 2x + 3

        b0, b1 = lr.fit(x, y)

        assert np.isclose(b1, 2.0)
        assert np.isclose(b0, 3.0)

    def test_fit_stores_params(self):
        """fit() almacena b0, b1, x_mean, y_mean."""
        lr = LinearRegression()
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 5, 4, 5])

        lr.fit(x, y)

        assert lr.b0 is not None
        assert lr.b1 is not None
        assert lr.x_mean is not None
        assert lr.y_mean is not None
        assert lr.sxx is not None
        assert lr.sxy is not None

    def test_predict_before_fit_raises(self):
        """predict() lanza ValueError si no se ha llamado fit()."""
        lr = LinearRegression()
        x = np.array([1, 2, 3])

        with pytest.raises(ValueError, match="not fitted"):
            lr.predict(x)

    def test_predict_after_fit(self):
        """predict() retorna predicciones correctas después de fit()."""
        lr = LinearRegression()
        x_train = np.array([1, 2, 3, 4, 5])
        y_train = 2 * x_train + 3

        lr.fit(x_train, y_train)

        x_test = np.array([2.5, 3.5])
        y_pred = lr.predict(x_test)
        y_expected = 2 * x_test + 3

        assert np.allclose(y_pred, y_expected)

    def test_predict_array_shape(self):
        """predict() retorna mismo shape que entrada."""
        lr = LinearRegression()
        x = np.array([1, 2, 3, 4, 5])
        y = 1.5 * x + 0.5

        lr.fit(x, y)

        x_test = np.array([1.5, 2.5, 3.5])
        y_pred = lr.predict(x_test)

        assert y_pred.shape == x_test.shape

    def test_fit_with_constant_x_raises(self):
        """fit() con x constante debe lanzar ValueError (Sxx=0)."""
        lr = LinearRegression()
        x = np.array([5, 5, 5, 5])
        y = np.array([1, 2, 3, 4])

        with pytest.raises(ValueError):
            lr.fit(x, y)

    def test_fit_with_mismatched_shapes_raises(self):
        """fit() con x e y de diferente tamaño debe lanzar ValueError."""
        lr = LinearRegression()
        x = np.array([1, 2, 3])
        y = np.array([1, 2])

        with pytest.raises(ValueError):
            lr.fit(x, y)

    def test_refit_updates_params(self):
        """fit() puede ser llamado nuevamente y actualiza los parámetros."""
        lr = LinearRegression()

        x1 = np.array([1, 2, 3])
        y1 = 2 * x1 + 1
        lr.fit(x1, y1)
        b1_first = lr.b1

        x2 = np.array([10, 20, 30])
        y2 = 0.5 * x2 + 5
        lr.fit(x2, y2)
        b1_second = lr.b1

        assert not np.isclose(b1_first, b1_second)
        assert np.isclose(lr.b1, 0.5, rtol=1e-5)
