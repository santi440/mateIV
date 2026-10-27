"""DoD issue #3: discretas documentadas y rango de displacement."""
import numpy as np

from src.data_loader import load_airfoil
from src.eda import resumen_discretas


def test_discretas_del_dataset():
    df = load_airfoil("data/airfoil_self_noise.dat")
    r = resumen_discretas(df)

    assert r["frequency"]["n_valores"] == 21
    assert r["chord_length"]["n_valores"] == 6
    assert r["velocity"]["n_valores"] == 4
    assert r["velocity"]["valores"] == [31.7, 39.6, 55.5, 71.3]
    assert r["velocity"]["conteos"] == {31.7: 281, 39.6: 480, 55.5: 277, 71.3: 465}

    assert np.isclose(r["displacement"]["min"], 4.00682e-4, rtol=1e-5)
    assert np.isclose(r["displacement"]["max"], 5.84113e-2, rtol=1e-5)
