"""Carga del dataset Airfoil Self-Noise (.dat sin header, tab-separated)."""
import pandas as pd

COLUMNS = ["frequency", "angle", "chord_length", "velocity", "displacement", "sound_level"]
URL = "https://raw.githubusercontent.com/dsgiitr/d2l-pytorch/master/data/airfoil_self_noise.dat"


def load_airfoil(path):
    """Carga el .dat, asigna nombres y valida forma 1503x6 y rangos básicos."""
    df = pd.read_csv(path, sep="\t", header=None, names=COLUMNS)
    assert df.shape == (1503, 6), f"forma inesperada: {df.shape}"
    assert df["sound_level"].between(103.4, 141.0).all(), "sound_level fuera de rango"
    return df
