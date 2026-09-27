import os
import pandas as pd
import urllib.request


COLUMNS = ["frequency", "angle", "chord_length", "velocity", "displacement", "sound_level"]
URL = "https://raw.githubusercontent.com/dsgiitr/d2l-pytorch/master/data/airfoil_self_noise.dat"


def load_airfoil(path: str) -> pd.DataFrame:
    """
    Load and validate the NASA airfoil self-noise dataset.

    Automatically downloads the dataset if it doesn't exist at the specified path.

    Parameters
    ----------
    path : str
        Path where the dataset should be loaded/saved (e.g., 'data/airfoil_self_noise.dat')

    Returns
    -------
    pd.DataFrame
        DataFrame with 1503 rows and 6 columns: frequency, angle, chord_length, velocity,
        displacement, sound_level

    Raises
    ------
    ValueError
        If dataset shape is not 1503 x 6, sound_level is outside [103.4, 141.0],
        or column order is incorrect
    """
    # Download if file doesn't exist
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(URL, path)

    # Load with pandas
    df = pd.read_csv(path, sep="\t", header=None, names=COLUMNS)

    # Validate shape
    if df.shape != (1503, 6):
        raise ValueError(
            f"Dataset shape must be (1503, 6), got {df.shape}"
        )

    # Validate sound_level range (with small tolerance for floating point precision)
    sound_level_min, sound_level_max = df["sound_level"].min(), df["sound_level"].max()
    tolerance = 0.1
    if not (103.4 - tolerance <= sound_level_min and sound_level_max <= 141.0 + tolerance):
        raise ValueError(
            f"sound_level must be approximately in range [103.4, 141.0], got min={sound_level_min}, max={sound_level_max}"
        )

    # Validate column order by checking column names
    if list(df.columns) != COLUMNS:
        raise ValueError(
            f"Column order mismatch. Expected {COLUMNS}, got {list(df.columns)}"
        )

    return df
