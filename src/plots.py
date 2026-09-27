"""Helpers de figuras: todos los ejes con unidades."""
import matplotlib.pyplot as plt

UNITS = {"frequency": "Hz", "angle": "grados", "chord_length": "m",
         "velocity": "m/s", "displacement": "m", "sound_level": "dB"}


def scatter_con_recta(x, y, b0, b1, xlabel):
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=8, alpha=0.4)
    xs = sorted(x)
    ax.plot(xs, [b0 + b1 * v for v in xs])
    ax.set_xlabel(f"{xlabel} ({UNITS.get(xlabel, '')})")
    ax.set_ylabel(f"sound_level ({UNITS['sound_level']})")
    return fig
