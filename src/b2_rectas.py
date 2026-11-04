"""B.2: Tabla de las 4 rectas simples (excluyendo velocity)."""
import numpy as np
from src import metrics as m
from src.data_loader import load_airfoil

VARIABLES_B2 = [
    ("frequency", "Hz", "dB/Hz"),
    ("angle", "grados", "dB/grados"),
    ("chord_length", "m", "dB/m"),
    ("displacement", "m", "dB/m"),
]

def ajustar_rectas_simples(df):
    """Calcula beta0, beta1, SCE, R2 y r para cada variable permitida."""
    y = df["sound_level"].values
    n = len(df)
    resultados = []
    
    for var, unidad_x, unidad_b1 in VARIABLES_B2:
        x = df[var].values
        xb = m.xbar(x)
        yb = m.ybar(y)
        sxx = m.Sxx(x)
        sxy = m.Sxy(x, y)
        b1 = m.beta1(sxy, sxx)
        b0 = m.beta0(yb, b1, xb)
        sce_v = m.sce(x, y, b0, b1)
        stc_v = m.stc(y)
        r2_v = m.r2(sce_v, stc_v)
        syy = ((y - yb) ** 2).sum()
        r_corr = m.r_corr(sxy, sxx, syy)
        
        resultados.append({
            "variable": var,
            "unidad_x": unidad_x,
            "unidad_b1": unidad_b1,
            "beta0": b0,
            "beta1": b1,
            "sce": sce_v,
            "r2": r2_v,
            "r": r_corr,
        })
    return resultados
