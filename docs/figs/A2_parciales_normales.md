# A.2 — Derivadas parciales, sistema normal y recta de mínimos cuadrados

## 1. Función a minimizar

Para la variable elegida `x = frequency` (Hz) y la respuesta
`y = sound_level` (dB), la suma de cuadrados del error es

$$
SCE(b_0,b_1)=\sum_{i=1}^{n}\left(y_i-b_0-b_1x_i\right)^2 .
$$

## 2. Derivadas parciales

$$
\frac{\partial SCE}{\partial b_0}
=-2\sum_{i=1}^{n}\left(y_i-b_0-b_1x_i\right),
\qquad
\frac{\partial SCE}{\partial b_1}
=-2\sum_{i=1}^{n}x_i\left(y_i-b_0-b_1x_i\right).
$$

## 3. Sistema de ecuaciones normales

Igualando ambas derivadas a cero y dividiendo por $-2$:

$$
\sum_{i=1}^{n}\left(y_i-b_0-b_1x_i\right)=0
\;\Longrightarrow\;
n\,b_0+\left(\sum_i x_i\right)b_1=\sum_i y_i,
$$

$$
\sum_{i=1}^{n}x_i\left(y_i-b_0-b_1x_i\right)=0
\;\Longrightarrow\;
\left(\sum_i x_i\right)b_0+\left(\sum_i x_i^2\right)b_1=\sum_i x_i y_i.
$$

En forma matricial:

$$
\begin{pmatrix}
n & \sum x_i\\
\sum x_i & \sum x_i^2
\end{pmatrix}
\begin{pmatrix}
b_0\\ b_1
\end{pmatrix}
=
\begin{pmatrix}
\sum y_i\\ \sum x_i y_i
\end{pmatrix}.
$$

## 4. Solución cerrada

Resolviendo el sistema de $2\times2$:

$$
b_1=\frac{n\sum x_i y_i-\left(\sum x_i\right)\left(\sum y_i\right)}
{n\sum x_i^2-\left(\sum x_i\right)^2}
=\frac{S_{xy}}{S_{xx}},
$$

$$
b_0=\frac{\sum y_i-b_1\sum x_i}{n}
=\bar y-b_1\bar x.
$$

## 5. Resultado numérico

Con las 1503 mediciones:

| Suma | Valor |
|---|---:|
| $n$ | 1503 |
| $\sum x_i$ | $4.338230\times10^{6}$ Hz |
| $\sum y_i$ | $1.876284\times10^{5}$ dB |
| $\sum x_i^2$ | $2.744974\times10^{10}$ Hz² |
| $\sum x_i y_i$ | $5.288039\times10^{8}$ Hz·dB |

La recta estimada es

$$
\widehat{sound\_level}=127.303738\ \text{dB}
-8.549790\times10^{-4}\ \frac{\text{dB}}{\text{Hz}}\cdot frequency.
$$

Interpretación: por cada 1000 Hz adicionales, el nivel de ruido estimado
baja aproximadamente 0.855 dB en este ajuste lineal simple.

## 6. Contraste con `polyfit`

| Método | $b_0$ (dB) | $b_1$ (dB/Hz) |
|---|---:|---:|
| Sistema normal | 127.303738 | $-8.549790\times10^{-4}$ |
| Fórmulas cerradas (`src.metrics`) | 127.303738 | $-8.549790\times10^{-4}$ |
| `numpy.polyfit` | 127.303738 | $-8.549790\times10^{-4}$ |

Diferencias absolutas:

- sistema normal vs fórmulas cerradas: $(1.42\times10^{-14},\ 1.08\times10^{-18})$
- sistema normal vs `polyfit`: $(4.26\times10^{-14},\ 4.23\times10^{-18})$
- gradiente en el óptimo: $(1.23\times10^{-11},\ 9.31\times10^{-9})\approx(0,0)$

Las diferencias son de orden numérico, por lo que los tres caminos
coinciden.

## 7. Reproducibilidad

```bash
python -m scripts.a2_normales
```

El cálculo vive en `src/normales.py` y usa las funciones congeladas de
`src/metrics.py`; no se duplican fórmulas del contrato.
