# A.1 — Ajuste cuadrático de `sound_level` en función de `frequency`

Figura: `docs/figs/a1_aproximacion.png`
Script: `python -m scripts.a1_cuadratico`

## 1. Modelo y ajuste

Se propone

$$
\widehat{sound\_level}(x)=b_0+b_1x+b_2x^2,
$$

con $x=$ `frequency` (Hz) e $y=$ `sound_level` (dB). Los coeficientes se
estiman por mínimos cuadrados minimizando

$$
SCE=\sum_i\left(y_i-b_0-b_1x_i-b_2x_i^2\right)^2.
$$

Coeficientes obtenidos:

| Coeficiente | Valor | Unidad |
|---|---:|---|
| $b_0$ | 128.3104 | dB |
| $b_1$ | $-1.561198\times10^{-3}$ | dB/Hz |
| $b_2$ | $5.649251\times10^{-8}$ | dB/Hz² |

La recta de comparación (A.2) es

$$
\widehat{sound\_level}(x)=127.3037\ \text{dB}
-8.549790\times10^{-4}\ \frac{\text{dB}}{\text{Hz}}\,x.
$$

## 2. Derivada

$$
\frac{d\widehat{y}}{dx}=b_1+2b_2x.
$$

| $x$ (Hz) | $d\widehat{y}/dx$ (dB/Hz) | Equivalente cada 100 Hz |
|---:|---:|---:|
| 1000 | $-1.448213\times10^{-3}$ | $-0.145$ dB |
| 10000 | $-4.313479\times10^{-4}$ | $-0.043$ dB |

Interpretación: a bajas frecuencias el ruido cae más rápido al aumentar la
frecuencia; cerca de 10 kHz la pendiente ya es menos de un tercio de la de
1 kHz, es decir, la curva se está aplanando.

## 3. Punto crítico

Se resuelve $d\widehat{y}/dx=0$:

$$
x^*=-\frac{b_1}{2b_2}=13817.7\ \text{Hz},
\qquad
\widehat{y}(x^*)=117.52\ \text{dB}.
$$

La segunda derivada es

$$
\frac{d^2\widehat{y}}{dx^2}=2b_2=1.130\times10^{-7}>0,
$$

por lo que el punto crítico es un **mínimo**. El valor $x^*=13817.7$ Hz cae
dentro del rango observado $[200,20000]$ Hz. De todos modos, la zona de
12–20 kHz tiene pocas mediciones, así que el mínimo debe leerse como una
tendencia interpolada y no como una extrapolación fuerte.

## 4. Comparación lineal vs cuadrático

| Modelo | $R^2$ | SCE (dB²) |
|---|---:|---:|
| Lineal | 0.1527 | 60570.2 |
| Cuadrático | 0.1734 | 59090.2 |

El cuadrático reduce la SCE y sube el $R^2$, pero la mejora es chica: la
frecuencia sola explica alrededor del 15–17 % de la variabilidad del ruido.
La curvatura captura el aplanamiento (y eventual mínimo) en frecuencias
altas, aunque la dispersión de los datos sigue siendo grande. Eso motiva
usar más variables en B.5.

## 5. Reproducibilidad

```bash
python -m scripts.a1_cuadratico
```

El ajuste, la derivada, el punto crítico y la comparación viven en
`src/cuadratico.py`; la bondad de ajuste usa `src/metrics.py`.
