# B.1 y B.4 — Varianza, R² e intervalos en x* = 2000 Hz

Modelo: regresión lineal simple de `sound_level` (dB) sobre `frequency` (Hz),
con la recta obtenida en A.2:

$$
\widehat{sound\_level}=127.3037\ \text{dB}-8.549790\times10^{-4}\ \frac{\text{dB}}{\text{Hz}}\cdot frequency
\qquad (n = 1503)
$$

## B.1 — Descomposición de la variabilidad

$$
STC=\sum_{i=1}^{n}(y_i-\bar y)^2,\qquad
SCR=\sum_{i=1}^{n}(\hat y_i-\bar y)^2,\qquad
SCE=\sum_{i=1}^{n}(y_i-\hat y_i)^2
$$

donde $y_i$ es el nivel de ruido medido en la fila $i$, $\bar y = 124.836$ dB
es el promedio de las 1503 mediciones e $\hat y_i$ es el valor que predice la
recta para la frecuencia de la fila $i$.

Las tres sumas se calcularon por separado, cada una con su fórmula:

| Fuente | Suma de cuadrados (dB²) |
|---|---:|
| Explicada por la regresión (SCR) | 10 912.1715 |
| No explicada, residuos (SCE) | 60 570.2062 |
| Total (STC) | 71 482.3777 |
| SCR + SCE | 71 482.3777 |
| STC − (SCR + SCE) | $1.46\times10^{-11}$ |

La diferencia es del orden del error de redondeo de la computadora, por lo que
se verifica numéricamente que **STC = SCR + SCE**.

### Varianza del modelo

$$
\hat\sigma^2=\frac{SCE}{n-2}=\frac{60\,570.2062}{1501}=40.3532\ \text{dB}^2
\qquad\Longrightarrow\qquad
\hat\sigma = 6.35\ \text{dB}
$$

Se divide por $n-2$ porque la recta usa dos parámetros estimados con los mismos
datos ($\hat\beta_0$ y $\hat\beta_1$). En promedio, las mediciones se alejan de
la recta unos 6.35 dB.

### Coeficiente de determinación

$$
R^2 = 1-\frac{SCE}{STC} = 1-\frac{60\,570.2062}{71\,482.3777} = 0.1527
$$

El 15.3 % de la variabilidad del nivel de ruido se explica por su relación
lineal con la frecuencia. El 84.7 % restante corresponde a las demás
condiciones del ensayo (ángulo, velocidad, cuerda, espesor de capa límite) y a
la variación propia de las mediciones.

**Qué NO significa R²:**

- No indica que la frecuencia *cause* el ruido: solo que ambas varían juntas.
- Un R² bajo no implica que no haya relación. La relación puede no ser lineal
  (ver el modelo cuadrático de A.1) o estar acompañada por otras variables
  (ver la regresión múltiple de B.5).
- No dice si la recta es el modelo adecuado; para eso hay que analizar los
  residuos.
- No es el porcentaje de predicciones acertadas ni la probabilidad de que el
  modelo sea correcto.
- No mide el error de una predicción individual: eso lo indican $\hat\sigma$ y
  el intervalo de predicción de B.4.

## B.4 — Intervalos al 95 % en x* = 2000 Hz

Con $t_{(0.025,\,1501)} = 1.96$ (el valor exacto es 1.9616 y solo cambia el
tercer decimal), $\bar x = 2886.38$ Hz y $S_{xx}=1.4928\times10^{10}$ Hz²:

$$
\hat y^* = 127.3037-8.549790\times10^{-4}\cdot 2000 = 125.59\ \text{dB}
$$

**Intervalo de confianza para la respuesta media**

$$
\hat y^* \pm t\,\sqrt{\hat\sigma^2\left(\frac1n+\frac{(x^*-\bar x)^2}{S_{xx}}\right)}
= 125.59 \pm 0.33\ \text{dB}
\;\Longrightarrow\; (125.26\ ;\ 125.93)\ \text{dB}
$$

**Intervalo de predicción para una medición futura**

$$
\hat y^* \pm t\,\sqrt{\hat\sigma^2\left(1+\frac1n+\frac{(x^*-\bar x)^2}{S_{xx}}\right)}
= 125.59 \pm 12.46\ \text{dB}
\;\Longrightarrow\; (113.14\ ;\ 138.05)\ \text{dB}
$$

### A qué se refiere cada intervalo

- **ICM:** con 95 % de confianza, el **nivel de ruido promedio** de todas las
  mediciones posibles a 2000 Hz está entre 125.26 y 125.93 dB. Como control,
  las 105 mediciones del dataset a 2000 Hz promedian 125.71 dB, dentro del
  intervalo.
- **IP:** con 95 % de confianza, **una medición nueva** a 2000 Hz va a dar entre
  113.14 y 138.05 dB.

### Por qué el IP es mucho más ancho

El IP es unas 37 veces más ancho que el ICM. El ICM solo tiene en cuenta la
incertidumbre de estimar la recta, que es muy chica porque hay 1503 datos y
2000 Hz está cerca de $\bar x$. El IP suma además la variación propia de cada
medición alrededor de la recta: es el término **1 +** dentro de la raíz, que
pesa $\hat\sigma^2$. Esa variación no disminuye aunque se agreguen más datos,
por eso el IP sigue siendo ancho.

## Reproducibilidad

```bash
python -m scripts.b1_b4_varianza
```

El cálculo vive en `src/varianza.py`, que usa las funciones de `src/metrics.py`
y la recta de `src/normales.py`.