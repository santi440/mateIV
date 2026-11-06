# B.2 y B.3 — Regresiones simples y test de la pendiente

Figura: `docs/figs/b2_dispersion_frequency.png`
Script: `python -m scripts.b2_regresionLineal`

## 1. B.2 — Las cuatro regresiones simples

Para cada variable se ajusta $\widehat{sound\_level}=\hat\beta_0+\hat\beta_1x$
por mínimos cuadrados ($n=1503$, intervalos al 95 %, $t(0.025,\,1501)=1.96$).

| $X_i$ | $\hat y=\hat\beta_1x+\hat\beta_0$ | $\hat\sigma^2$ (dB²) | $R^2$ | $r$ | IC($\beta_1$) | IC($\beta_0$) (dB) | ICM($Y\mid x^*$) (dB) | IP($Y\mid x^*$) (dB) |
|---|---|---:|---:|---:|---|---|---|---|
| frequency | $-0.000855\,x+127.30$ | 40.35 | 0.153 | −0.391 | [−0.000957, −0.000753] dB/Hz | [126.87, 127.74] | [125.26, 125.93] | [113.14, 138.05] |
| angle | $-0.182\,x+126.07$ | 46.46 | 0.024 | −0.156 | [−0.240, −0.124] dB/grado | [125.55, 126.59] | [124.49, 125.18] | [111.47, 138.20] |
| chord_length | $-17.42\,x+127.21$ | 44.97 | 0.056 | −0.236 | [−21.04, −13.79] dB/m | [126.61, 127.81] | [124.50, 125.18] | [111.69, 137.98] |
| displacement | $-164.03\,x+126.66$ | 42.97 | 0.098 | −0.313 | [−189.24, −138.82] dB/m | [126.23, 127.10] | [124.51, 125.17] | [111.98, 137.69] |

Para `frequency` se usa $x^*=2000$ Hz (el valor de B.4). Ese valor no tiene
sentido para las otras variables, así que para ellas se toma $x^*=\bar x$ de
cada una (angle 6.78°, chord_length 0.137 m, displacement 0.0111 m).

### Variable elegida: `frequency`

Es la que mejor responde a `sound_level` según los tres criterios a la vez:

- mayor $R^2$: explica el 15.3 % de la variabilidad del ruido (la siguiente,
  displacement, explica el 9.8 %);
- menor $\hat\sigma^2$: el error típico alrededor de la recta es el menor,
  $\hat\sigma\approx6.35$ dB;
- mayor $|r|$ y mayor $|T|$ (ver B.3).

La figura muestra el diagrama de dispersión de `frequency` contra
`sound_level` con la recta ajustada. Los puntos forman 21 columnas porque la
frecuencia solo toma 21 valores (bandas de tercio de octava). La nube es muy
ancha alrededor de la recta, coherente con un $R^2$ bajo: la frecuencia sola
explica una parte chica del ruido.

## 2. B.3 — Test de hipótesis sobre la pendiente

Sobre el modelo de `frequency`:

$$
H_0:\ \beta_1=0 \qquad\qquad H_1:\ \beta_1\neq0
$$

$H_0$ dice que la frecuencia no tiene relación lineal con el nivel de ruido.

**Estadístico de prueba.** Bajo $H_0$ sigue una $t$ de Student con
$n-2=1501$ grados de libertad:

$$
T=\frac{\hat\beta_1}{\sqrt{\hat\sigma^2/S_{xx}}}
=\frac{-8.5498\times10^{-4}}{\sqrt{40.353/1.4928\times10^{10}}}
=\frac{-8.5498\times10^{-4}}{5.199\times10^{-5}}
=-16.44 .
$$

**Valor crítico.** Para $\alpha=0.05$ se busca $t(0.025,\,1501)$. La tabla no
tiene una fila para 1501 grados de libertad; se usa la fila $\infty$, que da
$1.96$ (el valor exacto es 1.9615, la diferencia no cambia la decisión).

**Decisión.**

$$
|T|=16.44>1.96 \;\Longrightarrow\; \text{se rechaza } H_0 .
$$

**Conclusión.** Con un 95 % de confianza, la frecuencia del sonido está
relacionada con el nivel de ruido propio del perfil: a mayor frecuencia, menor
nivel sonoro, aproximadamente 0.86 dB menos cada 1000 Hz. La pendiente
observada no se explica por variación aleatoria de las mediciones.

**Qué no dice el test.** Rechazar $H_0$ indica que la relación existe, no que
sea fuerte ni que alcance para predecir: el modelo explica solo el 15 % de la
variabilidad. Con $n=1503$ incluso relaciones débiles resultan
significativas; las cuatro variables de B.2 rechazan $H_0$ (la más débil,
angle, tiene $T=-6.12$).

## 3. Reproducibilidad

```bash
python -m scripts.b2_regresionLineal
```

Las fórmulas ($\hat\beta$, $\hat\sigma^2$, $R^2$, $T$, intervalos y valor
crítico) vienen de `src/metrics.py`; el script imprime la tabla, la decisión
del test y genera la figura.
