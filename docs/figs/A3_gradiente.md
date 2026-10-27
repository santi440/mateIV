# A.3 — Descenso por gradiente para el modelo múltiple (B.5)

Figura: `docs/figs/a3_convergencia.png`
Script: `python -m scripts.a3_gradiente`

## 1. Modelo y función objetivo

Se usa el mismo modelo de B.5:

$$
y = \beta_0 + \beta_1\,frequency + \beta_2\,angle
+ \beta_3\,chord\_length + \beta_4\,displacement + \varepsilon,
$$

con `velocity` excluida. En forma matricial, $y=X\beta+\varepsilon$, donde
$X$ incluye la columna de unos. Se minimiza

$$
f(\beta)=\frac{SCE(\beta)}{2n}
=\frac{\lVert y-X\beta\rVert^2}{2n}.
$$

El gradiente es

$$
\nabla f(\beta)=-\frac{X^\top(y-X\beta)}{n},
$$

y la actualización usada es

$$
\beta^{(k+1)}=\beta^{(k)}-\alpha\,\nabla f(\beta^{(k)}).
$$

## 2. Normalización obligatoria

Las features tienen escalas muy distintas: `frequency` se mide en Hz
($\sim10^3$–$10^4$) y `displacement` en m ($\sim10^{-3}$). Sin normalizar,
el gradiente queda dominado por `frequency` y el paso útil para una variable
hace divergir a la otra. Por eso se estandariza cada feature:

$$
x_{j,i}^{(s)}=\frac{x_{j,i}-\bar x_j}{s_j}.
$$

| Feature | Media | Desvío |
|---|---:|---:|
| frequency (Hz) | 2886.381 | 3151.524 |
| angle (grados) | 6.782302 | 5.916159 |
| chord_length (m) | 0.1365482 | 0.09350961 |
| displacement (m) | 0.01113988 | 0.01314586 |

El intercepto no se estandariza. Al final, los coeficientes se devuelven a
la escala original para compararlos con la solución exacta de B.5.

## 3. ¿Por qué restar el gradiente?

La derivada direccional de $f$ en la dirección $-\nabla f$ es

$$
\nabla f^\top(-\nabla f)=-\lVert\nabla f\rVert^2<0,
$$

así que un paso pequeño en esa dirección disminuye $f$. Por eso se usa
$\beta^{(k+1)}=\beta^{(k)}-\alpha\nabla f$.

## 4. Elección de los tres $\alpha$

Para una cuadrática, la convergencia de gradiente descendente exige
$0<\alpha<2/L$, con $L=\lambda_{\max}(X_s^\top X_s/n)$. Con los datos
normalizados:

$$
L=2.109383,\qquad \alpha_{\max}=\frac{2}{L}=0.948144.
$$

Se eligieron tres pasos relativos a $\alpha_{\max}$:

| Caso | $\alpha$ | Criterio |
|---|---:|---|
| chica | 0.047407 | $0.05\,\alpha_{\max}$ |
| ok | 0.474072 | $0.50\,\alpha_{\max}$ |
| diverge | 1.042959 | $1.10\,\alpha_{\max}$ |

El tercero supera $\alpha_{\max}$ y por eso no puede converger.

## 5. Criterio de detención

- Convergencia: $\lVert\nabla f(\beta^{(k)})\rVert<10^{-10}$.
- Tope de iteraciones: 6000.
- Divergencia: $f>10^{12}$ o coeficientes no finitos.

## 6. Resultados

Solución exacta de B.5:

$$
\beta_{\text{exacta}}=
\begin{pmatrix}
137.127552\ \text{dB}\\
-1.196133\times10^{-3}\ \text{dB/Hz}\\
-0.351565\ \text{dB/grado}\\
-33.887292\ \text{dB/m}\\
-164.044870\ \text{dB/m}
\end{pmatrix},
\qquad R^2=0.4669.
$$

Valor mínimo: $f_{\text{exacta}}=12.676226\ \text{dB}^2$.

| Caso | $\alpha$ | Iteraciones | Estado | $f$ final (dB²) | $\lVert\nabla f\rVert$ | $\lVert\beta-\beta_{\text{exacta}}\rVert$ |
|---|---:|---:|---|---:|---:|---:|
| chica | 0.047407 | 2561 | convergió | 12.6762 | $9.93\times10^{-11}$ | $2.465\times10^{-8}$ |
| ok | 0.474072 | 247 | convergió | 12.6762 | $9.96\times10^{-11}$ | $2.474\times10^{-8}$ |
| diverge | 1.042959 | 85 | divergió | $1.338842\times10^{12}$ | $2.38\times10^{6}$ | $5.986\times10^{7}$ |

## 7. Lectura

Con $\alpha$ chico el método converge, pero necesita 2561 iteraciones;
con $\alpha=0.474$ llega al mismo mínimo en 247 iteraciones. El caso
$\alpha=1.043$ supera el límite estable y $f$ crece hasta divergir. La
normalización es la que permite que un único $\alpha$ funcione para
variables medidas en Hz, grados y metros.

## 8. Reproducibilidad

```bash
python -m scripts.a3_gradiente
```

La implementación está en `src/gradiente.py`; la solución exacta se toma de
`src/multiple.py` (B.5), sin duplicar el cálculo matricial.
