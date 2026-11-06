# B.5 — Regresión lineal múltiple (forma matricial)

## 1. Objetivo
Explicar el nivel de ruido aerodinámico (`sound_level`, en dB) a partir de cuatro variables a la vez, en lugar de una sola como en la regresión simple (B.2):

| Variable | Significado | Unidad |
| $x_1$ = frequency | Frecuencia | Hz |
| $x_2$ = angle | Ángulo de ataque | grados |
| $x_3$ = chord_length | Longitud de cuerda del perfil | m |
| $x_4$ = displacement | Espesor de desplazamiento | m |


El modelo que buscamos es:

$$\hat{y} = \beta_0 + \beta_1 \cdot \text{frecuencia} + \beta_2 \cdot \text{ángulo} + \beta_3 \cdot \text{cuerda} + \beta_4 \cdot \text{espesor}$$

## ¿Cómo calculamos los β?
Armamos una matriz $X$ con los datos (una fila por medición y una columna por variable, más una columna de unos para $\beta_0$) y un vector $Y$ con los niveles de ruido.

Los coeficientes salen de la fórmula de mínimos cuadrados:
$$\hat\beta = (X^T X)^{-1} X^T Y$$

## Resultado
$$\hat{y} = 137.13 - 0.0012 \cdot \text{frecuencia} - 0.35 \cdot \text{ángulo} - 33.89 \cdot \text{cuerda} - 164.04 \cdot \text{espesor}$$

Todos los coeficientes son negativos: si cualquiera de las variables aumenta (y las demás quedan iguales), el ruido baja. Por ejemplo:

- 1000 Hz más de frecuencia → el ruido baja unos 1.2 dB
- 1 grado más de ángulo → baja unos 0.35 dB
- 10 cm más de cuerda → baja unos 3.4 dB
- 1 cm más de espesor → baja unos 1.6 dB

## ¿Qué tan bueno es el modelo?
Usamos el $R^2$ ajustado, que mide qué porcentaje del ruido explica el modelo y penaliza usar muchas variables:

$$R^2_{aj} = 1 - (1 - R^2)\,\frac{n - k}{n - k - 1} \qquad (n = 1503,\ k = 4)$$

| Modelo | $R^2$ ajustado |
|---|---|
| Mejor recta simple (solo frecuencia) | 0.152 |
| Regresión múltiple (4 variables) | 0.467 |

## Conclusión
Con las cuatro variables juntas el modelo explica cerca del 47 % del ruido, unas tres veces más que la mejor recta simple (15 %). Por eso conviene usar el modelo múltiple.

Igual, más de la mitad del ruido queda sin explicar. Eso indica que hay otros factores (por ejemplo la velocidad, o relaciones que no son rectas) que este modelo no captura.


## 6. Reproducibilidad
Todos los valores se calculan con `src/multiple.py` (y el $R^2$ simple con `src/metrics.py`). Para obtenerlos:

```
python -m src.multiple
```