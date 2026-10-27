# EDA — Variables discretas y distribuciones

Figura generada por `python -m scripts.eda`:
`docs/figs/eda_histogramas.png`.

## 1. Descriptivos (n = 1503)

| Variable | Unidad | Media | Desvío | Mín. | Mediana | Máx. |
|---|---|---:|---:|---:|---:|---:|
| frequency | Hz | 2886.38 | 3152.57 | 200 | 1600 | 20000 |
| angle | grados | 6.7823 | 5.91813 | 0.0 | 5.4 | 22.2 |
| chord_length | m | 0.136548 | 0.0935407 | 0.0254 | 0.1016 | 0.3048 |
| velocity | m/s | 50.8607 | 15.5728 | 31.7 | 39.6 | 71.3 |
| displacement | m | 0.0111399 | 0.0131502 | 0.000400682 | 0.00495741 | 0.0584113 |
| sound_level | dB | 124.836 | 6.89866 | 103.38 | 125.721 | 140.987 |

## 2. Variables con pocos valores distintos

- **frequency**: 21 valores, de 200 a 20000 Hz, espaciados aproximadamente
  en forma logarítmica:
  200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150,
  4000, 5000, 6300, 8000, 10000, 12500, 16000 y 20000 Hz.
- **angle**: 27 valores entre 0.0 y 22.2 grados.
- **chord_length**: 6 valores: 0.0254, 0.0508, 0.1016, 0.1524, 0.2286 y
  0.3048 m (2.54 a 30.48 cm).
- **velocity**: 4 valores: 31.7, 39.6, 55.5 y 71.3 m/s. Conteos:
  281, 480, 277 y 465 mediciones, respectivamente.
- **displacement**: 105 valores distintos, con rango
  $[4.00682\times10^{-4},\ 5.84113\times10^{-2}]$ m.

> **Nota:** `velocity` se describe únicamente en el EDA. La consigna la
> excluye de la tabla de rectas simples (B.2) y de la regresión múltiple
> (B.5).

## 3. Lectura de los histogramas

- **frequency (Hz):** hay muchas más mediciones en frecuencias bajas y los
  conteos bajan al aumentar la frecuencia. La cola de 10–20 kHz es escasa,
  así que las conclusiones en esa zona tienen menos respaldo empírico.
- **angle (grados):** la mayor densidad está entre 0 y 5 grados, con pocas
  mediciones por encima de 15 grados. Esos ángulos altos pueden tener mayor
  influencia puntual en cualquier ajuste.
- **chord_length (m):** las seis cuerdas están razonablemente cubiertas,
  aunque 0.3048 m tiene menos casos. El rango va de perfiles de 2.54 cm a
  30.48 cm.
- **velocity (m/s):** las cuatro velocidades están presentes, con 39.6 y
  71.3 m/s como las más muestreadas. Se incluye solo como caracterización
  del experimento, no como regresor del TP.
- **displacement (m):** distribución fuertemente asimétrica a la derecha:
  la mayoría de las mediciones queda por debajo de 0.01 m y hay una cola
  larga hasta 0.058 m. Su escala es muy distinta de la de frequency.
- **sound_level (dB):** se concentra aproximadamente entre 120 y 130 dB,
  con media 124.84 dB y mediana 125.72 dB. El rango observado es
  103.38–140.99 dB, consistente con ruido aerodinámico medido en dB.

## 4. Reproducibilidad

```bash
python -m scripts.eda
```

El cálculo y la figura viven en `src/eda.py`; el script solo carga los
datos y presenta los resultados.
