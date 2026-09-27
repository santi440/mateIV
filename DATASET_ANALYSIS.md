# Análisis del Dataset: NASA Airfoil Self-Noise

## Resumen Ejecutivo

Dataset de ruido aerodinámico en perfiles alares de la NASA. Contiene 1503 muestras de 6 variables medidas durante ensayos experimentales de túnel de viento.

**Fuente**: https://raw.githubusercontent.com/dsgiitr/d2l-pytorch/master/data/airfoil_self_noise.dat

**Dimensiones**: 1503 filas × 6 columnas

---

## Estadísticos Descriptivos

| Variable | Count | Mean | Std Dev | Min | 25% | 50% (Median) | 75% | Max |
|----------|-------|------|---------|-----|-----|--------------|-----|-----|
| **frequency** | 1503 | 2886.38 | 3152.57 | 200.00 | 800.00 | 1600.00 | 4000.00 | 20000.00 |
| **angle** | 1503 | 6.78 | 5.92 | 0.00 | 2.00 | 5.40 | 9.90 | 22.20 |
| **chord_length** | 1503 | 0.14 | 0.09 | 0.03 | 0.08 | 0.10 | 0.15 | 0.30 |
| **velocity** | 1503 | 50.86 | 15.57 | 31.70 | 39.60 | 39.60 | 71.30 | 71.30 |
| **displacement** | 1503 | 0.011 | 0.013 | 0.0004 | 0.0025 | 0.0050 | 0.0156 | 0.0584 |
| **sound_level** | 1503 | 124.84 | 6.90 | 103.38 | 120.19 | 125.72 | 129.99 | 140.99 |

---

## Interpretación de Variables (Unidades del Dominio)

### 1. **frequency** (Hz)
- **Definición**: Número de Strouhal en frecuencia - representa la frecuencia no-dimensional de la interacción aerodinámica
- **Rango observado**: 200 - 20,000 Hz
- **Interpretación física**: 
  - Rango típico de frecuencias audibles humanas (20 Hz - 20,000 Hz)
  - Frecuencias altas correlacionan con ruido de mayor amplitud
  - Los datos cubren el espectro completo de ruido aerodinámico
- **Unidad**: Hertzios (ciclos por segundo)

### 2. **angle** (grados)
- **Definición**: Ángulo de ataque (Attack angle) - ángulo entre la cuerda del perfil y la dirección del flujo
- **Rango observado**: 0° - 22.2°
- **Interpretación física**:
  - 0° = perfil alineado con el flujo (mínimo ruido)
  - 0° - 12° = régimen de sustentación normal
  - 12° - 22.2° = acercamiento al ángulo de pérdida de sustentación (stall angle)
  - Mayor ángulo = generalmente mayor ruido aerodinámico
- **Unidad**: Grados sexagesimales

### 3. **chord_length** (metros)
- **Definición**: Longitud de la cuerda - distancia horizontal del borde de ataque al borde de fuga del perfil alar
- **Rango observado**: 0.03 m - 0.30 m (3 cm - 30 cm)
- **Interpretación física**:
  - Perfiles de pequeño tamaño (0.03 m ≈ pequeñas turbinas o drones)
  - Perfiles medianos (0.30 m ≈ hélices de aviones pequeños)
  - Mayor cuerda = mayor generación de ruido absoluto (pero mejor relación sustentación/ruido)
  - Correlación negativa con ángulo de ataque en los datos
- **Unidad**: Metros

### 4. **velocity** (m/s)
- **Definición**: Velocidad de la corriente libre (freestream velocity) - velocidad del fluido no perturbado
- **Rango observado**: 31.7 m/s - 71.3 m/s (114 - 257 km/h)
- **Interpretación física**:
  - 31.7 m/s ≈ velocidad de viento moderado (Escala Beaufort: tormenta fuerte)
  - 71.3 m/s ≈ velocidad de huracán
  - El ruido aerodinámico es muy sensible a velocidad (∝ V^5 a V^6 en algunos regímenes)
  - Discreto en el dataset: principalmente dos valores (31.7 y 71.3)
- **Unidad**: Metros por segundo

### 5. **displacement** (metros)
- **Definición**: Espesor de desplazamiento (Displacement thickness) - medida de la separación de la capa límite
- **Rango observado**: 0.0004 m - 0.0584 m (0.4 mm - 58.4 mm)
- **Interpretación física**:
  - Caracteriza cuánto el flujo se desvía por la presencia de la capa límite
  - Bajo desplazamiento = capa límite adherida (flujo laminar ideal)
  - Alto desplazamiento = capa límite separada, flujo turbulento
  - Correlación fuerte con ángulo de ataque (0.753)
  - Predictor importante del ruido radiado
- **Unidad**: Metros

### 6. **sound_level** (dB - Decibelios)
- **Definición**: Nivel de presión sonora - medida logarítmica de la intensidad acústica
- **Rango observado**: 103.38 dB - 140.99 dB
- **Interpretación física**:
  - **103.4 dB**: Ruido de perforación neumática a 1 metro
  - **124.8 dB**: Media del dataset ≈ sirena de ambulancia
  - **141 dB**: Umbral del dolor auditivo (≥140 dB causa daño inmediato)
  - Escala logarítmica: cada +10 dB = duplicación perceptual del volumen
  - Generado por turbulencia, presión y suction-side flow separation
- **Unidad**: Decibelios (dB re 20 μPa)
- **Variable objetivo**: Es la variable que se predice en problemas de regresión

---

## Análisis de Correlaciones

### Correlaciones con Sound Level:

| Variable | Correlación | Interpretación |
|----------|-------------|-----------------|
| **frequency** | -0.391 | Moderada negativa: frecuencias altas → ruido más bajo |
| **angle** | -0.156 | Débil negativa: ángulos mayores → ruido levemente menor |
| **chord_length** | -0.236 | Débil negativa: cuerdas más largas → ruido menor |
| **velocity** | +0.125 | Débil positiva: velocidades altas → ruido ligeramente mayor |
| **displacement** | -0.313 | Moderada negativa: desplazamiento mayor → ruido menor |

### Hallazgos:

1. **Frecuencia** es el predictor más fuerte del ruido (-0.39)
2. **Desplazamiento** muestra correlación fuerte con ángulo (0.75), indicando que cambian juntos
3. **Velocidad** tiene efecto débil, posiblemente porque es discreta en el dataset
4. **Relación no-lineal**: Las correlaciones son moderadas, sugiriendo relaciones complejas

---

## Validaciones Implementadas

✓ Forma del dataset: exactamente 1503 × 6
✓ Rango sound_level: entre 103.38 y 140.99 dB (dentro de tolerancia de 103.4 - 141.0)
✓ Orden de columnas: frequency, angle, chord_length, velocity, displacement, sound_level
✓ Sin valores faltantes (NaN)
✓ Descarga automática desde repositorio NASA si no existe

---

## Notas sobre Aplicaciones

Este dataset es comúnmente usado para:

1. **Predicción de ruido aerodinámico**: Regresión ML (random forest, neural networks)
2. **Investigación de acústica aerodinámica**: Análisis de mecanismos de generación de ruido
3. **Optimización de diseño de perfiles**: Minimizar ruido manteniendo sustentación
4. **Problemas de aprendizaje automático benchmark**: Validar métodos de ML
5. **Simulación y análisis CFD**: Correlacionar con datos experimentales

---

## Referencias

- Dataset original: https://archive.ics.uci.edu/ml/datasets/airfoil+self-noise
- Paper: "Airfoil Self-Noise Prediction Using Supervised Learning" 
- Técnica de medición: Túnel de viento anecóico (anechoic wind tunnel)
