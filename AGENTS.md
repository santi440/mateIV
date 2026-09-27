# AGENTS.md — TP Mate IV · Conjunto C · Ruido aerodinámico (Esquema T)

## Contexto
Regresión de `sound_level` (dB, 103.4–141.0) sobre datos UCI Airfoil Self-Noise (1503×6).
Columnas del `.dat` (sin header, tab-separated, en este orden):
`frequency, angle, chord_length, velocity, displacement, sound_level`.
Fuente: https://raw.githubusercontent.com/dsgiitr/d2l-pytorch/master/data/airfoil_self_noise.dat
Entrega: PDF ≤20 pág. sin anexos + link Colab con código trazable + Anexo I bitácora IA obligatoria.

## Stack fijo
`numpy + pandas + matplotlib` únicamente. Sin sklearn/scipy (A.2/A.3 exigen cálculo manual y contraste con herramienta base como `polyfit`).

## Esquema T — 4 bloques transversales (sin asignar)
- **T1 núcleo frequency-simple:** A.1, A.2, B.1, B.4. Publica la interfaz numérica.
- **T2 comparativa + inferencia:** B.2 (4 rectas), B.3 test H₀, scatter mejor variable.
- **T3 multivariado + optimización:** B.5 matricial, A.3 gradiente 3 alphas.
- **T0 integración e informe:** loader, EDA, tests, `informe.ipynb` → PDF, Colab, Parte C/Anexo I, conclusiones.

Participación cruzada obligatoria: cada PR lleva 1 reviewer de otro bloque. T0 edita la sección de informe de cada eje. Matriz sugerida: T1↔T2, T3↔T1, T0↔todos.

## Contratos (congelar primero, issue t0-2)
- `src/data_loader.py: load_airfoil(path) -> DataFrame` con columnas nombradas exactas. Valida 1503×6 y rangos.
- `src/metrics.py`: única implementación de `Sxx, Sxy, xbar, ybar, beta1, beta0, sce, stc, scr, sigma2, r2, ic_beta, icm_ip`. T1/T2/T3 la importan, NO la duplican.
- Normalización obligatoria en A.3 (`frequency`~1e4 vs `displacement`~1e-3) o el gradiente diverge por escala, no por alpha.
- `velocity` (4 valores discretos) solo EDA: la consigna la excluye de B.5 y de la tabla B.2.

## Workflow git
- Rama por issue: `t0/...`, `t1/...`, `t2/...`, `t3/...`. Un PR = una issue. No push directo a `main`.
- Commits en español, cortos. PR debe listar números/tablas/figuras agregadas + interpretación en lenguaje del dominio (dB, Hz, m, m/s, grados).
- Definición de hecho por issue: número reproducible desde `src/` + figura/tabla + 2–3 líneas de interpretación + fila en bitácora IA si usó asistente.

## Reglas de código e informe
- Nunca commitear el `.dat`, `__pycache__`, `.ipynb_checkpoints` (ver `.gitignore`).
- `notebooks/informe.ipynb` solo narrativa + llamadas a `src/`; nada de constantes hardcodeadas.
- Figuras con unidades en ejes. `x* = 2000 Hz` fijo para B.4. α = 0.05, IC 95%.
- Todo número del PDF debe poder rastrearse a una función de `src/` (requisito cátedra: lo no reproducible no se califica).
- Parte C: cada interacción IA incorporada → fila en Anexo I (pedido, resumen respuesta, aceptada/cambios/rechazo+motivo, ítem afectado). Si no se usó IA, declararlo por escrito.
