# TP Mate IV · Conjunto C · Ruido aerodinámico — Esquema T

Repo modular: el cálculo vive en `src/`, el informe en `notebooks/informe.ipynb` (subir a Colab para entrega).

## Estructura
- `src/data_loader.py` — carga y validación del `.dat`
- `src/metrics.py` — única implementación de Sxx, Sxy, betas, SCE/STC/SCR, σ², R², IC/IP, t-test
- `src/eda.py` — descriptivos, variables discretas e histogramas con unidades
- `src/normales.py` — A.2: derivadas parciales, sistema normal y contraste con `polyfit`
- `src/cuadratico.py` — A.1: ajuste cuadrático, derivada y punto crítico
- `src/multiple.py` — B.5: regresión múltiple matricial y R² ajustado
- `src/gradiente.py` — A.3: descenso por gradiente con features estandarizadas
- `src/normales.py` — A.2: derivadas parciales, sistema normal y contraste con `polyfit`
- `src/plots.py` — helpers de figuras con unidades
- `notebooks/informe.ipynb` — informe preliminar ejecutable (EDA, A.1, A.2 y B.5)
- `tests/test_repro.py` — chequeos de reproducibilidad

## Entorno virtual (venv) + requirements
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Uso (con el venv activado)
```bash
pip install -r requirements.txt
python -c "from src.data_loader import load_airfoil; load_airfoil('data/airfoil_self_noise.dat')"
python -m scripts.eda
python -m scripts.a2_normales
python -m scripts.a1_cuadratico
python -m scripts.b5_multiple
python -m scripts.a3_gradiente
python -m scripts.a2_normales
```

Datos: descargar sin commitear:
```bash
mkdir -p data && curl -L https://raw.githubusercontent.com/dsgiitr/d2l-pytorch/master/data/airfoil_self_noise.dat -o data/airfoil_self_noise.dat
```

## Tests
```bash
pip install pytest
python -m pytest -q
```

`tests/test_repro.py` verifica `STC = SCR + SCE`, beta manual vs `polyfit`
y gradiente de SCE nulo en el óptimo. El contraste completo de A.3
(descenso por gradiente vs solución exacta) se agrega con #11.

## Estado de issues (bloqueo nativo de GitHub)

Las dependencias están cableadas con el bloqueo nativo (`blocked by`):
GitHub muestra y desbloquea cada issue solo cuando cierran aquellas de las
que depende. Listado rápido:

```bash
python scripts/issue_status.py
```

## Workflow (ver AGENTS.md)
Ramas `t0/... t1/... t2/... t3/...`, un PR = una issue, reviewer cruzado obligatorio.
