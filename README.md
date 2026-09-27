# TP Mate IV · Conjunto C · Ruido aerodinámico — Esquema T

Repo modular: el cálculo vive en `src/`, el informe en `notebooks/informe.ipynb` (subir a Colab para entrega).

## Estructura
- `src/data_loader.py` — carga y validación del `.dat`
- `src/metrics.py` — única implementación de Sxx, Sxy, betas, SCE/STC/SCR, σ², R², IC/IP, t-test
- `src/plots.py` — helpers de figuras con unidades
- `notebooks/informe.ipynb` — narrativa + llamadas a `src/`
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
```

Datos: descargar sin commitear:
```bash
mkdir -p data && curl -L https://raw.githubusercontent.com/dsgiitr/d2l-pytorch/master/data/airfoil_self_noise.dat -o data/airfoil_self_noise.dat
```

## Estado de issues (derivado, sin etiquetas)

GitHub solo distingue OPEN/CLOSED. El bloqueo se deriva del estado de las
dependencias (ver `## Dependencias` en cada issue):

```bash
python scripts/issue_status.py
```

Una issue sale LISTA cuando todas sus dependencias están CLOSED; si no,
BLOQUEADA por las que sigan OPEN. Al cerrar una issue, volver a correr el
script para ver qué se desbloquea.

## Workflow (ver AGENTS.md)
Ramas `t0/... t1/... t2/... t3/...`, un PR = una issue, reviewer cruzado obligatorio.
