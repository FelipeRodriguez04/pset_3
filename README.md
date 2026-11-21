# PSet #3 — Experimentación y selección del modelo (ECU 911)

Modelo que anticipa **días de demanda inusualmente alta de emergencias por cantón a t+3**
(PSet 1): `y_t3 = 1` si el volumen del cantón en t+3 supera su **P90 histórico para ese día de
la semana**, calculado solo con datos de entrenamiento. Los datos son la OBT que construyó el
pipeline del PSet 2 en Snowflake.

```
OBT (Snowflake) → EDA → Features → Pipeline → Líneas base → Candidatos → Tuning → Selección → Model Registry
   src/datos.py    01      02      src/pipeline.py      03                  04          05        MLflow
```

## Estructura

```
Pset_3/
├── docker-compose.yml        # MLflow tracking server + Postgres (backend store) + volumen (artifact store)
├── .env.example              # plantilla de variables (copiar a .env)
├── requirements.txt          # versiones exactas (Python 3.12)
├── notebooks/
│   ├── 01_eda.ipynb                    # tamaño, tipos, faltantes, target, atípicos, Spearman, hallazgos
│   ├── 02_features.ipynb               # features derivadas, ablación, prueba de no-fuga, codificación
│   ├── 03_baseline_candidatos.ipynb    # 2 líneas base + 5 familias con hiperparámetros por defecto
│   ├── 04_tuning.ipynb                 # Optuna: 3 mejores x 30 configuraciones (corridas anidadas)
│   └── 05_seleccion.ipynb              # ganador vs segundo, test único, umbral por costo, registry, corridas.csv
├── src/
│   ├── datos.py              # lectura de la OBT (Snowflake, respaldo CSV), target, división train/test
│   └── pipeline.py           # Pipeline + ColumnTransformer, CV temporal, métricas, registro en MLflow
└── docs/
    ├── corridas.csv          # todas las corridas del experimento (mlflow.search_runs)
    ├── figuras/              # gráficos que generan los notebooks
    └── *.csv                 # tablas de resultados de cada notebook
```

## Requisitos

- Docker Desktop (contenedores Linux).
- Python 3.12 (recomendado con [uv](https://docs.astral.sh/uv/)).
- Acceso a la cuenta de Snowflake donde corrió el pipeline del PSet 2 (tabla
  `ECU911.OBT.OBT_EMERGENCIAS_CANTON_DIA`).
- Internet la primera vez: imagen de MLflow, paquetes de Python y los pesos de Chronos-Bolt
  (Hugging Face).

## 1. Configurar la conexión a Snowflake

```powershell
copy .env.example .env        # macOS/Linux: cp .env.example .env
```

Completar en `.env`:

| Variable | Valor |
|---|---|
| `SNOWFLAKE_ACCOUNT` | identificador de la cuenta, formato `ORGNAME-ACCOUNTNAME` (Snowsight → cuenta → *View account details*) |
| `SNOWFLAKE_USER` / `SNOWFLAKE_PASSWORD` | el usuario técnico que creó el flow `ecu911.snowflake_setup` del PSet 2 (`ECU911_USER`) |
| `SNOWFLAKE_ROLE`, `SNOWFLAKE_WAREHOUSE`, `SNOWFLAKE_DATABASE` | `ECU911_ROLE`, `ECU911_WH`, `ECU911` (valores por defecto) |
| `MLFLOW_DB_PASSWORD` | contraseña local de la base de MLflow (cualquiera) |

`src/datos.py::cargar_obt()` lee la tabla desde Snowflake. Si Snowflake no responde, usa el
respaldo `data/obt_emergencias_canton_dia.csv` (exportación de la misma tabla, no versionada).
Las dos fuentes se normalizan al mismo esquema y producen el **mismo hash**
(`cf74c52dd569…`), que queda registrado en cada corrida. **`.env` nunca se sube a Git.**

## 2. Levantar MLflow

```powershell
docker compose up -d
docker compose ps             # mlflow y mlflow-db en estado "healthy"
```

- UI: **http://localhost:5000** — experimento `ecu911-alta-demanda`, modelo registrado `ecu911-modelo`.
- Backend store: Postgres (`mlflow-db`, volumen `mlflow-db`). Artifact store: volumen
  `mlflow-artifacts`, servido por el propio MLflow (`--serve-artifacts`), así que los notebooks
  suben los modelos por HTTP sin compartir carpetas.
- Apagar: `docker compose stop` (conserva todo) · `docker compose down -v` (borra corridas y modelos).

## 3. Entorno de Python

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt --index-strategy unsafe-best-match
```

Sin uv: `python -m venv .venv` y `.venv\Scripts\pip install -r requirements.txt`.
En VS Code/Jupyter, elegir el intérprete `.venv`.

## 4. Orden de ejecución de los notebooks

Ejecutar en orden, cada uno con **Restart** y luego **Run All** (MLflow debe estar arriba desde el 03):

| # | Notebook | Qué produce | Tiempo aprox. (CPU 8 núcleos) |
|---|---|---|---|
| 1 | `01_eda` | hallazgos con evidencia (`docs/eda_hallazgos.csv`) | 1 min |
| 2 | `02_features` | ablación y prueba de no-fuga | 4 min |
| 3 | `03_baseline_candidatos` | 7 corridas: 2 líneas base + 5 candidatos; `docs/top3_para_tuning.csv` | ~1 h (Chronos ~45 min) |
| 4 | `04_tuning` | 3 corridas padre + 90 hijas | 1,5–3 h |
| 5 | `05_seleccion` | corrida final, versiones `champion`/`challenger`, reproducción, `docs/corridas.csv` | ~15 min |

Desde la terminal (equivalente a Restart + Run All, guarda las salidas):

```powershell
.venv\Scripts\jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks\01_eda.ipynb
```

## Decisiones principales

- **Target y división:** `y_t3` con el P90 por (cantón, día de la semana). Hold-out = las fechas
  objetivo de los últimos 12 meses (2025-09-01 → 2026-08-31), usado una sola vez en el notebook 05.
  Enero de 2024 (subregistro detectado en el PSet 2) se excluye del entrenamiento.
- **Validación:** CV temporal de 5 pliegues de ventana creciente **sobre fechas**, con hueco de 3
  días (horizonte) y el **umbral P90 del target recalculado en cada pliegue** con su entrenamiento.
- **Preparación:** `Pipeline` = `PerfilCantonal` (P90 aprendido en entrenamiento; conteos
  relativos a ese umbral) + `ColumnTransformer` (imputación, escalado, one-hot), igual para todos
  los modelos.
- **Métricas:** técnica = PR-AUC; negocio = reducción de días de alta demanda no anticipados
  frente a la regla estacional con el mismo número de alertas, y costo 5·FN + 1·FP para el umbral.
- **Reproducibilidad:** cada corrida registra semilla, hiperparámetros, tabla de origen, fecha de
  corte y hash del snapshot, commit de Git, métricas por pliegue y el pipeline con firma y ejemplo.
