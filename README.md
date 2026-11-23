# PSet #3 — Experimentación y selección del modelo (ECU 911)

Predice si un **cantón tendrá un día de demanda inusualmente alta de emergencias dentro de 3 días**
(`y = 1` si las emergencias en `t+3` superan el P90 de las 52 ocurrencias anteriores del mismo día de la semana
en ese cantón). Parte de la OBT del PSet 2 en Snowflake y termina con el modelo ganador registrado en MLflow.

```
OBT (Snowflake) → EDA → Features → Pipeline → Línea base → Candidatos → Tuning → Selección → Model Registry
   01_eda           02_features      src/pipeline.py   03_baseline_candidatos   04_tuning  05_seleccion  06_reproduccion
```

## Estructura

```
.
├── docker-compose.yml        # + servicio mlflow (tracking + registry); Kestra/Spark/dbt del PSet 2 siguen igual
├── .env.example              # plantilla (copiar a .env): Snowflake + MLFLOW_TRACKING_URI
├── requirements.txt          # entorno Python 3.12 de los notebooks
├── notebooks/
│   ├── 01_eda.ipynb                  # calidad, distribución del target, atípicos (IQR / z), Spearman, 5 hallazgos
│   ├── 02_features.ipynb             # features derivadas, evidencia, ablación y prueba de no-fuga
│   ├── 03_baseline_candidatos.ipynb  # 2 líneas base, 6 candidatos tabulares, Chronos-Bolt y un LLM por prompting
│   ├── 04_tuning.ipynb               # RandomizedSearchCV 30 configs × top 3, corridas padre/hija
│   ├── 05_seleccion.ipynb            # ganador vs segundo, test único, métrica de negocio, registry, corridas.csv
│   └── 06_reproduccion.ipynb         # notebook limpio: carga @champion y reproduce la métrica registrada
├── src/
│   ├── datos.py              # Snowflake → snapshot parquet, hash, target, features, split y CV temporal
│   ├── pipeline.py           # ColumnTransformer + Pipeline, regla estacional, métricas, candidatos y espacios
│   ├── experimento.py        # logging estándar de corridas en MLflow
│   └── sota.py               # Chronos-Bolt (zero-shot) y LLM local por prompting
├── docs/
│   ├── corridas.csv          # export de TODAS las corridas (mlflow.search_runs)
│   ├── PSet3_memo_Tierra_Rodriguez_Rodriguez_Cadena_Jarrin.pdf
│   ├── figuras/              # gráficos usados en el memo (los generan los notebooks)
│   └── README_pset2.md       # instrucciones del pipeline de datos (Kestra → dbt → Spark)
├── dbt/, kestra/, spark/     # pipeline del PSet 2 que construye la OBT
```

## 1. Variables y conexión a Snowflake

```powershell
copy .env.example .env        # macOS/Linux: cp .env.example .env
```
Completar en `.env` (nunca se sube a Git):

| Variable | Uso |
|---|---|
| `SNOWFLAKE_ACCOUNT` | identificador `orgname-accountname` (sin `.snowflakecomputing.com`) |
| `SNOWFLAKE_USER` / `SNOWFLAKE_PASSWORD` | usuario del pipeline (`ECU911_USER`, lo crea el flow `snowflake_setup` del PSet 2) |
| `SNOWFLAKE_ROLE`, `SNOWFLAKE_WAREHOUSE`, `SNOWFLAKE_DATABASE` | `ECU911_ROLE`, `ECU911_WH`, `ECU911` |
| `SNOWFLAKE_ADMIN_USER` / `SNOWFLAKE_ADMIN_PASSWORD` | sólo para crear la infraestructura con Kestra |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` (default) |

Si la OBT aún no existe en tu cuenta, constrúyela con el pipeline del PSet 2
([docs/README_pset2.md](docs/README_pset2.md)): `docker compose up -d` y en Kestra ejecutar `ecu911.run_all`
con `desde = 2021-07`, `hasta = 2026-08`, `setup = true`. Resultado esperado:
`ECU911.OBT.OBT_EMERGENCIAS_CANTON_DIA` = 422,912 filas (224 cantones × 1,888 días).

`src/datos.py::cargar_obt()` lee la OBT **una vez** y la guarda en `data/obt.parquet` (ignorado por Git).
Todos los notebooks usan ese snapshot, cuyo hash (`datos_hash`) y fecha de corte (`datos_fecha_corte = 2026-08-31`)
quedan en cada corrida. Para volver a leer Snowflake: `cargar_obt(refrescar=True)`.

## 2. Levantar MLflow

```powershell
docker compose up -d mlflow          # sólo MLflow (no hace falta Kestra/Spark para el PSet 3)
```
- UI: http://localhost:5000 — experimento **`ecu911-demanda-alta`**, modelo registrado **`ecu911-modelo`**.
- Backend store: SQLite (`/mlflow/mlflow.db`) y artifact store (`/mlflow/artifacts`) en el volumen `mlflow-data`;
  el servidor sirve los artefactos (`--serve-artifacts`), así el cliente sube los modelos por HTTP.
- `docker compose down` conserva las corridas; `docker compose down -v` las borra.

## 3. Entorno Python

```powershell
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m ipykernel install --user --name pset3 --display-name "Python (pset3)"
```
Los modelos fundacionales (`amazon/chronos-bolt-small`, `Qwen/Qwen2.5-0.5B-Instruct`) se descargan de
Hugging Face la primera vez (~1.2 GB) y corren en CPU; no se necesita ninguna API key.

## 4. Orden de ejecución de los notebooks

Ejecutar en orden, cada uno con **Restart & Run All** (las salidas guardadas son las de esa ejecución):

| # | Notebook | Tiempo aprox. (12 núcleos, CPU) | Corridas en MLflow |
|---|---|---|---|
| 1 | `01_eda.ipynb` | 2 min (+1 min la primera lectura de Snowflake) | — |
| 2 | `02_features.ipynb` | 3 min | — |
| 3 | `03_baseline_candidatos.ipynb` | ~60 min (Chronos y el LLM son lo más lento) | 2 baseline + 8 candidatos |
| 4 | `04_tuning.ipynb` | ~60 min | 3 padres + 90 hijas |
| 5 | `05_seleccion.ipynb` | ~10 min | 3 finales + registry + `docs/corridas.csv` |
| 6 | `06_reproduccion.ipynb` | 1 min | — (sólo carga `@champion`) |

Desde la terminal:
```powershell
cd notebooks
foreach ($n in "01_eda","02_features","03_baseline_candidatos","04_tuning","05_seleccion","06_reproduccion") {
  ..\.venv\Scripts\jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=14400 "$n.ipynb"
}
```

## Decisiones principales

- **Validación:** `CVTemporal` = TimeSeriesSplit sobre fechas (5 pliegues de 120 días, train expansivo, `gap` = 3
  días); hold-out = fechas ≥ 2026-03-01 (6 meses), usado una sola vez en `05_seleccion`.
- **Métrica técnica:** PR-AUC (average precision) — desbalance 1:9.
- **Métrica de negocio:** reducción de picos **no anticipados** frente a la regla estacional del PSet 1 con el
  **mismo número de alertas**, y costo `5·FN + 1·FP` por 1000 cantón-días con el umbral elegido en CV.
- **Sin fuga:** features calculadas sólo con días ≤ t (verificado truncando el futuro en `02_features`);
  imputación/one-hot/escalado dentro del `Pipeline`, ajustados en cada pliegue.

Los resultados (tablas, figuras, run_id y versión del champion) están en los notebooks, en `docs/` y en el memo.
