# PSet #3 — Experimentación y selección del modelo (ECU 911)

Encontrar, con evidencia reproducible, el mejor modelo para anticipar **días de demanda inusualmente alta
de emergencias por cantón** en dos horizontes: **t+3** (ciclo de turnos) y **t+7** (planificación semanal).

`y_h = 1` si las emergencias del cantón en t+h superan el **P90** de ese cantón para el día de la semana de t+h.
El P90 se calcula **solo con fechas de entrenamiento** (se recalcula en cada pliegue de la validación cruzada).

```
OBT (Snowflake, PSet 2) → snapshot parquet → EDA → Features → Pipeline → Líneas base → Candidatos
→ Tuning (Optuna) → Selección → Model Registry (MLflow)
```

## Estructura

```
pset_3/
├── docker-compose.yml          # pipeline del PSet 2 + MLflow (perfil "mlflow")
├── docker/mlflow/Dockerfile    # servidor MLflow 3.18 + psycopg2
├── .env.example                # plantilla de variables (copiar a .env; .env nunca va a Git)
├── requirements.txt            # Python 3.11, versiones fijadas
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_features.ipynb
│   ├── 03_baseline_candidatos.ipynb   # 2 líneas base, 5 familias (incl. TimesFM) y LLM (Ollama)
│   ├── 04_tuning.ipynb                # Optuna, 30 configuraciones × 3 modelos × 2 horizontes
│   ├── 05_seleccion.ipynb             # hold-out (una vez), métrica de negocio, Model Registry
│   └── 06_reproduccion.ipynb          # notebook limpio: carga @champion y reproduce la métrica
├── src/
│   ├── datos.py          # snapshot de la OBT y tabla de modelado por horizonte
│   ├── validacion.py     # hold-out temporal, pliegues por fecha con gap, P90 y target
│   ├── pipeline.py       # Pipeline + ColumnTransformer (P90 aprendido en fit), familias por defecto
│   ├── experimento.py    # bucle de CV, umbral de negocio, registro en MLflow
│   ├── fundacional.py    # TimesFM 2.5 (zero-shot)
│   ├── llm.py            # prompting con Ollama
│   └── reporte.py        # guardado de figuras y tablas para el memo
├── tests/                # pytest (sin Snowflake ni MLflow: usan datos sintéticos)
├── docs/
│   ├── corridas.csv      # todas las corridas del experimento (mlflow.search_runs)
│   ├── figuras/, resultados/   # salidas de los notebooks usadas en el memo
│   ├── PSet3_memo_Tierra_Rodriguez_Rodriguez_Cadena_Jarrin.pdf
│   └── pipeline_pset2.md # cómo funciona el pipeline de datos del PSet 2
└── dbt/, kestra/, spark/, docker/dbt/   # pipeline del PSet 2 (sin cambios)
```

## Requisitos

- macOS/Linux/Windows con **Docker Desktop**. Probado en un Mac M3 de 8 GB.
- **Python 3.11** (TimesFM/torch no soportan 3.13+). Recomendado: [`uv`](https://docs.astral.sh/uv/).
- **Ollama** instalado de forma nativa (en Docker para Mac no usa la GPU).
- Una cuenta de **Snowflake** (solo para cargar los datos la primera vez).

## 1. Variables de entorno

```bash
cp .env.example .env
```

Completa `SNOWFLAKE_ADMIN_USER` (el **login_name**: `SHOW USERS;` en Snowflake), `SNOWFLAKE_ADMIN_PASSWORD`,
`SNOWFLAKE_ACCOUNT` (formato `ORGANIZACION-CUENTA`, p. ej. `SELECT CURRENT_ORGANIZATION_NAME(), CURRENT_ACCOUNT_NAME();`),
`SNOWFLAKE_PASSWORD` (inventada: la del usuario de servicio que se crea) y las contraseñas locales.

> Si una contraseña tiene `$`, escríbela entre **comillas simples** (`CLAVE='a$b'`): Docker Compose interpreta `$`.

## 2. Cargar los datos en Snowflake (pipeline del PSet 2, una sola vez)

```bash
docker compose up -d                # Kestra, Postgres, Spark, dbt
```

Kestra (http://localhost:8080) → Flows → `ecu911.run_all` → **Execute** con `desde = 2021-07`,
`hasta = 2026-08`, `setup = true` (~40 min). Verificación:

```sql
SELECT COUNT(*), COUNT(DISTINCT COD_CANTON, FECHA), MAX(FECHA) FROM ECU911.OBT.OBT_EMERGENCIAS_CANTON_DIA;
-- 422,912 | 422,912 | 2026-08-31
```

Detalle del pipeline: [`docs/pipeline_pset2.md`](docs/pipeline_pset2.md). Al terminar, el pipeline se puede apagar:
`docker compose stop kestra postgres spark-master spark-worker`.

## 3. Entorno de Python y snapshot de los datos

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -c "from src.datos import descargar_snapshot; descargar_snapshot()"
.venv/bin/pytest                    # 52 tests, ~30 s
```

`descargar_snapshot` lee la OBT una vez y la guarda en `data/obt_snapshot.parquet` (fuera de Git). Todos los
notebooks leen ese archivo; cada corrida de MLflow registra la tabla de origen, la fecha de corte y el
**SHA-256** del snapshot, así que una corrida solo se reproduce con el mismo archivo.
Snapshot usado en la entrega: corte `2026-08-31`, SHA-256 `ff042a2e8d3bbef0555c5658f546f3a44b59df069d415b102ea13e91cb5a6c90`.

## 4. Levantar MLflow

```bash
docker compose --profile mlflow up -d mlflow      # solo MLflow + su Postgres (~600 MB)
curl localhost:5001/health                          # OK
```

- UI: **http://localhost:5001** (el puerto 5000 lo usa AirPlay en macOS).
- Backend store: Postgres propio (`mlflow-db`, volumen `mlflow-db-data`).
- Artifact store: volumen `mlflow-artifacts`, servido por el propio servidor (`--serve-artifacts`).
- Experimento: `ecu911-demanda-alta`. Modelos registrados: `ecu911-t3-modelo` y `ecu911-t7-modelo`
  (alias `champion` y `challenger`).

## 5. Ollama (LLM por prompting, notebook 03)

```bash
brew install ollama && brew services start ollama   # o el instalador de ollama.com
ollama pull qwen2.5:3b                              # ~2 GB; cabe en 8 GB de RAM
```

## 6. Orden de ejecución de los notebooks

Desde la carpeta `notebooks/`, con el kernel del `.venv`, **de uno en uno** y con **Restart → Run All**:

| # | Notebook | Necesita | Tiempo aprox. (M3, 8 GB) |
|---|---|---|---|
| 1 | `01_eda.ipynb` | snapshot | 1 min |
| 2 | `02_features.ipynb` | snapshot | 10 min |
| 3 | `03_baseline_candidatos.ipynb` | MLflow, Ollama | 2–3 h (TimesFM ≈ 100 min) |
| 4 | `04_tuning.ipynb` | MLflow, salida del 03 | varias horas (reanudable: Optuna guarda en `data/optuna.db`) |
| 5 | `05_seleccion.ipynb` | MLflow, salida del 04 | 10 min |
| 6 | `06_reproduccion.ipynb` | MLflow, registry | 2 min; exporta `docs/corridas.csv` |

También desde la terminal: `jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb`.

> **Memoria (8 GB):** baja Docker Desktop a 2 GB después de la carga de datos, cierra apps pesadas y no corras
> dos notebooks a la vez. El 03 libera TimesFM y el modelo de Ollama apenas los usa.

## Problemas conocidos

| Síntoma | Solución |
|---|---|
| Kestra se reinicia con `password authentication failed for user "kestra"` | Su Postgres guardó otra contraseña: `docker compose exec postgres sh -c 'psql -U kestra -d kestra -c "ALTER USER kestra PASSWORD '"'"'$POSTGRES_PASSWORD'"'"';"'` y `docker compose restart kestra` |
| `Incorrect username or password` en Snowflake | Usar el `login_name`, no el nombre visible; contraseñas con `$` entre comillas simples |
| `localhost:5000` no responde | MLflow está en el **5001** |
| El servidor MLflow reinicia sus workers | Ya resuelto en Compose: `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false` y `--workers 2` |

## Documento técnico

[`docs/PSet3_memo_Tierra_Rodriguez_Rodriguez_Cadena_Jarrin.pdf`](docs/PSet3_memo_Tierra_Rodriguez_Rodriguez_Cadena_Jarrin.pdf)
(≤ 6 páginas + anexos): decisiones de EDA, features, validación, diseño experimental, resultados, tuning,
selección y limitaciones.
