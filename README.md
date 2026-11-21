# PSet #2 — Pipeline de datos ECU 911

Pipeline ELT batch que prepara los datos para predecir **días de demanda inusualmente alta de emergencias por cantón** (horizontes t+3 y t+7, definido en el PSet #1).

```
Portal Datos Abiertos ──► Kestra ──► Snowflake BRONZE ──dbt──► SILVER ──dbt──► GOLD ──Spark──► OBT
(62 archivos mensuales)   (ingesta,     (dato crudo +          (limpio)     (star schema)   (1 tabla para ML)
                           retries,       metadata)
                           backfill)
```

## Fuente

**ECU911 Base de Emergencias** — https://www.datosabiertos.gob.ec/dataset/base-de-emergencias
Un archivo por mes (2021-07 → 2026-08), 7 columnas: `Fecha; provincia; Canton; Cod_Parroquia; Parroquia; Servicio; Subtipo`.
CSV con `;` (y 2 meses en XLSX). ~17.7 millones de registros. Sin ID de incidente.

## Estructura

```
.
├── docker-compose.yml        # postgres, kestra, kestra-init, spark-master, spark-worker y preparación de dbt
├── docker/dbt/               # Dockerfile y versiones de dbt; construcción nativa AMD64/ARM64
├── .env.example              # plantilla de variables (copiar a .env)
├── kestra/
│   ├── deploy_flows.sh       # lo ejecuta kestra-init: sube los flows y el proyecto dbt a Kestra por API
│   └── flows/
│       ├── ecu911_run_all.yml            # setup + carga por rango + transform en una ejecución
│       ├── ecu911_snowflake_setup.yml    # crea warehouse, database, schemas, rol y usuario
│       ├── ecu911_ingest_month.yml       # carga UN mes a BRONZE (idempotente)
│       ├── ecu911_ingest_scheduled.yml   # trigger mensual + backfill; al final lanza transform
│       ├── ecu911_transform.yml          # dbt_build -> spark_obt -> reconciliación de capas
│       ├── ecu911_dbt_build.yml          # corre dbt (source freshness + build) en un contenedor
│       └── ecu911_spark_obt.yml          # corre el job de Spark que construye la OBT
├── spark/jobs/               # build_obt.py (OBT desde GOLD)
├── dbt/
│   ├── dbt_project.yml, profiles.yml     # profiles.yml sin secretos (env_var)
│   ├── macros/               # generate_schema_name (schemas SILVER/GOLD exactos), limpiar_texto
│   ├── seeds/                # feriados_ecuador.csv, cantones_reasignados.csv
│   ├── models/silver/        # _sources.yml (BRONZE), emergencias.sql, _silver.yml (tests)
│   ├── models/gold/          # dim_canton, dim_fecha, fct_emergencias_canton_dia, _gold.yml (tests)
│   └── tests/                # tests singulares (conservación de filas, grain, date spine, sumas)
└── docs/
    ├── modelo_dimensional.md # star schema: diagrama, grain y decisiones
    └── data_quality.sql      # profiling de calidad sobre BRONZE
```

## Requisitos

- Docker Desktop (probado con Docker 29.8, Compose v5, 8 GB RAM asignados)
- En Windows, Docker Desktop debe usar contenedores Linux.
- Una cuenta de Snowflake con un usuario ACCOUNTADMIN

## Cómo ejecutar

### 1. Variables
```powershell
copy .env.example .env
```
En macOS/Linux: `cp .env.example .env`.
Completar en `.env`: `SNOWFLAKE_ADMIN_USER`, `SNOWFLAKE_ADMIN_PASSWORD`, `SNOWFLAKE_ACCOUNT` (formato `orgname-accountname`)
y cambiar las contraseñas de ejemplo. **`.env` nunca se sube a Git.**

### 2. Levantar la infraestructura
```powershell
docker compose up -d
docker compose ps -a          # kestra, postgres, spark-* activos; dbt y kestra-init "Exited (0)"
docker compose logs kestra-init   # "Todos los flows desplegados."
```
- Kestra: http://localhost:8080 (usuario/contraseña de `.env`)
- Spark master: http://localhost:8090

Compose construye `ecu911-dbt:1.9.0` desde `docker/dbt/Dockerfile`, instala dbt Core y el
adaptador Snowflake 1.9.0 y ejecuta `dbt --version`. Kestra espera a que esta comprobación
termine correctamente y usa esa misma imagen para sus tareas. La construcción usa la
arquitectura del equipo: ARM64 en Mac Apple Silicon y AMD64 en Intel/AMD, incluyendo
Windows con Docker Desktop. No hay que forzar `platform` ni descargar una imagen AMD64
en un Mac ARM. La primera construcción requiere Internet y tarda más que los siguientes arranques.

> Si se edita `.env`, volver a correr `docker compose up -d` (Docker solo lee `.env` al crear el contenedor).

### Atajo: todo con un clic
Flows → `ecu911.run_all` → **Execute** (defaults: `desde = 2021-07`, `hasta` vacío = mes anterior, `setup = true`).
Ejecuta `snowflake_setup` → `ingest_month` por cada mes del rango (4 a la vez) → `transform`. Idempotente.
Para reproducir el histórico de 62 meses y los conteos documentados abajo, usa
`desde = 2021-07`, `hasta = 2026-08` y `setup = true`.
Si dejas `hasta` vacío y el mes anterior todavía no está publicado, esa carga falla con
un mensaje explícito y `run_all` puede terminar en `WARNING`; la transformación continúa
con los meses disponibles. Revisa que `transform`, `dbt_build`, `spark_obt` y la
reconciliación hayan terminado correctamente. `setup = false` requiere que los objetos
y permisos de Snowflake ya existan.
Los pasos 3–4 y 6–7 de abajo hacen lo mismo por partes.

### 3. Crear la infraestructura en Snowflake
Kestra → Flows → `ecu911.snowflake_setup` → **Execute**.
Crea `ECU911_WH` (X-Small, auto-suspend 60 s), `ECU911` con schemas `BRONZE/SILVER/GOLD/OBT`, rol `ECU911_ROLE` y usuario `ECU911_USER`. Idempotente.

### 4. Ingesta
- **Un mes:** Flows → `ecu911.ingest_month` → Execute con `month = 2026-08`.
- **Programada (pipeline completo):** `ecu911.ingest_scheduled` corre el día 10 de cada mes (06:00 Ecuador), carga los 2 meses
  anteriores y luego lanza `ecu911.transform` (dbt → Spark → reconciliación de capas). ~7 min.
- **Backfill histórico:** Flows → `ingest_scheduled` → Triggers → `monthly` → **Backfill executions**:
  inicio `2021-08-01`, fin = hoy, inputs `meses = 1` y **`transformar = false`**. Carga 2021-07 → último mes publicado (~25 min).
  Al terminar, ejecutar **una vez** `ecu911.transform`.

### 5. Verificar en Snowflake
```sql
SELECT _PERIODO, COUNT(*) filas, COUNT(DISTINCT _EJECUCION_KESTRA) ejecuciones
FROM ECU911.BRONZE.EMERGENCIAS_RAW GROUP BY 1 ORDER BY 1;
-- esperado: 62 meses, 17,680,253 filas, ejecuciones = 1 en cada mes
```

### 6. dbt (Silver y Gold)
- **Desde Kestra (pipeline):** Flows → `ecu911.dbt_build` → Execute (input `select = *`).
  Corre `dbt source freshness` y `dbt build` en un contenedor de la imagen local `ecu911-dbt:1.9.0`.
- **Manual (desarrollo):**
  ```powershell
  docker compose run --rm dbt debug      # prueba la conexión
  docker compose run --rm dbt build      # seeds + SILVER + GOLD + tests: PASS=39 (~45 s)
  ```
- Si se edita algo en `dbt/`, re-subirlo a Kestra con `docker compose up -d kestra-init`.
- Demo de un test que falla (sin tocar código): `docker compose run --rm dbt build --vars "{periodos_anomalos: []}"`
  → falla `silver_volumen_mensual_estable` (2024-01) y se saltan los modelos Gold. Luego restaurar con `dbt build` normal.

```sql
SELECT COUNT(*) filas, SUM(IFF(NOT ES_UBICACION_VALIDA,1,0)) sin_ubicacion
FROM ECU911.SILVER.EMERGENCIAS;   -- esperado: 17,680,253 filas (= BRONZE), 2,160 sin ubicación

SELECT COUNT(*) filas, SUM(N_EMERGENCIAS) emergencias
FROM ECU911.GOLD.FCT_EMERGENCIAS_CANTON_DIA;   -- esperado: 422,912 (224 cantones x 1,888 días), 17,678,093
```
Modelo dimensional: [`docs/modelo_dimensional.md`](docs/modelo_dimensional.md).

### 7. Spark (OBT)
- **Desde Kestra:** Flows → `ecu911.spark_obt` → Execute (~4 min).
- **Manual:**
  ```powershell
  docker compose exec spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 `
    --packages net.snowflake:spark-snowflake_2.12:3.2.2-spark_3.5 --conf spark.jars.ivy=/tmp/.ivy2 `
    /opt/spark-jobs/build_obt.py
  ```
  Debe terminar en `OBT OK` con 7 líneas `[validar] OK` (filas, duplicados, suma, match de cantón y de fecha,
  `es_feriado_t3` y `es_feriado_t7` sin nulos). La OBT tiene 37 columnas; la convención temporal de las features
  está en el docstring de `build_obt.py`.
- UI del cluster: http://localhost:8090 (la aplicación `pset2-obt` aparece mientras corre).

```sql
SELECT COUNT(*), COUNT(DISTINCT COD_CANTON, FECHA) FROM ECU911.OBT.OBT_EMERGENCIAS_CANTON_DIA;  -- 422,912 y 422,912
```

## Documento técnico

El [documento técnico (PDF)](docs/PSet2_memo_Tierra_Rodriguez_Rodriguez_Cadena_Jarrin.pdf)
está incluido en el repositorio y tiene seis páginas.
Incluye la arquitectura, las decisiones de calidad, el modelo dimensional, la OBT,
los resultados verificados y las limitaciones.

## Apagar
```powershell
docker compose stop      # conserva todo (flows, historial de Kestra); los datos viven en Snowflake
docker compose down -v   # borra también los volúmenes (historial de Kestra); Snowflake no se toca
```
