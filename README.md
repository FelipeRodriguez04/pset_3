# Juan Diego Cadena, 329220

## PSet 3 - Experimentación y selección del modelo ECU 911

Este repositorio compara modelos para anticipar **demanda inusualmente alta de emergencias por cantón**.
El modelo se ejecuta en el día `t` y estima alta demanda en `t+3`; también se registra una sensibilidad
para `t+7`. La etiqueta vale 1 cuando el conteo futuro supera el percentil 90 del cantón y día de semana,
calculado únicamente con el periodo de entrenamiento.

La implementación usa actualmente la OBT local completa. Cuando exista la cuenta nueva de Snowflake,
solo se cambia `DATA_SOURCE=snowflake`: notebooks, features, división y modelos permanecen iguales.

## Estructura

```text
pset_3/
├── docker-compose.yml
├── .env.example
├── requirements.txt
├── README.md
├── run_pset3.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_features.ipynb
│   ├── 03_baseline_candidatos.ipynb
│   ├── 04_tuning.ipynb
│   └── 05_seleccion.ipynb
├── src/
│   ├── datos.py
│   ├── pipeline.py
│   └── experimentos.py
└── docs/
    ├── corridas.csv
    ├── tablas y gráficos
    └── PSet3_memo_Tierra_Rodriguez_Rodriguez_Cadena_Jarrin.pdf
```

## 1. Preparar el entorno

En PowerShell, desde esta carpeta:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

`.env` está ignorado por Git. No se deben guardar credenciales en notebooks, scripts ni commits.

## 2. Levantar MLflow

La opción de entrega usa Docker y conserva tanto SQLite como artefactos en `.mlflow/`:

```powershell
docker compose up -d
docker compose ps
```

Abrir <http://127.0.0.1:5050>. Para apagar sin borrar corridas:

```powershell
docker compose stop
```

Para pruebas sin Docker también se puede iniciar el mismo servidor con el entorno local:

```powershell
.\.venv\Scripts\mlflow.exe server `
  --backend-store-uri sqlite:///C:/ruta/absoluta/pset_3/.mlflow/mlflow.db `
  --default-artifact-root file:///C:/ruta/absoluta/pset_3/.mlflow/artifacts `
  --host 127.0.0.1 --port 5050
```

## 3. Fuente local

Copiar `.env.example` a `.env` y conservar:

```dotenv
DATA_SOURCE=local
OBT_LOCAL_PATH=./data/obt_emergencias_canton_dia.csv
```

Crear la carpeta `data` y copiar allí la OBT con ese nombre. La carpeta está ignorada por Git para evitar
subir accidentalmente el CSV. También se puede usar una ruta absoluta en `OBT_LOCAL_PATH`.

La OBT local tiene 429.632 filas, 224 cantones, 37 columnas y fechas 2021-07-01 a 2026-09-30.
La división se fija antes de experimentar: las etiquetas de entrenamiento terminan el 2026-03-31,
se purgan tres días en el borde y el holdout empieza el 2026-04-01.

Antes de ejecutar, se puede comprobar que la ruta local está disponible sin entrenar ningún modelo:

```powershell
$obt = (Get-Content .env | Select-String '^OBT_LOCAL_PATH=').Line.Split('=', 2)[1]
Test-Path (Join-Path (Get-Location) $obt)
```

La OBT no se versiona por su tamaño. Para la revisión compartida cada integrante debe conservar el CSV local
o cambiar `OBT_LOCAL_PATH` por su ruta absoluta. La conexión a Snowflake queda preparada, pero no forma parte
de esta revisión local.

## 4. Configurar Snowflake después

Cuando esté creada la cuenta, copiar la OBT a `ECU911.OBT.OBT_EMERGENCIAS_CANTON_DIA` y cambiar:

```dotenv
DATA_SOURCE=snowflake
SNOWFLAKE_ACCOUNT=organizacion-cuenta
SNOWFLAKE_USER=...
SNOWFLAKE_PASSWORD=...
SNOWFLAKE_ROLE=ECU911_ROLE
SNOWFLAKE_WAREHOUSE=ECU911_WH
SNOWFLAKE_DATABASE=ECU911
SNOWFLAKE_SCHEMA=OBT
SNOWFLAKE_TABLE=OBT_EMERGENCIAS_CANTON_DIA
```

`src/datos.py` ejecuta el `SELECT`, normaliza nombres de columnas y valida el mismo contrato que el CSV.
No hay que modificar ningún notebook.

## 5. Orden de ejecución

Con MLflow activo, abrir los notebooks en este orden y en cada uno usar **Restart Kernel and Run All**:

1. `01_eda.ipynb`
2. `02_features.ipynb`
3. `03_baseline_candidatos.ipynb`
4. `04_tuning.ipynb`
5. `05_seleccion.ipynb`

También existe un ejecutor equivalente:

```powershell
.\.venv\Scripts\python.exe run_pset3.py all
```

Las fases son idempotentes y reutilizan los CSV de resultados. Usar `--force` únicamente cuando se quiera
repetir deliberadamente los entrenamientos y crear nuevas corridas.

## Diseño experimental

- Cuatro backtests expansivos con semestres completos: enero-junio y julio-diciembre de 2024 y 2025.
- Purga temporal de tres días antes de cada validación y del holdout, porque el objetivo es `t+3`.
- PR-AUC como métrica técnica y PR-AUC ponderado para dar peso x2 a ventanas de feriado y x2.5 a fin de año.
- Pruebas especiales para feriados, ventanas ±2 días, Carnaval/Semana Santa, diciembre, fin de año y días regulares.
- Features de calendario objetivo, ciclos anuales y rezagos 364/365/371, además de historia reciente.
- Costo de negocio `15*FN + FP`; además, el umbral busca al menos 10% menos picos no anticipados que la regla estacional manteniendo una precisión mínima.
- Baselines: prevalencia y promedio de las cuatro semanas equivalentes.
- Candidatos: regresión logística, Random Forest, HistGradientBoosting, MLP y CatBoost.
- Tuning: HistGradientBoosting, CatBoost y MLP, 30 configuraciones por familia y corridas hijas en MLflow.
- Random Forest no pasó al tuning porque su brecha de generalización fue `0.2682`.
- Holdout: se usa una sola vez con el ganador y el segundo mejor.
- Registry: `ecu911-alta-demanda-modelo@champion` y `@challenger`.

No se usa un LLM por prompting como candidato porque no puede recibir de manera reproducible las 429 mil
observaciones tabulares ni producir probabilidades calibradas sin enviar datos a un servicio externo. La familia
SOTA se representa con CatBoost, adecuada para tabulares y ejecutable localmente.

### Comparación inicial con hiperparámetros por defecto

La brecha es `PR-AUC train - PR-AUC CV`. Los tiempos muestran ajuste final y los cuatro backtests. La tasa de
aprendizaje no aplica a reglas, modelos lineales ni Random Forest.

| Modelo | PR-AUC train | PR-AUC CV ± desv. | Brecha | Learning rate | Fit / CV (s) |
|---|---:|---:|---:|---:|---:|
| Dummy | 0.0817 | 0.0591 ± 0.0073 | 0.0225 | N/A | 3.6 / 12.2 |
| Regla estacional | 0.1030 | 0.0750 ± 0.0166 | 0.0279 | N/A | 0.0 / 0.0 |
| Regresión logística | 0.1813 | 0.1411 ± 0.0367 | 0.0401 | N/A | 6.9 / 23.5 |
| Random Forest | 0.4284 | 0.1601 ± 0.0341 | **0.2682** | N/A | 50.3 / 136.3 |
| HistGradientBoosting | 0.2591 | 0.1805 ± 0.0404 | 0.0787 | 0.08 | 12.1 / 37.3 |
| MLP | 0.2301 | 0.1658 ± 0.0294 | 0.0643 | 0.001 | 12.6 / 36.4 |
| CatBoost | 0.2416 | 0.1780 ± 0.0398 | 0.0636 | 0.08 | 21.7 / 64.9 |

### Espacios de búsqueda del tuning

`ParameterSampler` eligió 30 configuraciones reproducibles por familia con semilla 42. Las tasas de aprendizaje
y regularizaciones se muestrearon en escala logarítmica porque diferencias multiplicativas son más informativas.

| Familia | Espacio principal | Escala logarítmica |
|---|---|---|
| HistGradientBoosting | learning rate 0.01–0.224; 60–160 iteraciones; 15–63 hojas; mínimo 15–100 observaciones; L2 0.001–10 | learning rate y L2 |
| CatBoost | learning rate 0.01–0.316; 60–160 iteraciones; profundidad 4–8; L2 0.1–19.95; random strength 0–2 | learning rate y L2 |
| MLP | learning rate 0.0001–0.01; alpha 1e-5–0.1; capas entre 16 y 48/24; batch 256–1024 | learning rate y alpha |

## Resultados obtenidos

- Se registraron 105 corridas en el experimento estacional; incluye las 90 configuraciones válidas y dos
  corridas padre adicionales causadas por la pausa y recuperación segura del entrenamiento.
- HistGradientBoosting fue el *champion*: PR-AUC temporal `0.1838`, PR-AUC ponderado `0.2316`,
  PR-AUC de feriados `0.5405` y PR-AUC `0.2693` en holdout.
- CatBoost fue el *challenger*: PR-AUC temporal `0.1787`, PR-AUC ponderado `0.2276`
  y PR-AUC `0.2394` en holdout.
- La diferencia ponderada de CV fue `0.0039`, menor que la desviación de ambos modelos (`0.0603` y `0.0578`);
  por eso la elección también considera el holdout y la reducción de falsos negativos, no solo el ranking medio.
- El umbral del champion es `0.35`. Se eligió únicamente con predicciones fuera de muestra del entrenamiento,
  usando costo `15*FN + FP` y la meta de negocio.
- El champion alcanzó recall `48.38%`, accuracy `82.74%` y redujo `36.27%` los picos omitidos
  frente a la regla estacional; supera la meta de `10%`.
- Al unir las predicciones OOF de los backtests, el champion obtuvo PR-AUC `0.5499` en feriados, `0.5285`
  en fin de año y `0.4682` en Carnaval/Semana Santa. El `0.5405` anterior es el promedio de los cuatro pliegues.
- Sobre las 40.320 filas del holdout, el ajuste final y la inferencia tardaron `21.30 s` y `0.576 s` para
  HistGradientBoosting, frente a `14.78 s` y `0.572 s` para CatBoost. La diferencia operativa es pequeña;
  ambos requieren explicaciones posprocesadas, por ejemplo importancia por permutación o SHAP.
- La sensibilidad a `t+7` obtuvo PR-AUC `0.2634`, por lo que el horizonte más largo sigue siendo posible,
  aunque se mantiene `t+3` como caso principal.
- El modelo se recargó desde `models:/ecu911-alta-demanda-modelo@champion` y reprodujo exactamente
  el PR-AUC del holdout (`0.2692505238`).
- El registro local deja al champion en la versión `3` (`run_id=b94b74256a114988b926217055c66134`) y al
  challenger en la versión `4` (`run_id=07a10eaa99a74a6dafc8b8b4800b0f26`).

Los resultados detallados están en `docs/corridas.csv`, `docs/finalistas_test.csv` y el memo técnico.

## Pruebas

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Las siete pruebas revisan disponibilidad temporal de features, calendario conocido, exclusión de columnas futuras,
purga del holdout, separación temporal, ejecución del pipeline y comportamiento de la regla de negocio.

## Estado de la revisión local

- Los cinco notebooks tienen todas sus celdas ejecutadas, salidas guardadas y cero errores.
- Las siete pruebas automatizadas terminan en `OK`.
- `docs/corridas.csv` contiene 105 corridas; los tres CSV de tuning contienen 30 configuraciones cada uno.
- El registro local conserva `champion` en la versión 3 y `challenger` en la versión 4.
- El champion cargado desde el Model Registry reproduce exactamente el PR-AUC `0.2692505238`.
- `.env`, `.mlflow/`, `mlartifacts/`, `cache/` y archivos temporales están excluidos de Git.
