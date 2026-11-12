# PSet 3 — Experimentación y selección del modelo ECU 911

Tierra · Rodríguez · Rodríguez · Cadena · Jarrín

Continuación del [PSet 2](https://github.com/junilai/Pset_2), commit
`cc4df2dc56d9f17691956ba9c325245726eeba9b`, y de la propuesta del PSet 1:
clasificación de alta demanda cantonal exclusivamente a **t+3**. El alcance
del PSet 3 se limita a ese horizonte por decisión del equipo.

**Estado:** el login y la extracción de Snowflake ya se comprobaron y existen
corridas previas. Esta versión del código se ha adaptado a t+3 **sin ejecutarla**.
Los notebooks tienen sus salidas anteriores limpiadas. El PDF y
`docs/corridas.csv` conservan el historial de la ejecución anterior y todavía
no se han regenerado para el alcance nuevo.

La búsqueda prevista es **3 familias × 30 configuraciones = 90 configuraciones
de tuning**, únicamente para t+3. Las corridas compatibles de t+3 ya terminadas
pueden reutilizarse al reanudar; las de otros horizontes se excluyen del flujo.

## Estructura del enunciado

La entrega contiene únicamente estos 13 archivos:

```text
pset_3/
├── docker-compose.yml
├── .env.example
├── requirements.txt
├── README.md
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_features.ipynb
│   ├── 03_baseline_candidatos.ipynb
│   ├── 04_tuning.ipynb
│   └── 05_seleccion.ipynb
├── src/
│   ├── datos.py
│   └── pipeline.py
└── docs/
    ├── corridas.csv
    └── PSet3_memo_Tierra_Rodriguez_Rodriguez_Cadena_Jarrin.pdf
```

`src/datos.py` contiene acceso a datos, EDA, métricas, experimentación,
tracking y generación del documento. `src/pipeline.py` contiene features,
`Pipeline` y `ColumnTransformer`. Los notebooks importan estas funciones.
Los snapshots, estados, figuras y modelos intermedios se guardan fuera del
repositorio, en `~/.cache/pset_3/`, configurable mediante `PSET3_DATA_DIR`. Los nuevos
intermedios de presentación se guardan en `~/.cache/pset_3/evidencia_t3/`.
Las figuras y tablas se incorporan al PDF; los artefactos MLflow persisten
además en volúmenes Docker. No se generan archivos adicionales de entrega.

## Instalación y MLflow

Requiere Python 3.12 o superior y Docker Desktop. Crear el entorno fuera
de la carpeta de entrega:

```bash
python3 -m venv ../pset_3_entorno
source ../pset_3_entorno/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --prefix ../pset_3_entorno --name python3 --display-name "Python 3 (PSet 3)"
docker compose up -d
docker compose ps
```

En Windows activar `..\pset_3_entorno\Scripts\activate`.
Abrir **http://127.0.0.1:5001**. La primera ejecución instala MLflow 3.18.0
dentro del contenedor y puede tardar unos minutos. No se necesita Dockerfile.
SQLite persiste en `mlflow_backend`; los artefactos en `mlflow_artifacts`.
`docker compose stop` conserva la evidencia; no eliminar los volúmenes
con `down -v`.

## Conexión a Snowflake

Configurar como variables de entorno los valores de `.env.example` o
copiarlo a `.env` privado y completar cuenta, usuario, contraseña, rol y
warehouse. `.env` es configuración local: **no se entrega ni se sube a Git**.
Se excluye mediante `.git/info/exclude`. Para autenticación interactiva,
usar `SNOWFLAKE_AUTHENTICATOR=externalbrowser` cuando corresponda.

```bash
python -m src.datos snapshot
```

Se consulta `ECU911Final.OBT.OBT_EMERGENCIAS_CANTON_DIA` hasta
`SNAPSHOT_END=2026-09-30`. El snapshot guarda tabla, corte, query ID,
conteos y SHA256. Comprueba grain, cobertura diaria, lags, medias y targets.
No sobrescribe snapshots, modifica tablas remotas ni imputa objetivos finales.
El snapshot real obtenido tiene 429.632 filas y 37 columnas. La OBT puede
conservar columnas de otros horizontes; el modelado las excluye y construye
únicamente la etiqueta `obj_n_emergencias_t3 > P90`.

## Orden de ejecución de los notebooks

Seleccionar el kernel del entorno y ejecutar **Restart Kernel** y **Run All**,
guardando las salidas, en este orden:

1. `01_eda.ipynb`: tipos, nulos, descriptivos, target, IQR/Z-score,
   Spearman y hallazgos con evidencia y consecuencia.
2. `02_features.ipynb`: derivaciones, disponibilidad en inferencia y
   preparación importada de `src/pipeline.py`.
3. `03_baseline_candidatos.ipynb`: dos baselines primero, cinco familias
   por defecto, ablación y prueba LLM mediante prompting.
4. `04_tuning.ipynb`: tres mejores familias para t+3, con 30
   configuraciones por familia y nested runs.
5. `05_seleccion.ipynb`: dos finalistas de t+3, test final,
   Registry, reproducción exacta por alias y exportación de corridas.

Alternativa con un kernel nuevo por notebook:

```bash
python -m src.datos notebooks
```

Sin snapshot se guardan mensajes de pendiente. Para completar la entrega
hay que ejecutar de nuevo con datos reales. Revisar la EDA antes de dar
por definitivo el diseño de features.

## Validación y experimentos

El primer año de train calibra el P90 por cantón y weekday, congelado antes
de todos los folds y separado del ajuste de clasificadores. El 20% final de
fechas es hold-out. Tres folds expansivos sobre fechas únicas con gap de siete
días mantienen juntos todos los cantones del mismo día y evitan que las
etiquetas futuras de train alcancen la validación. Se conserva gap=7 como
separación conservadora para t+3 y para mantener los pliegues y el entrenamiento
de las corridas ya realizadas. El hold-out solo exige ahora que y_t3 esté
observado; sus últimas tres fechas sin etiqueta se excluyen.

Métrica técnica: Average Precision (AP, PR-AUC no interpolada). Métrica de
negocio: `(FN_estacional − FN_modelo) / FN_estacional`, con meta del 10%
del PSet 1. Se elige umbral OOF minimizando `5 FN + FP`, sujeto a precision
≥ 0.30: son supuestos provisionales pendientes de validación operativa.
El .70 del memo era un umbral inicial.

Baselines: clase más frecuente y promedio de cuatro weekdays previos > P90.
Candidatos: logística, random forest, hist gradient boosting, MLP y GaussianNB,
con defaults salvo semilla, igual preparación y CV. Se documentan los que no
superan ambas bases. Se ajustan los tres mejores mediante búsqueda aleatoria
reproducible (`ParameterSampler`), registrando cada configuración. Una mejora
que no supera la mayor SD de default/ajustado no se considera demostrada.
Los dos finalistas se fijan en CV antes de abrir test; este no cambia ganador
ni umbral.

Para t+3 se preparan **2 baselines, 5 candidatos y 90 configuraciones de tuning**.
El PDF establece un mínimo de 90 configuraciones (3 modelos × 30). Aquí se
registra una corrida final por cada finalista: 2 finales, para un total de 99
corridas evaluadas, más 3 padres de tuning, 1 ablación y 1 prueba LLM. Registrar
ambos finalistas por separado no duplica la búsqueda. El entrenamiento completo
puede tardar horas en CPU. Cuando se decida reanudar, se pueden ejecutar fases
desde terminal:

```bash
python -m src.datos eda
python -m src.datos candidatos
python -m src.datos ablacion
python -m src.datos llm
python -m src.datos tuning
python -m src.datos seleccion
python -m src.datos verificar
```

Los estados completados se leen de caché sin repetir selección. La transición
a t+3 incluye compatibilidad explícita con los checkpoints anteriores de ese
horizonte: se comprueban snapshot, split, contexto y una huella del código que
determina el entrenamiento. Al reanudar tuning se reabre el padre original
por familia y se omiten sus configuraciones terminadas. Las corridas históricas
se conservan con su procedencia original; no se convierten registros t+7 en t+3.

Los cambios científicos posteriores invalidan esa compatibilidad y requieren
un protocolo/directorio nuevo; el código bloquea la reutilización silenciosa.
Se conservan los rangos de búsqueda existentes. Versionar el código antes de
reanudar. No se ha probado esta adaptación ejecutando entrenamientos.

## LLM, Registry y documento

Se usa **qwen2.5:1.5b mediante Ollama local**, sin API key. Mantener Ollama
abierto; el código y `.env.example` incluyen esta configuración por defecto:

```text
LLM_BASE_URL=http://127.0.0.1:11434/v1
LLM_API_KEY=
LLM_MODEL=qwen2.5:1.5b
LLM_MAX_ROWS=100
```

La prueba zero-shot usa una muestra
del último fold, sin etiquetas futuras, nombres geográficos ni fechas exactas.
Se compara contra OOF de las mismas filas. Son 100 filas de validación de t+3;
la evaluación requiere primero el snapshot real y las corridas de candidatos.

El registro `ecu911-alta-demanda-t3-modelo` contendrá los aliases
`champion` y `challenger` de los dos finalistas de t+3.
El notebook final carga por alias y exige igualdad exacta de predicciones
y AP respecto de MLflow.

```bash
python -m playwright install chromium
python -m src.datos captura
python -m src.datos report
```

La captura real queda en la caché y se incorpora al PDF. El documento sigue
los ocho apartados del enunciado en su orden: EDA; Features; Preparación y
validación; Diseño experimental; Resultados; Tuning; Decisión; Limitaciones.
Máximo seis páginas sin anexos. `docs/corridas.csv` se exporta mediante
`mlflow.search_runs`.

## Limitaciones y pendientes

La fuente mensual no demuestra disponibilidad diaria de n_t o lags; la
evaluación es retrospectiva y condicionada a una fuente oportuna. Se conserva
enero 2024 como anomalía y se excluye su marca retrospectiva de las features.
No hay población INEC ni capacidad operativa integradas; la geografía incluye
reasignaciones retrospectivas y el P90 congelado puede perder vigencia.
No se afirma reducir tiempos de respuesta ni costos monetarios.

Falta completar el tuning de t+3, la evaluación final y el registro de los
finalistas, y volver a guardar las salidas de los notebooks y regenerar PDF/CSV.
Las credenciales, el snapshot y las corridas históricas se conservan. Nada se
reanuda automáticamente por editar estos archivos.
