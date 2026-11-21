# PSet 3 — Experimentación y selección del modelo (ECU 911): diseño

Fecha: 2026-11-21 · Repo: `pset_3` (base copiada de `Pset_2`, commit `443ac51`)
Fuentes: enunciado `CD_PSet_3.pdf`, memo del PSet 1 (`PSet1_memo_*.pdf`), README y `docs/modelo_dimensional.md` del PSet 2.

Convención: lo que decidió el equipo va sin marca. Lo marcado **[decisión del diseñador]** lo propuse yo y el equipo lo aceptó en bloque; el builder puede ajustarlo si la evidencia lo pide, documentándolo.

---

## 1. Intención

Encontrar, con evidencia reproducible, el mejor modelo para anticipar **días de demanda inusualmente alta de emergencias por cantón** en dos horizontes (t+3 y t+7), y dejarlo registrado en MLflow listo para el PSet 4.
El enunciado califica **la calidad de la evidencia, no la métrica más alta**: un modelo modesto bien evaluado vale más que uno brillante sin fundamento. Toda decisión de diseño se subordina a eso.

Flujo exigido: OBT → EDA → Features → Pipeline → Línea base → Candidatos → Tuning → Selección → Model Registry, **completo para cada horizonte**.

## 2. Problema (heredado del PSet 1, no se rediscute)

| Elemento | Definición |
|---|---|
| Unidad | cantón × día de ejecución t; se predice para t+h, h ∈ {3, 7} |
| Target | `y_h = 1` si `obj_n_emergencias_t{h}` (volumen real en t+h) > P90 del cantón para el **día de la semana de t+h**; 0 si no |
| P90 | calculado **solo con fechas de entrenamiento** (ver §5.2 para cómo se recalcula por pliegue) |
| Métrica técnica | PR-AUC (`average_precision`); secundaria: Recall con precisión mínima |
| Línea base de negocio | promedio de las últimas 4 ocurrencias del mismo día de la semana que t+h; alerta si supera el P90 |
| Métrica de negocio | reducción de días de alta demanda **no anticipados** (FN) frente a la línea base de negocio; objetivo ≥ 10 %, al menos en t+3 |
| Umbral inicial | 0.70, se ajusta en validación (ver §6) |
| Acción | alerta para revisión humana; el modelo no asigna recursos |

## 3. Metas, no-metas y anti-metas

**Metas**
- Cumplir todo el enunciado del PSet 3 (EDA, features, pipeline sin fuga, 2 líneas base, 5 familias, LLM por prompting, tuning de los 3 mejores con ≥ 30 configuraciones, selección, registry, `docs/corridas.csv`, README, memo ≤ 6 páginas) **para t+3 y para t+7**.
- Que cada corrida sea reproducible desde el repo + el snapshot de datos.

**No-metas**
- Desplegar el modelo (eso es el PSet 4).
- Mejorar el pipeline del PSet 2 (dbt/Kestra/Spark) salvo que algo bloquee; se reutiliza tal cual para cargar datos.
- Población INEC u otras fuentes nuevas (queda como limitación, igual que en el PSet 2).

**Anti-metas** (si pasa esto, fracasó aunque "funcione")
- Cualquier fuga temporal: features que usen información posterior a t, P90 calculado con datos de validación/prueba, pliegues que mezclen el mismo día entre train y validación.
- Mirar el hold-out antes de la selección final o usarlo más de una vez.
- Credenciales, tokens o `.env` en Git.
- **Ninguna atribución a Claude/IA como coautor en commits, PRs, archivos ni memo.**
- Copiar la lógica de preparación dentro de los notebooks en vez de importarla de `src/`.

## 4. Restricciones de entorno

- **Mac M3 con 8 GB de RAM.** Es la restricción dominante.
  - El pipeline del PSet 2 (Kestra + Postgres + Spark) se corre **una sola vez** para cargar datos y luego se apaga (`docker compose stop`).
  - MLflow va en un **perfil de Compose separado** (`--profile mlflow`) para levantar solo MLflow + su backend (~300–500 MB).
  - Modelos pesados con límites de memoria (ver §7).
- **Snowflake:** la cuenta anterior caducó. El equipo crea una nueva, completa `.env`, y corre `ecu911.run_all` (`desde=2021-07`, `hasta=2026-08`, `setup=true`). Resultado esperado: OBT con 422,912 filas.
- **Python local:** el sistema tiene Python 3.14, demasiado nuevo para TimesFM/torch. Usar un entorno virtual con **Python 3.11** (p. ej. `uv venv --python 3.11`) **[decisión del diseñador]**. Versiones fijadas en `requirements.txt`.
- **Ollama** se instala **nativo en macOS** (no en Docker: en Docker para Mac no usa la GPU). Modelo de ~3B parámetros (`qwen2.5:3b` por defecto; `llama3.2:3b` como alternativa) **[decisión del diseñador]**.
- Sin API keys de pago: no se usa la API de Anthropic ni otra.

## 5. Datos, target y validación

### 5.1 Acceso a datos — `src/datos.py`
- Lee `ECU911.OBT.OBT_EMERGENCIAS_CANTON_DIA` desde Snowflake con credenciales del `.env` (nunca hardcodeadas) y guarda **un snapshot** `data/obt_snapshot.parquet` (carpeta `data/` ya está en `.gitignore`).
- **Orden [decisión del diseñador]:** mientras no exista el snapshot, `src/` y su infraestructura (MLflow en Compose, CV, logging) se construyen y prueban con un **fixture sintético pequeño** con las 37 columnas exactas de la OBT (`build_obt.py`). Los notebooks solo se ejecutan con los datos reales.
- Todos los notebooks leen del snapshot, no de Snowflake. Así la experimentación no depende de que la cuenta siga viva.
- Expone metadatos del dataset para MLflow: tabla de origen, **fecha de corte** (`max(fecha)` del snapshot) y **hash SHA-256** del parquet.
- Construye, por horizonte h, la tabla de modelado: target crudo `obj_n_emergencias_t{h}`, día de la semana de t+h, `es_feriado_t{h}`, y las features de §5.4.
- **Contrato de entrada del modelo [decisión del diseñador]:** las features que necesitan historia (p. ej. `prom_4_mismo_dia`, que con h=3 usa los conteos de t−4, t−11, t−18, t−25) se construyen en `datos.py` con una función pública `construir_tabla_modelado(obt, h)` que solo usa días ≤ t. El modelo registrado recibe **filas de esa tabla** (no filas crudas de la OBT); esa es su firma e `input_example`. El PSet 4 reutiliza la misma función para servir. Lo que depende de estadísticas de entrenamiento (P90, codificaciones) sí vive dentro del Pipeline.
- Filas excluidas: las primeras ~28 días de cada cantón (lags/medias nulos) y los últimos h días (sin target). Se documenta cuántas.

### 5.2 Target y P90 sin fuga
- El P90 por (cantón, día de semana) se calcula sobre `n_emergencias` de las **fechas de entrenamiento** del contexto en que se evalúa:
  - en cada pliegue de CV, con las fechas de train de ese pliegue;
  - en la evaluación final, con todo el conjunto de entrenamiento.
- Por eso la CV es un **bucle propio** en `src/` (no `cross_val_score` directo): en cada pliegue recalcula el umbral, etiqueta train y validación con ese umbral, ajusta el pipeline y puntúa. **[decisión del diseñador]**
- EDA debe revisar si P90 es razonable (comparar P85/P90/P95 y prevalencia por año), como pide la matriz de riesgos del PSet 1.

### 5.3 División y esquema de CV
- **Hold-out temporal:** los últimos 6 meses de datos (2026-03-01 → 2026-08-31, ajustado a la fecha de corte real) **[decisión del diseñador]**. Se usa una sola vez, en `05_seleccion`, con ganador y segundo.
- **Purga:** se eliminan del entrenamiento las filas cuyo t+h cae dentro del hold-out (si no, el target de train "ve" el periodo de prueba).
- **CV:** `TimeSeriesSplit` de 5 pliegues **sobre fechas únicas** (todas las filas de un mismo día caen en el mismo pliegue), con `gap = h` días para purgar el solapamiento del target. Justificación: es una serie temporal con fuerte dependencia entre días y entre cantones de la misma fecha; un split aleatorio o por grupo filtraría el futuro.
- La misma CV (mismos pliegues, misma semilla 42) para todos los modelos, líneas base y tuning de un horizonte.

### 5.4 Features
Base (ya en la OBT, todas disponibles al cierre de t): `lag_1d/7d/14d/28d`, `media_prev_7d/28d`, `n_emergencias` y desgloses `n_*` del día t, calendario de t y de t+h, `es_feriado_t{h}`, `region`, `cod_provincia`, `es_periodo_anomalo`.
Excluidas siempre: `obj_n_emergencias_t3/t7` (son el target) y cualquier columna del futuro.

Derivadas (mínimo 2; al menos una motivada por un hallazgo de la EDA) **[decisión del diseñador; el builder las confirma o cambia según lo que muestre la EDA]**:
1. **`prom_4_mismo_dia`**: promedio de las 4 ocurrencias anteriores del día de la semana de t+h (valores en t+h−7, −14, −21, −28; todas ≤ t para h ∈ {3,7}). Es la señal de la línea base de negocio convertida en feature.
2. **`ratio_nivel_p90`**: `prom_4_mismo_dia / P90(cantón, dow de t+h)`. Normaliza la escala entre cantones de volúmenes muy distintos (hallazgo esperado de la EDA: heterogeneidad extrema de escala). El P90 lo calcula un **transformador sklearn propio** que aprende la tabla (cantón, dow) → P90 en `fit` (solo train) y la aplica en `transform`, de modo que vive dentro del Pipeline.
3. **`tendencia_7_28`**: `media_prev_7d / media_prev_28d` (aceleración reciente).

Para cada feature, el notebook `02_features` muestra: evidencia de que aporta (relación con el target, Spearman, o ganancia en CV), por qué existe en el momento de inferencia, y su codificación.
- Categóricas: `region` (5 valores) → One-Hot; `cod_canton` (224) → **no** One-Hot denso; usar target/ordinal encoding ajustado en train o dejarlo fuera y confiar en `ratio_nivel_p90` (decide el builder con evidencia). Día de la semana → cíclico (seno/coseno) u One-Hot.

### 5.5 Pipeline — `src/pipeline.py`
- `Pipeline` + `ColumnTransformer` de scikit-learn: imputación, codificación, escalado (este último solo donde importa: lineal y MLP) y el transformador de P90.
- Ajustado **solo con datos de entrenamiento** del pliegue o del conjunto de entrenamiento.
- Los notebooks **importan** de `src/`; no copian código de preparación.
- `src/` también contiene el bucle de CV de §5.2 y helpers de logging a MLflow, para que baseline, candidatos y tuning registren exactamente lo mismo.

## 6. Métricas y umbral

- **Técnica (igual para todos):** PR-AUC en validación (media y desviación por pliegue) y en entrenamiento.
- **Secundaria:** Recall a precisión mínima.
- **Negocio:** número de días de alta demanda no anticipados (FN) frente a la línea base de negocio, al mismo nivel de precisión. **[decisión del diseñador]** Regla del umbral: elegir, con las predicciones fuera de pliegue de la CV (nunca con el hold-out), el umbral que **maximiza Recall con precisión ≥ la precisión de la línea base de negocio**. Así la comparación de FN es justa (no se "compran" aciertos con alertas basura). Se reporta también qué pasa con el umbral inicial 0.70 del PSet 1.
- Si en el experimento otra métrica resulta mejor, se cambia y se justifica (lo permite el enunciado).

## 7. Experimentos (por cada horizonte)

### 7.1 Líneas base (2 corridas por horizonte)
1. **Ingenua:** `DummyClassifier(strategy="prior")` (o `stratified`).
2. **De negocio:** regla del PSet 1; score = `prom_4_mismo_dia / P90` para poder calcular PR-AUC; alerta si > 1.

### 7.2 Candidatos — 5 familias con hiperparámetros por defecto (5 corridas por horizonte)
| Familia | Modelo |
|---|---|
| Lineal regularizado | `LogisticRegression` |
| Bagging | `RandomForestClassifier` (con `max_samples`, `n_jobs` bajo por RAM) |
| Boosting | `HistGradientBoostingClassifier` (o LightGBM) |
| Red neuronal | `MLPClassifier` (sobre una muestra si no cabe en memoria; documentarlo) |
| State of the Art | **TimesFM** (modelo fundacional de series de tiempo, el más pequeño ~200M, CPU/MPS, por lotes de cantones): pronostica el conteo en t+h; score = pronóstico / P90. Zero-shot, evaluado en los mismos pliegues. **Plan B si no cabe en 8 GB o es inviable en tiempo: Chronos-Bolt Small**, documentado como limitación. |

**Cobertura de TimesFM [decisión del diseñador]:** pronosticar desde cada fecha de cada pliegue para 224 cantones es inviable en 8 GB/CPU. TimesFM se evalúa sobre un **submuestreo de fechas de origen (1 de cada 7) en los 5 pliegues, todos los cantones**; un solo pronóstico de 7 pasos sirve para t+3 y t+7. Para compararlo en igualdad, los modelos tabulares también se re-puntúan sobre ese mismo subconjunto y se reporta en una tabla aparte; el ranking para elegir los 3 a tunear usa esa comparación para TimesFM. Si aun así no corre en un tiempo razonable (orientativo: > 2 h por horizonte), se pasa a Chronos-Bolt Small.

Para cada uno: PR-AUC en train, en CV (media ± desviación), brecha train−CV, tiempo de entrenamiento; diagnóstico de sobreajuste. Todo candidato debe superar a **ambas** líneas base; si no, se documenta como hallazgo.
Todas las familias usan la misma preparación, la misma CV y la misma métrica. TimesFM no se ajusta (zero-shot); sus "segundos de entrenamiento" son 0 y se reporta el tiempo de inferencia.

### 7.3 LLM por prompting (Ollama local)
- Modelo `qwen2.5:3b` en Ollama nativo. Prompt con: últimas 4 semanas de conteos del cantón, P90 del cantón y día de semana, feriado en t+h, día de semana. Salida JSON estructurada con una probabilidad.
- Se evalúa sobre una **muestra** de cantón-días del último pliegue de validación (~400 por horizonte, prevalencia natural, semilla fija) **[decisión del diseñador]**. Los demás modelos se re-puntúan sobre **la misma muestra** para comparar en igualdad, con intervalo bootstrap.
- Se registra en MLflow (tag `fase=llm`): modelo, plantilla del prompt, tamaño de muestra, PR-AUC, tiempo por predicción, % de respuestas inválidas. Las predicciones van como artefacto CSV.
- Resultado esperado: peor que los tabulares; vale como hallazgo (costo/latencia vs. valor).

### 7.4 Tuning — 3 mejores candidatos por horizonte (≥ 90 corridas por horizonte)
- **Optuna** (TPE, semilla fija), 30 configuraciones por modelo **[decisión del diseñador]**. Justificación: el bucle de CV propio (P90 por pliegue) no encaja con `RandomizedSearchCV`; Optuna permite pausar/retomar (útil con 8 GB y horas de cómputo) y concentra pruebas en zonas prometedoras.
- Espacio de búsqueda definido y justificado por modelo (p. ej., `learning_rate` y `C` en escala logarítmica; `max_depth`/`max_leaf_nodes` enteros; `min_samples_leaf`, regularización L2).
- Misma CV que los candidatos. Cada configuración = **corrida hija** (nested) dentro de una **corrida padre por modelo**.
- Si TimesFM queda entre los 3 mejores, no tiene hiperparámetros entrenables significativos; se reemplaza por el 4.º y se documenta. **[decisión del diseñador]**
- Mejora reportada vs. versión por defecto; **si la mejora < desviación entre pliegues, no cuenta como mejora.**

### 7.5 Selección y registro (por horizonte)
1. Comparar ganador vs. segundo: media y desviación de CV; ¿la diferencia supera la variación entre pliegues?
2. Reentrenar ambos en todo el entrenamiento y evaluar **una sola vez** en el hold-out (corridas `fase=final`).
3. Métrica de negocio con el umbral elegido en §6.
4. Costo y explicabilidad: tiempo de entrenamiento e inferencia; importancia de features / SHAP o permutación.
5. Registrar en el Model Registry:
   - **`ecu911-t3-modelo`** y **`ecu911-t7-modelo`**, cada uno con alias `champion` (ganador) y `challenger` (segundo). Dos nombres porque un nombre solo admite un `champion`; el memo justifica la desviación del `<proyecto>-modelo` del enunciado.
6. En un notebook limpio, cargar `models:/ecu911-t{h}-modelo@champion` y **reproducir exactamente** la métrica de prueba registrada.

### 7.6 Presupuesto mínimo de corridas
Por horizonte: 2 baseline + 5 candidatos + 90 tuning (+ 3 padres) + 1 LLM + 2 final ≈ 103. **Total ≈ 206.**

## 8. MLflow

- Servidor en `docker-compose.yml` bajo el perfil `mlflow`: contenedor MLflow + **Postgres propio** (backend store) + **volumen** como artifact store servido por el propio servidor (`--serve-artifacts`, para que el cliente en el host suba artefactos por HTTP). Puerto 5000. Misma versión de MLflow fijada en servidor y cliente.
- Los servicios del PSet 2 no deben arrancar con el perfil `mlflow` y viceversa (que `docker compose --profile mlflow up -d` levante solo MLflow).
- Experimento: **`ecu911-demanda-alta`**.
- Cada corrida registra:
  - **Parámetros:** hiperparámetros, semilla, horizonte.
  - **Datos:** tabla de origen, fecha de corte, hash del snapshot (`mlflow.log_input` o params).
  - **Métricas:** PR-AUC train, CV media y desviación, brecha, segundos de entrenamiento (+ Recall@precisión y métrica de negocio donde aplique).
  - **Tags:** `fase` (`baseline`, `candidato`, `tuning`, `llm`, `final`), `familia`, `horizonte` (`t3`/`t7`), `git_commit`.
  - **Artefactos:** pipeline entrenado con `mlflow.sklearn.log_model`, firma e `input_example`. TimesFM y el LLM no son pipelines sklearn: registran config + predicciones como artefactos (excepción documentada en el memo).
- Un commit de Git limpio antes de cada fase, para que `git_commit` apunte a código real.
- Al final: `mlflow.search_runs` → **`docs/corridas.csv`** (el profesor no tiene acceso al servidor).

## 9. Repositorio

```
pset_3/
├── docker-compose.yml      # PSet 2 + perfil mlflow
├── .env.example            # + variables de MLflow/Postgres de MLflow
├── requirements.txt        # versiones fijadas, Python 3.11
├── README.md               # reescrito para PSet 3
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_features.ipynb
│   ├── 03_baseline_candidatos.ipynb   # incluye TimesFM y LLM
│   ├── 04_tuning.ipynb
│   └── 05_seleccion.ipynb             # incluye hold-out, registry y recarga del champion
├── src/
│   ├── datos.py
│   └── pipeline.py         # (+ módulos auxiliares en src/ si hace falta)
├── docs/
│   ├── corridas.csv
│   ├── figuras/            # gráficos exportados para el memo
│   └── PSet3_memo_Tierra_Rodriguez_Rodriguez_Cadena_Jarrin.pdf
└── (dbt/, kestra/, spark/, docker/ del PSet 2, intactos)
```

- **README:** cómo cargar datos en la cuenta nueva de Snowflake (resumen del pipeline PSet 2), cómo levantar MLflow, cómo instalar Ollama y el modelo, cómo crear el entorno Python, cómo configurar `.env`, y el orden de ejecución de los notebooks.
- Cada notebook se entrega ejecutado con **Restart → Run All**, con salidas guardadas.
- `.gitignore`: agregar `CD_PSet_3.pdf`, `mlruns/`, `.ipynb_checkpoints/`; mantener `data/` y `PSet1_memo_*`.
- Quitar del repo `docs/PSet2_memo_*.pdf` y `docs/modelo_dimensional.md`/`data_quality.sql` solo si estorban; el memo del PSet 2 vive en su propio repo **[decisión del diseñador: quitar el PDF del PSet 2; mantener la documentación técnica del pipeline]**.

## 10. Memo técnico (≤ 6 páginas + anexos)
Secciones exigidas: EDA, Features, Preparación y validación, Diseño experimental (incluye captura de la comparación en la UI de MLflow), Resultados, Tuning, Decisión (run_id, versión en registry, métrica de prueba, métrica de negocio, por qué gana), Limitaciones. Explica decisiones; no define conceptos vistos en clase. Cubre ambos horizontes. Autores: Jarod Tierra, Felipe Rodríguez, Santiago Rodríguez, Juan Diego Cadena, Miguel Jarrín.
- **Lo escribe el builder** (decisión del equipo), en **LaTeX** compilado con `pdflatex` (mismo estilo que el memo del PSet 1) **[formato: decisión del diseñador]**. Fuente `.tex` local fuera de Git (patrón `/docs/PSet3_memo_*.tex` en `.gitignore`, igual que el PSet 2 hacía con sus borradores); solo el PDF final va al repo. Figuras exportadas desde los notebooks a `docs/figuras/`. La captura de la UI de MLflow la toma el builder con el navegador si hay herramienta disponible; si no, se pide al equipo.

## 10b. Reparto de trabajo (decisión del equipo)
- **El equipo** crea la cuenta nueva de Snowflake y corre la carga del PSet 2. Si el builder necesita credenciales o acceso, lo pide.
- **El builder** hace todo lo demás: código, MLflow, notebooks ejecutados, corridas, registry, `corridas.csv`, README y memo.
- Sin fecha límite declarada; la única máquina es el Mac M3 de 8 GB (Docker con ~4.8 GB asignados).

## 11. Limitaciones previstas (a confirmar con evidencia)
- 8 GB de RAM: muestreo/límites en RF y MLP; LLM de 3B; TimesFM pequeño o reemplazado por Chronos.
- LLM evaluado solo en una muestra.
- Cantones de bajo volumen: P90 muy bajo o 0 → targets ruidosos; reportar métricas por estrato de volumen (pedido en el PSet 1).
- Periodo anómalo 2024-01 marcado en datos.
- Sin población cantonal.

## 12. Lo que queda al builder
- Versiones exactas de librerías y de la imagen de MLflow.
- Espacios de búsqueda concretos por modelo y cuáles 3 modelos se tunean (lo decide la fase de candidatos).
- Codificación final de `cod_canton` y selección final de features, según evidencia.
- Diseño exacto del prompt del LLM y del parseo/reintentos de JSON.
- Organización interna de `src/` más allá de `datos.py` y `pipeline.py`.
- Gráficos y tablas concretos de cada notebook y del memo.
