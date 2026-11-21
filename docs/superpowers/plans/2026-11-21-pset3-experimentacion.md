# PSet 3 — Experimentación ECU 911: plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Experimentación completa (EDA → registry) para t+3 y t+7 sobre la OBT del PSet 2, trazada en MLflow, con memo LaTeX.

**Architecture:** `src/` contiene toda la lógica reutilizable (datos, validación, pipeline, bucle de CV + logging MLflow, modelos fundacionales, LLM). Los notebooks solo orquestan y grafican. MLflow corre en Docker (perfil `mlflow`, Postgres + volumen de artefactos). Los datos se leen una vez de Snowflake a un parquet local.

**Tech Stack:** Python 3.11 (uv venv), pandas, pyarrow, scikit-learn, lightgbm (opcional), optuna, mlflow (misma versión cliente/servidor), snowflake-connector-python, timesfm o chronos-forecasting (torch CPU/MPS), ollama (cliente HTTP vía `requests`), matplotlib/seaborn, scipy, pytest, nbconvert, LaTeX.

**Spec:** `docs/superpowers/specs/2026-11-21-pset3-experimentacion-design.md`

## Global Constraints
- Experimento MLflow: `ecu911-demanda-alta`. Modelos registrados: `ecu911-t3-modelo`, `ecu911-t7-modelo`; alias `champion` / `challenger`.
- Semilla global `SEED = 42`. Horizontes `HORIZONTES = (3, 7)`.
- Tags obligatorios en cada corrida: `fase` ∈ {baseline, candidato, tuning, llm, final}, `familia`, `horizonte` ∈ {t3, t7}, `git_commit`.
- Datos: `data/obt_snapshot.parquet` (fuera de Git); tabla de origen `ECU911.OBT.OBT_EMERGENCIAS_CANTON_DIA`.
- Hold-out = últimos 6 meses según fecha de corte; CV = 5 pliegues `TimeSeriesSplit` sobre fechas únicas con `gap = h`.
- P90 por (cantón, dow de t+h) calculado solo con fechas de train del contexto.
- Nunca credenciales en Git. **Nunca atribución a Claude/IA en commits, PRs, archivos ni memo.**
- RAM 8 GB: `n_jobs ≤ 4`, `float32` donde se pueda, RF con `max_samples`.

## Review Focus
- Cantón-días con P90 = 0 (cantones de bajo volumen): `y = n > P90` debe seguir definido; ratio con P90=0 no debe producir inf/NaN → usar `p90 + 1` en el denominador. Test en Task 3.
- Fecha de corte distinta a 2026-08-31 (snapshot real puede diferir): hold-out y pliegues deben derivarse de `max(fecha)`, nunca de fechas fijas. Test en Task 3.
- Filas cuyo t+h cae en el hold-out no deben quedar en train (purga). Test en Task 3.
- Respuestas del LLM no-JSON o fuera de [0,1]: se cuentan como inválidas y no rompen la corrida. Test en Task 6.
- Servidor MLflow caído: los helpers deben fallar con mensaje claro (`MLFLOW_TRACKING_URI`), no colgarse. Cubierto en Task 5 (timeout de conexión).

---

### Task 1: Entorno, Compose de MLflow y limpieza del repo
**Files:** Create `requirements.txt`, `pytest.ini`, `tests/__init__.py`, `src/__init__.py`, `docker/mlflow/Dockerfile`. Modify `docker-compose.yml`, `.env.example`, `.gitignore`. Delete `docs/PSet2_memo_*.pdf`.
- [ ] `requirements.txt` con versiones fijadas; venv `uv venv --python 3.11 .venv && uv pip install -r requirements.txt`.
- [ ] Compose: servicios `mlflow-db` (postgres:16) y `mlflow` (imagen con mlflow + psycopg2, misma versión que el cliente) con `profiles: [mlflow]`; `--backend-store-uri postgresql://…`, `--artifacts-destination /mlartifacts`, `--serve-artifacts`, puerto 5000, volúmenes `mlflow-db-data`, `mlflow-artifacts`. Variables `MLFLOW_DB_PASSWORD` en `.env.example`.
- [ ] Verificar: `docker compose --profile mlflow up -d mlflow` levanta solo `mlflow` y `mlflow-db`; `curl localhost:5000/health` → `OK`.
- [ ] `.gitignore`: `CD_PSet_3.pdf`, `mlruns/`, `.ipynb_checkpoints/`, `/docs/PSet3_memo_*.tex`, `/docs/PSet3_memo_*.aux|log|out`.
- [ ] Commit.

### Task 2: `src/datos.py` — snapshot y tabla de modelado
**Interfaces (Produces):**
- `COLUMNAS_OBT: list[str]` (37 columnas de `build_obt.py`, minúsculas).
- `descargar_snapshot(ruta=RUTA_SNAPSHOT) -> Path` (lee `.env` vía variables de entorno `SNOWFLAKE_*`).
- `cargar_snapshot(ruta=RUTA_SNAPSHOT) -> pd.DataFrame` (fecha como datetime64, cod_canton str).
- `metadatos_dataset(ruta) -> dict` con `tabla_origen`, `fecha_corte` (ISO), `sha256`, `filas`.
- `construir_tabla_modelado(obt: pd.DataFrame, h: int) -> pd.DataFrame`: añade `fecha_obj`, `dow_obj` (1–7), `es_feriado_obj`, `n_obj` (= `obj_n_emergencias_t{h}`), `prom_4_mismo_dia`, `tendencia_7_28`; descarta filas con `n_obj` nulo o features de historia nulas. Solo usa valores con fecha ≤ t.
- `obt_sintetica(n_cantones=6, n_dias=400, seed=42) -> pd.DataFrame` con las 37 columnas (fixture de tests).
- `FEATURES_NUM`, `FEATURES_CAT` (listas usadas por el pipeline).
**Tests (`tests/test_datos.py`):**
- `test_sintetica_tiene_columnas_obt`: `list(df.columns) == COLUMNAS_OBT`.
- `test_prom_4_mismo_dia_sin_futuro`: con h=3, para una fila concreta, `prom_4_mismo_dia == mean(n en t-4,t-11,t-18,t-25)`; con h=7, `mean(n en t, t-7, t-14, t-21)`.
- `test_tabla_sin_nulos_en_features_y_target`.
- `test_metadatos_hash_estable`: hash idéntico en dos llamadas sobre el mismo archivo.
- [ ] Tests fallan → implementar → pasan → commit.

### Task 3: `src/validacion.py` — split, pliegues, P90 y target
**Interfaces (Produces):**
- `separar_holdout(tabla, h, meses=6) -> tuple[pd.DataFrame, pd.DataFrame]` (train, test); inicio de test = primer día del mes `meses` antes del mes de `max(fecha)`+1; purga filas de train con `fecha_obj >= inicio_test`.
- `pliegues_temporales(tabla, h, n_splits=5) -> list[tuple[np.ndarray, np.ndarray]]` (índices posicionales) con `TimeSeriesSplit(n_splits, gap=h)` sobre fechas únicas ordenadas.
- `tabla_p90(train) -> pd.Series` indexada por (`cod_canton`, `dia_semana`) con el cuantil 0.9 de `n_emergencias` de las fechas de train.
- `etiquetar(df, p90) -> np.ndarray[int]`: `y = n_obj > p90[(cod_canton, dow_obj)]`.
**Tests (`tests/test_validacion.py`):**
- `test_holdout_derivado_de_fecha_corte` (fixture que termina en fecha ≠ 2026-08-31).
- `test_purga_target_en_holdout`: `train.fecha_obj.max() < test.fecha.min()`.
- `test_pliegues_misma_fecha_mismo_lado` y `test_pliegues_gap`: `min(fecha val) - max(fecha train) > h` días.
- `test_p90_solo_train`: cambiar valores en validación no cambia `tabla_p90(train)`.
- `test_p90_cero_define_target`: cantón todo ceros → `y` = 1 solo donde n_obj > 0.
- [ ] TDD → commit.

### Task 4: `src/pipeline.py` — preparación dentro del Pipeline
**Interfaces (Produces):**
- `class RatioP90(BaseEstimator, TransformerMixin)`: `fit(X)` aprende P90 por (cod_canton, dia_semana) desde columnas `cod_canton`, `dia_semana`, `n_emergencias` de X; `transform` añade `ratio_nivel_p90 = prom_4_mismo_dia / (p90(cod_canton, dow_obj) + 1)` (cantón/dow no visto → mediana global).
- `construir_preprocesador(escalar: bool) -> ColumnTransformer` (imputación mediana, One-Hot `region`, dow cíclico, `StandardScaler` si `escalar`).
- `construir_pipeline(modelo, escalar: bool) -> Pipeline` = `RatioP90` → preprocesador → modelo.
- `MODELOS_DEFECTO: dict[str, tuple[estimator, bool]]` para familias `lineal` (LogisticRegression max_iter=1000), `bagging` (RF n_estimators=200, max_samples=0.3, n_jobs=4), `boosting` (HistGradientBoosting), `red_neuronal` (MLPClassifier early_stopping=True).
**Tests:** `test_ratio_no_inf_con_p90_cero`, `test_pipeline_fit_predict_proba` (sintética, salida en [0,1]), `test_preprocesador_ajusta_solo_train` (stats del scaler cambian solo con train).
- [ ] TDD → commit.

### Task 5: `src/experimento.py` — CV, métricas, umbral y MLflow
**Interfaces (Produces):**
- `evaluar_cv(fabrica: Callable[[], Pipeline], tabla, h, pliegues) -> ResultadoCV` (dataclass: `pr_auc_folds: list[float]`, `pr_auc_train: float`, `segundos: float`, `oof: pd.DataFrame` con `idx, y, score`). En cada pliegue: P90 de train del pliegue → etiquetas → fit → score.
- `evaluar_regla(score_fn, tabla, h, pliegues) -> ResultadoCV` para baselines y modelos zero-shot (sin fit).
- `umbral_negocio(oof_modelo, precision_min) -> float`; `metricas_negocio(y, score, umbral) -> dict` (`fn`, `tp`, `fp`, `precision`, `recall`).
- `registrar_corrida(nombre, fase, familia, h, params, resultado, meta, pipeline=None, X_ejemplo=None, nested=False) -> str` (run_id): params + `seed`; meta del dataset; métricas `pr_auc_train`, `pr_auc_cv_mean`, `pr_auc_cv_std`, `brecha`, `segundos_entrenamiento`; tags incl. `git_commit` (`git rev-parse HEAD`); `mlflow.sklearn.log_model` con `infer_signature` e `input_example`.
- `configurar_mlflow(uri=os.getenv("MLFLOW_TRACKING_URI","http://localhost:5000"))`: verifica `/health` con timeout 5 s y lanza `RuntimeError` claro si no responde.
**Tests:** con `mlflow.set_tracking_uri(tmp_path.as_uri())`: `test_evaluar_cv_devuelve_5_folds`, `test_registrar_corrida_tags_y_metricas`, `test_umbral_respeta_precision_min`, `test_configurar_mlflow_error_claro` (puerto cerrado).
- [ ] TDD → commit.

### Task 6: `src/fundacional.py` y `src/llm.py`
**Interfaces:**
- `pronosticar_fundacional(obt, fechas_origen, contexto=128, pasos=7) -> pd.DataFrame` (`cod_canton, fecha, pred_t3, pred_t7`); intenta TimesFM, si falla importación o memoria usa Chronos-Bolt Small; devuelve también `modelo_usado`.
- `construir_prompt(fila_historia: dict) -> str`; `parsear_respuesta(texto) -> float | None` (JSON `{"probabilidad": x}`, x∈[0,1], si no → None); `predecir_llm(df, modelo="qwen2.5:3b", url="http://localhost:11434") -> pd.DataFrame` (score, latencia, valido).
**Tests:** `test_parsear_respuesta_invalida` (texto libre, x=1.5, JSON roto → None), `test_parsear_respuesta_valida`, `test_prompt_no_incluye_target` (no contiene `n_obj`).
- [ ] TDD → commit. Instalar Ollama (`brew install ollama`), `ollama pull qwen2.5:3b`; smoke test de 3 predicciones.

### Task 7: Datos reales (bloqueado por el equipo)
- [ ] Pedir al equipo la carga en Snowflake (OBT 422,912 filas aprox.) y el `.env` completo.
- [ ] `python -c "from src.datos import descargar_snapshot; descargar_snapshot()"`; verificar filas, fecha de corte y hash.

### Task 8: Notebooks 01–02 (EDA y features)
- [ ] `01_eda.ipynb`: tamaño/tipos/nulos, describe, distribución del target por h (prevalencia), IQR + boxplots y Z≥3, Spearman (justificado por colas pesadas/conteos), heatmap, P85/P90/P95 y prevalencia por año, estratos por volumen, ≥3 hallazgos en tabla (Hallazgo | Evidencia | Consecuencia). Figuras → `docs/figuras/`.
- [ ] `02_features.ipynb`: evidencia por feature derivada (Spearman con target, PR-AUC univariada en CV), prueba de no-fuga (recalcular feature truncando datos posteriores a t y comparar igualdad), codificación.
- [ ] Ejecutar con `jupyter nbconvert --execute --inplace`; commit.

### Task 9: Notebook 03 (baselines, candidatos, fundacional, LLM)
- [ ] Por h: 2 baselines, 4 familias sklearn, fundacional (submuestreo 1/7 fechas origen, tabla comparativa sobre el mismo subconjunto), LLM (≈400 filas del último pliegue, mismos modelos re-puntuados, bootstrap IC). Tabla train/CV/brecha/tiempo + gráficos. Top-3 por horizonte guardados en `docs/top3_t{h}.json`.
- [ ] Ejecutar y commit.

### Task 10: Notebook 04 (tuning)
- [ ] Optuna TPE seed 42, 30 trials por modelo top-3 y por h; corrida padre por modelo + hijas `nested=True`; espacios justificados en markdown. Mejora vs. defecto con regla "mejora < std ⇒ no cuenta". Gráficos (historia de optimización, importancia de hiperparámetros). Mejores params → `docs/mejores_params_t{h}.json`.
- [ ] Ejecutar (permite reanudar con `optuna` storage SQLite en `data/`), commit.

### Task 11: Notebook 05 (selección, registry, reproducción) y export
- [ ] Por h: comparación ganador vs. segundo (media±std), reentreno en train completo, test una vez (`fase=final`), umbral de negocio con OOF, FN vs. baseline de negocio, tiempos de inferencia, importancia por permutación; registrar versiones y alias.
- [ ] Sección final "notebook limpio": nuevo kernel cargando `models:/ecu911-t{h}-modelo@champion` y reproduciendo la métrica registrada (`assert abs(diff) < 1e-12`). Si conviene, notebook separado `06_reproduccion.ipynb`.
- [ ] `mlflow.search_runs` → `docs/corridas.csv`. Commit.

### Task 12: README y memo
- [ ] README reescrito (carga de datos PSet 2 resumida, MLflow, Ollama, venv, `.env`, orden de notebooks).
- [ ] Memo LaTeX ≤ 6 páginas → `docs/PSet3_memo_Tierra_Rodriguez_Rodriguez_Cadena_Jarrin.pdf`, captura de la UI de MLflow incluida. Commit y push.
