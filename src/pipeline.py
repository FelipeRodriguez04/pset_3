"""
Preparación, validación, métricas y registro en MLflow (compartido por todos los notebooks).

Pipeline de cada modelo (se ajusta SOLO con datos de entrenamiento):
    PerfilCantonal  -> aprende el P90 por (cantón, día de la semana) y crea features relativas
    ColumnTransformer
        numéricas  -> SimpleImputer(mediana) + StandardScaler
        booleanas  -> SimpleImputer(más frecuente)
        categóricas-> OneHotEncoder(handle_unknown="ignore")
    estimador

Validación: SplitTemporal (5 pliegues de ventana creciente sobre fechas, con 3 días de hueco).
En cada pliegue el umbral P90 del target se recalcula con el pliegue de entrenamiento.
"""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.datos import HORIZONTE, PERCENTIL, RAIZ, etiquetar, umbrales_p90

SEMILLA = 42
EXPERIMENTO = "ecu911-alta-demanda"
NOMBRE_MODELO = "ecu911-modelo"
TARGET = "y_t3"

# -----------------------------------------------------------------------------
# Columnas de entrada del modelo (todas disponibles al cierre del día t)
# -----------------------------------------------------------------------------
CONTEOS = ["n_emergencias", "lag_1d", "lag_7d", "lag_14d", "lag_28d",
           "media_prev_7d", "media_prev_28d", "media_mismo_dia_4s", "media_anio_previo"]
ENTRADA = (["cod_canton", "dia_semana", "dia_semana_obj", "mes_obj", "region",
            "es_feriado", "es_feriado_t3", "es_fin_de_semana_obj"]
           + CONTEOS)

# Columnas que salen de PerfilCantonal y entran al ColumnTransformer
RELATIVAS = [f"rel_{c}" for c in CONTEOS]
NUMERICAS = RELATIVAS + ["log_p90_obj"]
BOOLEANAS = ["es_feriado", "es_feriado_t3", "es_fin_de_semana_obj"]
CATEGORICAS = ["region", "dia_semana_obj", "mes_obj"]


class PerfilCantonal(BaseEstimator, TransformerMixin):
    """Normaliza el volumen de cada cantón por su propio umbral de alta demanda.

    fit : P90 de n_emergencias por (cantón, día de la semana), solo con las filas de fit.
    transform : p90_obj = P90 del cantón para el día de la semana de t+3 y
                rel_<x> = x / max(p90_obj, 1) para cada conteo (lags, medias, día t).
    Así un mismo valor (p. ej. rel = 1.2) significa "20% sobre su umbral" en Quito
    y en un cantón con 1 emergencia diaria; el código de cantón no se usa como feature.
    """

    def __init__(self, percentil: float = PERCENTIL, conteos: tuple = tuple(CONTEOS)):
        self.percentil = percentil
        self.conteos = conteos

    def fit(self, X: pd.DataFrame, y=None):
        self.umbrales_ = umbrales_p90(X, self.percentil)
        # Respaldo para un cantón que no estuvo en el entrenamiento: P90 mediano global
        self.p90_global_ = float(self.umbrales_.median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        u = self.umbrales_.rename_axis(["cod_canton", "dia_semana_obj"]).rename("p90_obj")
        claves = pd.MultiIndex.from_frame(X[["cod_canton", "dia_semana_obj"]])
        p90 = u.reindex(claves).to_numpy(dtype=float)
        p90 = np.where(np.isnan(p90), self.p90_global_, p90)
        out = X.copy()
        out["p90_obj"] = p90
        out["log_p90_obj"] = np.log1p(p90)
        base = np.maximum(p90, 1.0)
        for c in self.conteos:
            out[f"rel_{c}"] = X[c].to_numpy(dtype=float) / base
        for c in out.columns:                    # booleanas -> 0/1 numérico para el imputador
            if out[c].dtype == bool:
                out[c] = out[c].astype(float)
        return out


def crear_preprocesador(numericas=None, booleanas=None, categoricas=None) -> ColumnTransformer:
    num = Pipeline([("imputar", SimpleImputer(strategy="median")),
                    ("escalar", StandardScaler())])
    boo = SimpleImputer(strategy="most_frequent")
    cat = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    bloques = [("num", num, list(numericas if numericas is not None else NUMERICAS)),
               ("bool", boo, list(booleanas if booleanas is not None else BOOLEANAS)),
               ("cat", cat, list(categoricas if categoricas is not None else CATEGORICAS))]
    return ColumnTransformer([b for b in bloques if b[2]], remainder="drop",
                             verbose_feature_names_out=False)


def crear_pipeline(estimador, conteos=None, **columnas) -> Pipeline:
    """Misma preparación para todos los modelos; solo cambia el estimador.

    `conteos` y `columnas` (numericas / booleanas / categoricas) solo se cambian en el
    análisis de ablación de features (notebook 02).
    """
    return Pipeline([("perfil", PerfilCantonal(conteos=tuple(conteos or CONTEOS))),
                     ("prep", crear_preprocesador(**columnas)),
                     ("modelo", estimador)])


class ReglaEstacional(ClassifierMixin, BaseEstimator):
    """Línea base de negocio del PSet 1.

    Alerta si el promedio de las últimas 4 ocurrencias del mismo día de la semana
    (media_mismo_dia_4s) supera el P90 histórico del cantón para ese día.
    Score = m / (m + p90): es > 0.5 exactamente cuando m > p90 (umbral 0.5 = la regla).
    """

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        m = X["media_mismo_dia_4s"].to_numpy(dtype=float)
        p = X["p90_obj"].to_numpy(dtype=float)
        tot = m + p
        s = np.divide(m, tot, out=np.zeros_like(m), where=tot > 0)
        return np.column_stack([1 - s, s])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


def crear_regla_estacional() -> Pipeline:
    return Pipeline([("perfil", PerfilCantonal()), ("modelo", ReglaEstacional())])


# =============================================================================
# State of the Art: modelo fundacional de series de tiempo (zero-shot)
# =============================================================================
NIVELES_CUANTIL = np.round(np.arange(0.1, 1.0, 0.1), 1)


def prob_superar(q: np.ndarray, umbral: np.ndarray) -> np.ndarray:
    """P(volumen en t+3 > umbral) a partir de los 9 cuantiles pronosticados (0.1 .. 0.9).

    Entre cuantiles: interpolación lineal de la función de distribución. Fuera del rango:
    colas exponenciales (cada "paso" entre los dos cuantiles extremos divide la
    probabilidad restante a la mitad), así el score sigue siendo continuo y ordena bien.
    """
    q = np.maximum.accumulate(q, axis=1) + np.arange(q.shape[1]) * 1e-6   # monótono estricto
    u = umbral[:, None]
    k = (q < u).sum(axis=1)                                   # cuántos cuantiles quedan bajo el umbral
    F = np.empty(len(umbral))
    medio = (k > 0) & (k < q.shape[1])
    i = np.where(medio, k, 1)
    q_lo = np.take_along_axis(q, (i - 1)[:, None], 1)[:, 0]
    q_hi = np.take_along_axis(q, i[:, None], 1)[:, 0]
    F[medio] = (NIVELES_CUANTIL[i - 1] + 0.1 * (umbral - q_lo) / (q_hi - q_lo))[medio]
    esc_alta = np.maximum(q[:, -1] - q[:, -2], 1e-3) / np.log(2)
    esc_baja = np.maximum(q[:, 1] - q[:, 0], 1e-3) / np.log(2)
    arriba, abajo = k == q.shape[1], k == 0
    F[arriba] = 1 - 0.1 * np.exp(-(umbral - q[:, -1])[arriba] / esc_alta[arriba])
    F[abajo] = 0.1 * np.exp(-(q[:, 0] - umbral)[abajo] / esc_baja[abajo])
    return np.clip(1 - F, 0, 1)


class ChronosAltaDemanda(ClassifierMixin, BaseEstimator):
    """Chronos-Bolt (Amazon) usado zero-shot como clasificador de alta demanda.

    Para cada fila (cantón, t) toma como contexto la serie diaria del cantón hasta t
    (inclusive, nunca después), pronostica los cuantiles de t+1..t+3 y convierte los
    cuantiles de t+3 en P(volumen > P90 del cantón).
      fit    : no entrena pesos; guarda la serie histórica de las filas de entrenamiento.
      predict: completa la serie con los n_emergencias de las filas a predecir (cada fila
               solo ve fechas <= su t). Huecos (p. ej. los 3 días entre pliegues) -> NaN,
               que Chronos admite como faltantes.
    Los pesos no se guardan con el modelo (se descargan de Hugging Face al predecir).
    """

    def __init__(self, modelo: str = "amazon/chronos-bolt-small", contexto: int = 400,
                 lote: int = 512):
        self.modelo = modelo
        self.contexto = contexto
        self.lote = lote

    def fit(self, X: pd.DataFrame, y=None):
        self.classes_ = np.array([0, 1])
        self.historia_ = (X[["cod_canton", "fecha", "n_emergencias"]]
                          .astype({"cod_canton": str}).drop_duplicates(["cod_canton", "fecha"]))
        return self

    def __getstate__(self):
        estado = self.__dict__.copy()
        estado.pop("_pipe", None)                 # no serializar los pesos del modelo
        return estado

    def _cargar(self):
        if getattr(self, "_pipe", None) is None:
            import torch
            from chronos import BaseChronosPipeline
            torch.set_num_threads(os.cpu_count() or 4)
            self._pipe = BaseChronosPipeline.from_pretrained(
                self.modelo, device_map="cpu", torch_dtype=torch.float32)
        return self._pipe

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        import torch
        pipe = self._cargar()
        nuevas = X[["cod_canton", "fecha", "n_emergencias"]].astype({"cod_canton": str})
        serie = (pd.concat([self.historia_, nuevas]).drop_duplicates(["cod_canton", "fecha"], keep="last"))
        inicio = serie["fecha"].min()
        n_dias = (serie["fecha"].max() - inicio).days + 1
        cantones = {c: i for i, c in enumerate(sorted(serie["cod_canton"].unique()))}
        matriz = np.full((len(cantones), n_dias), np.nan, dtype=np.float32)
        matriz[serie["cod_canton"].map(cantones).to_numpy(),
               (serie["fecha"] - inicio).dt.days.to_numpy()] = serie["n_emergencias"].to_numpy(dtype=np.float32)

        fila_c = X["cod_canton"].astype(str).map(cantones).to_numpy()
        dia = (X["fecha"] - inicio).dt.days.to_numpy()
        umbral = X["p90_obj"].to_numpy(dtype=float)
        out = np.empty(len(X))
        for a in range(0, len(X), self.lote):
            b = min(a + self.lote, len(X))
            ctx = np.full((b - a, self.contexto), np.nan, dtype=np.float32)
            for j in range(a, b):
                fin = dia[j] + 1
                trozo = matriz[fila_c[j], max(0, fin - self.contexto):fin]
                ctx[j - a, self.contexto - len(trozo):] = trozo
            q, _ = pipe.predict_quantiles(torch.from_numpy(ctx), prediction_length=HORIZONTE,
                                          quantile_levels=list(NIVELES_CUANTIL))
            out[a:b] = prob_superar(q[:, HORIZONTE - 1, :].numpy().astype(float), umbral[a:b])
        return np.column_stack([1 - out, out])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


def crear_chronos(**kw) -> Pipeline:
    """PerfilCantonal aporta el umbral P90 (aprendido en entrenamiento); Chronos el pronóstico."""
    return Pipeline([("perfil", PerfilCantonal()), ("modelo", ChronosAltaDemanda(**kw))])


# =============================================================================
# Validación cruzada temporal
# =============================================================================
class SplitTemporal:
    """TimeSeriesSplit sobre FECHAS (no sobre filas).

    Cada fecha tiene las 224 filas de cantones, así que se parte por fecha para que un
    mismo día nunca quede repartido entre entrenamiento y validación. Entre ambos se deja
    un hueco de `gap_dias` = horizonte: el target de la última fila de entrenamiento
    (t+3) cae en el hueco y no en la validación.
    """

    def __init__(self, n_splits: int = 5, gap_dias: int = HORIZONTE):
        self.n_splits = n_splits
        self.gap_dias = gap_dias

    def split(self, X: pd.DataFrame, y=None, groups=None):
        from sklearn.model_selection import TimeSeriesSplit
        fechas = np.sort(X["fecha"].unique())
        tss = TimeSeriesSplit(n_splits=self.n_splits, gap=self.gap_dias)
        f = X["fecha"].to_numpy()
        for i_tr, i_va in tss.split(fechas):
            tr = np.flatnonzero(np.isin(f, fechas[i_tr]))
            va = np.flatnonzero(np.isin(f, fechas[i_va]))
            yield tr, va

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def describir(self, X: pd.DataFrame) -> pd.DataFrame:
        filas = []
        for k, (tr, va) in enumerate(self.split(X), 1):
            filas.append({"pliegue": k,
                          "train_desde": X["fecha"].iloc[tr].min().date(),
                          "train_hasta": X["fecha"].iloc[tr].max().date(),
                          "val_desde": X["fecha"].iloc[va].min().date(),
                          "val_hasta": X["fecha"].iloc[va].max().date(),
                          "filas_train": len(tr), "filas_val": len(va)})
        return pd.DataFrame(filas)


# =============================================================================
# Métricas
# =============================================================================
def pr_auc(y, score) -> float:
    """Métrica técnica (PSet 1): área bajo la curva Precision-Recall (average precision)."""
    return float(average_precision_score(y, score))


def reduccion_fn_vs_regla(y, score, alerta_regla) -> dict:
    """Métrica de negocio (PSet 1): días de alta demanda NO anticipados.

    Se compara con la regla estacional dándole al modelo el MISMO número de alertas que
    emitió la regla (mismo costo de revisión): el modelo alerta en sus k filas de mayor
    score. reduccion_fn = (FN_regla - FN_modelo) / FN_regla. Meta del PSet 1: >= 10%.
    """
    y = np.asarray(y).astype(bool)
    alerta_regla = np.asarray(alerta_regla).astype(bool)
    k = int(alerta_regla.sum())
    orden = np.argsort(-np.asarray(score), kind="stable")
    alerta_modelo = np.zeros_like(y)
    alerta_modelo[orden[:k]] = True
    fn_regla = int((y & ~alerta_regla).sum())
    fn_modelo = int((y & ~alerta_modelo).sum())
    return {"alertas": k,
            "fn_regla": fn_regla,
            "fn_modelo": fn_modelo,
            "reduccion_fn": (fn_regla - fn_modelo) / fn_regla if fn_regla else 0.0,
            "recall_regla": float((y & alerta_regla).sum() / max(y.sum(), 1)),
            "recall_modelo_k": float((y & alerta_modelo).sum() / max(y.sum(), 1))}


@dataclass
class ResultadoCV:
    pliegues: pd.DataFrame
    resumen: dict = field(default_factory=dict)


def evaluar_cv(pipe, train: pd.DataFrame, cv: SplitTemporal | None = None,
               verbose: bool = False, columnas: list | None = None,
               muestra_train: int | None = None) -> ResultadoCV:
    """Validación cruzada temporal sin fuga.

    En cada pliegue: (1) el umbral P90 del target se calcula SOLO con el pliegue de
    entrenamiento; (2) se ajusta un clon del pipeline (PerfilCantonal + preparación +
    modelo) con ese pliegue; (3) se mide en el pliegue de validación.
    `columnas`: entrada del modelo (por defecto ENTRADA). `muestra_train`: si se indica, la
    métrica de entrenamiento se calcula sobre esa cantidad de filas (modelos costosos de
    predecir, como Chronos); la de validación siempre usa el pliegue completo.
    """
    cv = cv or SplitTemporal()
    cols = list(columnas or ENTRADA)
    filas = []
    for k, (i_tr, i_va) in enumerate(cv.split(train), 1):
        tr, va = train.iloc[i_tr], train.iloc[i_va]
        u = umbrales_p90(tr)
        tr, va = etiquetar(tr, u), etiquetar(va, u)
        modelo = clone(pipe)
        t0 = time.perf_counter()
        modelo.fit(tr[cols], tr[TARGET])
        seg_fit = time.perf_counter() - t0
        t0 = time.perf_counter()
        s_va = modelo.predict_proba(va[cols])[:, 1]
        seg_pred = time.perf_counter() - t0
        tr_m = tr.sample(muestra_train, random_state=SEMILLA) if muestra_train else tr
        s_tr = modelo.predict_proba(tr_m[cols])[:, 1]
        regla_va = va["media_mismo_dia_4s"] > va["umbral_p90"]
        neg = reduccion_fn_vs_regla(va[TARGET], s_va, regla_va)
        filas.append({"pliegue": k,
                      "pr_auc_train": pr_auc(tr_m[TARGET], s_tr),
                      "pr_auc_val": pr_auc(va[TARGET], s_va),
                      "roc_auc_val": float(roc_auc_score(va[TARGET], s_va)),
                      "prevalencia_val": float(va[TARGET].mean()),
                      "reduccion_fn_val": neg["reduccion_fn"],
                      "recall_k_val": neg["recall_modelo_k"],
                      "recall_regla_val": neg["recall_regla"],
                      "seg_fit": seg_fit, "seg_pred": seg_pred})
        if verbose:
            print(f"  pliegue {k}: PR-AUC val={filas[-1]['pr_auc_val']:.4f} "
                  f"train={filas[-1]['pr_auc_train']:.4f} ({seg_fit:.1f}s)")
    p = pd.DataFrame(filas)
    resumen = {
        "train_pr_auc": p["pr_auc_train"].mean(),
        "cv_pr_auc_mean": p["pr_auc_val"].mean(),
        "cv_pr_auc_std": p["pr_auc_val"].std(ddof=1),
        "brecha_pr_auc": p["pr_auc_train"].mean() - p["pr_auc_val"].mean(),
        "cv_roc_auc_mean": p["roc_auc_val"].mean(),
        "cv_reduccion_fn_mean": p["reduccion_fn_val"].mean(),
        "cv_reduccion_fn_std": p["reduccion_fn_val"].std(ddof=1),
        "cv_recall_k_mean": p["recall_k_val"].mean(),
        "cv_prevalencia_mean": p["prevalencia_val"].mean(),
        "seg_fit_promedio": p["seg_fit"].mean(),
        "seg_fit_total": p["seg_fit"].sum(),
        "seg_pred_promedio": p["seg_pred"].mean(),
    }
    return ResultadoCV(p, resumen)


# =============================================================================
# MLflow
# =============================================================================
def contexto_git() -> dict:
    def git(*args):
        return subprocess.run(["git", *args], cwd=RAIZ, capture_output=True,
                              text=True).stdout.strip()
    commit = git("rev-parse", "HEAD") or "sin-commit"
    sucio = bool(git("status", "--porcelain", "--", "src", "requirements.txt"))
    return {"git_commit": commit, "git_src_modificado": str(sucio).lower()}


def configurar_mlflow():
    import mlflow
    from src.datos import _cargar_env
    _cargar_env()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(EXPERIMENTO)
    return mlflow


def parametros_datos(meta: dict, train: pd.DataFrame) -> dict:
    return {"datos_tabla_origen": meta["tabla_origen"],
            "datos_fuente": meta["fuente"],
            "datos_fecha_corte_snapshot": meta["fecha_corte_snapshot"],
            "datos_hash": meta["dataset_hash"],
            "datos_filas_train": len(train),
            "datos_train_hasta": str(train["fecha_obj"].max().date()),
            "horizonte_dias": HORIZONTE, "percentil_target": PERCENTIL,
            "cv": "SplitTemporal(n_splits=5, gap_dias=3)"}


def registrar_corrida(nombre: str, pipe, train: pd.DataFrame, meta: dict, *,
                      fase: str, familia: str, params: dict | None = None,
                      resultado: ResultadoCV | None = None, registrar_modelo: bool = True,
                      nested: bool = False, tags_extra: dict | None = None,
                      columnas: list | None = None, muestra_train: int | None = None):
    """Crea una corrida de MLflow con parámetros, datos, métricas, tags y el pipeline.

    Si registrar_modelo=True, reajusta el pipeline con TODO el entrenamiento (umbral P90
    de todo el entrenamiento) y lo guarda con mlflow.sklearn.log_model + firma + ejemplo.
    Devuelve (run_id, pipeline_ajustado).
    """
    import mlflow
    from mlflow.models import infer_signature

    cols = list(columnas or ENTRADA)
    resultado = resultado or evaluar_cv(pipe, train, columnas=cols, muestra_train=muestra_train)
    with mlflow.start_run(run_name=nombre, nested=nested) as run:
        mlflow.set_tags({"fase": fase, "familia": familia, "modelo": nombre,
                         **contexto_git(), **(tags_extra or {})})
        hp = {k: v for k, v in pipe.named_steps["modelo"].get_params().items()
              if not callable(v) and not isinstance(v, BaseEstimator)}
        mlflow.log_params({**{f"hp_{k}": v for k, v in hp.items()},
                           "semilla": SEMILLA, **parametros_datos(meta, train),
                           **(params or {})})
        mlflow.log_metrics({k: float(v) for k, v in resultado.resumen.items()})
        for _, f in resultado.pliegues.iterrows():
            paso = int(f["pliegue"])
            mlflow.log_metric("pliegue_pr_auc_val", f["pr_auc_val"], step=paso)
            mlflow.log_metric("pliegue_pr_auc_train", f["pr_auc_train"], step=paso)
            mlflow.log_metric("pliegue_reduccion_fn_val", f["reduccion_fn_val"], step=paso)

        ajustado = None
        if registrar_modelo:
            datos = etiquetar(train, umbrales_p90(train))
            X, y = datos[cols], datos[TARGET]
            ajustado = clone(pipe)
            t0 = time.perf_counter()
            ajustado.fit(X, y)
            mlflow.log_metric("seg_entrenamiento_final", time.perf_counter() - t0)
            m = datos.sample(muestra_train, random_state=SEMILLA) if muestra_train else datos
            mlflow.log_metric("train_completo_pr_auc",
                              pr_auc(m[TARGET], ajustado.predict_proba(m[cols])[:, 1]))
            ejemplo = X.head(5)
            firma = infer_signature(ejemplo, ajustado.predict_proba(ejemplo))
            mlflow.log_input(mlflow.data.from_pandas(
                datos[list(dict.fromkeys(cols + [TARGET, "fecha"]))], source=meta["tabla_origen"],
                name="obt_train_t3", targets=TARGET), context="training")
            mlflow.sklearn.log_model(ajustado, name="modelo", signature=firma,
                                     input_example=ejemplo,
                                     pyfunc_predict_fn="predict_proba",
                                     code_paths=[str(RAIZ / "src")])
        return run.info.run_id, ajustado
