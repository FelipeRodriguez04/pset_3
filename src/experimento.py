"""Bucle de validación cruzada, métricas, umbral de negocio y registro en MLflow.

Toda corrida (baseline, candidato, tuning, final) pasa por `registrar_corrida`,
así cada una registra exactamente lo mismo.
"""
from __future__ import annotations

import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import mlflow
import numpy as np
import pandas as pd
import requests
from mlflow.models import infer_signature
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.datos import COLUMNAS_X, RAIZ, SEED
from src.validacion import etiquetar, p90_objetivo, tabla_p90

EXPERIMENTO = "ecu911-demanda-alta"
MAX_FILAS_TRAIN_SCORE = 100_000  # PR-AUC de entrenamiento sobre una muestra (velocidad/RAM)


@dataclass
class ResultadoCV:
    pr_auc_folds: list[float]
    pr_auc_train_folds: list[float]
    segundos: float
    oof: pd.DataFrame = field(repr=False)

    @property
    def pr_auc_cv_mean(self) -> float:
        return float(np.mean(self.pr_auc_folds))

    @property
    def pr_auc_cv_std(self) -> float:
        return float(np.std(self.pr_auc_folds))

    @property
    def pr_auc_train(self) -> float:
        return float(np.mean(self.pr_auc_train_folds))

    @property
    def brecha(self) -> float:
        return self.pr_auc_train - self.pr_auc_cv_mean


def _pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(average_precision_score(y, s)) if y.sum() > 0 else float("nan")


def _muestra(n: int, k: int) -> np.ndarray:
    return np.sort(np.random.default_rng(SEED).choice(n, k, replace=False)) if n > k else np.arange(n)


def evaluar_cv(fabrica: Callable, tabla: pd.DataFrame, h: int, pliegues: list,
               max_train_filas: int | None = None) -> ResultadoCV:
    """En cada pliegue: P90 con las fechas de train del pliegue -> etiquetas ->
    fit del pipeline (nuevo) -> score de validación. `max_train_filas` submuestrea
    el entrenamiento (solo para modelos que no caben en memoria; se documenta)."""
    folds, train_folds, oof, segundos = [], [], [], 0.0
    for k, (tr, va) in enumerate(pliegues):
        dtr, dva = tabla.iloc[tr], tabla.iloc[va]
        p90 = tabla_p90(dtr)
        ytr, yva = etiquetar(dtr, p90), etiquetar(dva, p90)
        if max_train_filas:
            sel = _muestra(len(dtr), max_train_filas)
            dfit, yfit = dtr.iloc[sel], ytr[sel]
        else:
            dfit, yfit = dtr, ytr
        modelo = fabrica()
        t0 = time.perf_counter()
        modelo.fit(dfit[COLUMNAS_X], yfit)
        segundos += time.perf_counter() - t0
        s = modelo.predict_proba(dva[COLUMNAS_X])[:, 1]
        folds.append(_pr_auc(yva, s))
        sel = _muestra(len(dtr), MAX_FILAS_TRAIN_SCORE)
        train_folds.append(_pr_auc(ytr[sel], modelo.predict_proba(dtr.iloc[sel][COLUMNAS_X])[:, 1]))
        oof.append(pd.DataFrame({"idx": va, "fold": k, "y": yva, "score": s}))
    return ResultadoCV(folds, train_folds, segundos, pd.concat(oof, ignore_index=True))


def score_regla_negocio(valor: np.ndarray, p90: np.ndarray) -> np.ndarray:
    """> 0 si el valor esperado supera el P90 (la regla alerta). Normalizado por
    (p90 + 1) para que el ranking entre cantones sea comparable (PR-AUC)."""
    valor, p90 = np.asarray(valor, float), np.asarray(p90, float)
    return (valor - p90) / (p90 + 1)


def evaluar_regla(score_fn: Callable, tabla: pd.DataFrame, h: int, pliegues: list,
                  columna: str = "prom_4_mismo_dia") -> ResultadoCV:
    """Para reglas y modelos zero-shot (sin entrenamiento): score = score_fn(columna, P90)."""
    folds, train_folds, oof = [], [], []
    for k, (tr, va) in enumerate(pliegues):
        dtr, dva = tabla.iloc[tr], tabla.iloc[va]
        p90 = tabla_p90(dtr)
        ytr, yva = etiquetar(dtr, p90), etiquetar(dva, p90)
        s = score_fn(dva[columna].to_numpy(), p90_objetivo(dva, p90))
        folds.append(_pr_auc(yva, s))
        train_folds.append(_pr_auc(ytr, score_fn(dtr[columna].to_numpy(), p90_objetivo(dtr, p90))))
        oof.append(pd.DataFrame({"idx": va, "fold": k, "y": yva, "score": s}))
    return ResultadoCV(folds, train_folds, 0.0, pd.concat(oof, ignore_index=True))


def umbral_negocio(y: np.ndarray, score: np.ndarray, precision_min: float) -> float:
    """Umbral más bajo (máximo recall) cuya precisión es >= precision_min.
    Si ninguno la alcanza, devuelve un umbral que no genera alertas."""
    p, r, u = precision_recall_curve(y, score)
    ok = np.flatnonzero(p[:-1] >= precision_min)
    if len(ok) == 0:
        return float(np.nextafter(np.max(score), np.inf))
    return float(u[ok[np.argmax(r[:-1][ok])]])


def metricas_negocio(y: np.ndarray, score: np.ndarray, umbral: float) -> dict:
    y = np.asarray(y).astype(int)
    alerta = np.asarray(score) >= umbral
    tp = int((alerta & (y == 1)).sum())
    fp = int((alerta & (y == 0)).sum())
    fn = int((~alerta & (y == 1)).sum())
    tn = int((~alerta & (y == 0)).sum())
    return {
        "umbral": float(umbral), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "alertas": tp + fp,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }


# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
def configurar_mlflow(uri: str | None = None, experimento: str = EXPERIMENTO) -> str:
    import os
    from dotenv import load_dotenv

    load_dotenv(RAIZ / ".env")
    uri = uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    try:
        ok = requests.get(f"{uri}/health", timeout=5).ok
    except requests.RequestException:
        ok = False
    if not ok:
        raise RuntimeError(f"El servidor de MLflow no responde en {uri}. "
                           "Levántalo con: docker compose --profile mlflow up -d mlflow")
    mlflow.set_tracking_uri(uri)
    return mlflow.set_experiment(experimento).experiment_id


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=RAIZ, capture_output=True,
                          text=True, check=True).stdout.strip()


def registrar_corrida(nombre: str, fase: str, familia: str, h: int, params: dict,
                      resultado: ResultadoCV | None, meta: dict, pipeline=None,
                      X_ejemplo: pd.DataFrame | None = None, nested: bool = False,
                      metricas_extra: dict | None = None, tags_extra: dict | None = None,
                      guardar_oof: bool = False, artefactos: dict | None = None) -> str:
    """Registra una corrida con parámetros, datos, métricas, tags y (si hay) el pipeline."""
    with mlflow.start_run(run_name=nombre, nested=nested) as run:
        mlflow.set_tags({"fase": fase, "familia": familia, "horizonte": f"t{h}",
                         "git_commit": git_commit(), **(tags_extra or {})})
        mlflow.log_params({**{k: str(v)[:500] for k, v in params.items()}, "seed": SEED, "horizonte": h,
                           "tabla_origen": meta["tabla_origen"], "fecha_corte": meta["fecha_corte"],
                           "dataset_sha256": meta["sha256"], "dataset_filas": meta["filas"]})
        if resultado is not None:
            mlflow.log_metrics({
                "pr_auc_train": resultado.pr_auc_train,
                "pr_auc_cv_mean": resultado.pr_auc_cv_mean,
                "pr_auc_cv_std": resultado.pr_auc_cv_std,
                "brecha": resultado.brecha,
                "segundos_entrenamiento": resultado.segundos,
                **{f"pr_auc_fold_{i}": v for i, v in enumerate(resultado.pr_auc_folds)},
            })
        if metricas_extra:
            mlflow.log_metrics({k: float(v) for k, v in metricas_extra.items()})
        with tempfile.TemporaryDirectory() as d:
            if guardar_oof and resultado is not None:
                p = Path(d) / "oof.parquet"
                resultado.oof.to_parquet(p, index=False)
                mlflow.log_artifact(str(p))
            for nombre_art, df in (artefactos or {}).items():
                p = Path(d) / nombre_art
                df.to_csv(p, index=False) if nombre_art.endswith(".csv") else p.write_text(str(df))
                mlflow.log_artifact(str(p))
        if pipeline is not None:
            # enteros como double en la firma: admite nulos al servir y MLflow convierte int -> double
            X_ej = X_ejemplo.astype({c: "float64" for c in X_ejemplo.select_dtypes("integer").columns})
            firma = infer_signature(X_ej, pipeline.predict_proba(X_ej)[:, 1])
            # cloudpickle + code_paths: el pipeline usa clases propias (src.pipeline.RatioP90)
            mlflow.sklearn.log_model(pipeline, name="modelo", signature=firma, input_example=X_ej.head(5),
                                     serialization_format="cloudpickle",
                                     code_paths=[str(RAIZ / "src")])
        return run.info.run_id
