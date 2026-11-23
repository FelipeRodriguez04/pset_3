"""
Registro de corridas en MLflow: cada corrida guarda parámetros (hiperparámetros y
semilla), datos (tabla, fecha de corte, hash), métricas (train, CV media/desv,
segundos), tags (fase, familia, commit) y el pipeline con firma y ejemplo de entrada.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.base import clone
from sklearn.model_selection import cross_validate

from src.datos import FEATURES, RAIZ, SEMILLA

EXPERIMENTO = "ecu911-demanda-alta"
MODELO_REGISTRO = "ecu911-modelo"
SCORING = "average_precision"            # PR-AUC: métrica técnica del PSet 1


def configurar(uri: str | None = None) -> str:
    """Apunta al servidor de tracking (docker compose up -d mlflow)."""
    from dotenv import load_dotenv

    load_dotenv(RAIZ / ".env")
    uri = uri or os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENTO)
    return uri


def commit_git() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=RAIZ, text=True).strip()
        sucio = subprocess.check_output(["git", "status", "--porcelain", "src"], cwd=RAIZ, text=True).strip()
        return sha + ("-dirty" if sucio else "")
    except Exception:
        return "desconocido"


def _params_modelo(pipe) -> dict:
    est = pipe.named_steps["modelo"] if hasattr(pipe, "named_steps") else pipe
    p = {k: v for k, v in est.get_params(deep=False).items()
         if isinstance(v, (int, float, str, bool, type(None), tuple))}
    return {f"m_{k}": str(v)[:250] for k, v in p.items()}


def evaluar_cv(pipe, X, y, cv) -> dict:
    t0 = time.perf_counter()
    r = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, return_train_score=True, n_jobs=1)
    return {
        "cv_pr_auc_media": float(np.mean(r["test_score"])),
        "cv_pr_auc_desv": float(np.std(r["test_score"])),
        "cv_train_pr_auc_media": float(np.mean(r["train_score"])),
        "cv_segundos": time.perf_counter() - t0,
        "cv_fit_segundos_media": float(np.mean(r["fit_time"])),
        "_folds": [float(s) for s in r["test_score"]],
    }


def registrar_corrida(nombre: str, pipe, X: pd.DataFrame, y: pd.Series, cv, *, fase: str,
                      familia: str, meta_datos, extra_params: dict | None = None,
                      extra_tags: dict | None = None, nested: bool = False,
                      log_modelo: bool = True, cv_resultado: dict | None = None,
                      usar_run_activo: bool = False) -> dict:
    """CV + ajuste final en todo el train + log completo en MLflow. Devuelve métricas.
    Con usar_run_activo=True escribe en la corrida ya abierta (p. ej. la corrida padre del tuning)."""
    from contextlib import nullcontext
    from sklearn.metrics import average_precision_score

    ctx = nullcontext(mlflow.active_run()) if usar_run_activo else mlflow.start_run(run_name=nombre, nested=nested)
    with ctx as run:
        res = cv_resultado or evaluar_cv(pipe, X, y, cv)
        t0 = time.perf_counter()
        final = clone(pipe).fit(X, y)
        seg_train = time.perf_counter() - t0
        score_train = final.predict_proba(X)[:, 1]
        t1 = time.perf_counter()
        final.predict_proba(X.iloc[:10000])
        seg_por_fila = (time.perf_counter() - t1) / min(len(X), 10000)

        metricas = {k: v for k, v in res.items() if not k.startswith("_")}
        metricas.update({
            "train_pr_auc": float(average_precision_score(y, score_train)),
            "brecha_train_cv": float(average_precision_score(y, score_train)) - res["cv_pr_auc_media"],
            "segundos_entrenamiento": seg_train,
            "ms_inferencia_por_1000": seg_por_fila * 1000 * 1000,   # ms para predecir 1000 filas
        })
        for i, s in enumerate(res["_folds"]):
            metricas[f"cv_pr_auc_fold{i}"] = s

        mlflow.log_params({"semilla": SEMILLA, "n_train": len(X), "positivos_train": int(y.sum()),
                           "cv_esquema": type(cv).__name__, "cv_n_splits": cv.get_n_splits(),
                           **meta_datos.como_params(), **_params_modelo(pipe),
                           **(extra_params or {})})
        mlflow.log_metrics(metricas)
        mlflow.set_tags({"fase": fase, "familia": familia, "modelo": nombre,
                         "git_commit": commit_git(), **(extra_tags or {})})
        if log_modelo:
            ejemplo = X[FEATURES].head(5)
            firma = infer_signature(ejemplo, final.predict_proba(ejemplo))
            mlflow.sklearn.log_model(final, artifact_path="modelo", signature=firma,
                                     input_example=ejemplo, pyfunc_predict_fn="predict_proba")
        metricas["run_id"] = run.info.run_id
        metricas["_modelo"] = final
        metricas["_folds"] = res["_folds"]
        return metricas
