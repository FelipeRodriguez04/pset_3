"""EDA, experimentacion, tracking y seleccion para el PSet 3."""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import ParameterSampler

from .datos import DatasetInfo, ModelData, cargar_obt, cortes_temporales, preparar_modelado
from .pipeline import (
    FAMILIES,
    SEARCH_SPACES,
    SeasonalRuleClassifier,
    construir_pipeline,
)


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PLOTS = DOCS / "plots"
CACHE = ROOT / "cache"
MLFLOW_DIR = ROOT / ".mlflow"
SEED = int(os.getenv("RANDOM_SEED", "42"))
CV_SPLITS = int(os.getenv("CV_SPLITS", "4"))
TUNING_CONFIGS = int(os.getenv("TUNING_CONFIGS", "30"))
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "ecu911-alta-demanda")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "ecu911-alta-demanda-modelo")


def preparar_directorios() -> None:
    for path in (DOCS, PLOTS, CACHE, MLFLOW_DIR, MLFLOW_DIR / "artifacts"):
        path.mkdir(parents=True, exist_ok=True)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "sin_commit"


def configurar_mlflow() -> str:
    preparar_directorios()
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        uri = f"sqlite:///{(MLFLOW_DIR / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return uri


def _json_default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def guardar_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _scores(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    raw = np.asarray(model.decision_function(X), dtype=float)
    return 1.0 / (1.0 + np.exp(-raw))


def metricas(y_true: pd.Series | np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "pr_auc": float(average_precision_score(y, scores)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "weighted_error_cost": float(15 * fn + fp),
        "alert_rate": float(predicted.mean()),
    }


def elegir_umbral(
    y_true: np.ndarray,
    scores: np.ndarray,
    baseline_metrics: dict[str, float] | None = None,
) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in np.linspace(0.05, 0.90, 86):
        row = metricas(y_true, scores, float(threshold))
        row["threshold"] = float(threshold)
        rows.append(row)
    table = pd.DataFrame(rows)
    # El objetivo del PSet 1 exige 10% menos picos no anticipados que la regla actual.
    # Si puede lograrse manteniendo una precision razonable, se elige la alternativa
    # con menos falsas alarmas. Si no, se usa el costo FN:FP = 15:1, definido antes
    # de abrir el holdout y suficiente para no favorecer la prediccion siempre negativa.
    table["business_goal_met"] = False
    eligible = pd.DataFrame()
    if baseline_metrics is not None:
        max_fn = np.floor(0.90 * baseline_metrics["false_negatives"])
        min_precision = max(0.10, 0.80 * baseline_metrics["precision"])
        table["business_goal_met"] = (table["false_negatives"] <= max_fn) & (
            table["precision"] >= min_precision
        )
        eligible = table.loc[table["business_goal_met"]]
    if len(eligible):
        best = eligible.sort_values(["false_positives", "false_negatives"], ascending=True).iloc[0]
    else:
        best = table.sort_values(["weighted_error_cost", "false_negatives", "false_positives"]).iloc[0]
    return float(best["threshold"]), table


def _common_tags(phase: str, family: str, horizon: int) -> dict[str, str]:
    return {
        "fase": phase,
        "familia": family,
        "horizonte": f"t+{horizon}",
        "git_commit": git_commit(),
        "validacion": f"backtest_semestral_purgado_{CV_SPLITS}_folds",
        "ponderacion_evaluacion": "ventana_feriado_x2_fin_anio_x2_5",
    }


def _log_data(info: DatasetInfo, data: ModelData) -> None:
    mlflow.log_params(
        {
            "data_source": info.source,
            "snapshot_end": info.snapshot_end,
            "dataset_sha256": info.dataset_hash,
            "dataset_rows": info.rows,
            "dataset_columns": info.columns,
            "holdout_start": data.holdout_start,
            "purge_days": data.purge_days,
            "horizon": data.horizon,
            "seed": SEED,
        }
    )


def _log_model(model, X_example: pd.DataFrame, artifact_path: str = "modelo") -> None:
    example = X_example.head(5).copy()
    predictions = _scores(model, example)
    signature = infer_signature(example, predictions)
    mlflow.sklearn.log_model(
        model,
        artifact_path=artifact_path,
        signature=signature,
        input_example=example,
        pip_requirements=[
            "numpy==1.26.4",
            "pandas==2.2.3",
            "scikit-learn==1.5.2",
            "catboost==1.2.8",
        ],
    )


def ejecutar_eda(df: pd.DataFrame, data: ModelData, info: DatasetInfo) -> dict[str, Any]:
    preparar_directorios()
    target = data.target_column
    target_count = data.target_count_column
    numeric_interest = [
        "n_emergencias",
        target_count,
        "lag_1d",
        "lag_7d",
        "lag_28d",
        "media_prev_7d",
        "media_prev_28d",
        "media_mismo_dia_4",
        "media_mismo_periodo_anio",
        "tendencia_7_28",
        "volatilidad_prev_28d",
        "ratio_estacional_p90",
        "ratio_media_7_p90",
        "ratio_anual_p90",
    ]
    profile = pd.DataFrame(
        {
            "dtype": data.frame.dtypes.astype(str),
            "nulos": data.frame.isna().sum(),
            "pct_nulos": data.frame.isna().mean() * 100,
            "unicos": data.frame.nunique(dropna=True),
        }
    ).reset_index(names="columna")
    profile.to_csv(DOCS / "eda_columnas.csv", index=False)
    data.frame[numeric_interest].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).T.to_csv(
        DOCS / "estadisticas_descriptivas.csv"
    )

    series = data.train[target_count].dropna()
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    z = (series - series.mean()) / series.std(ddof=0)
    spearman = data.train[numeric_interest + [target]].corr(method="spearman")
    spearman.to_csv(DOCS / "correlacion_spearman.csv")

    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sns.countplot(x=target, data=data.train, ax=axes[0], color="#169c9c")
    axes[0].set_title("Balance del target en entrenamiento")
    axes[0].set_xlabel("Alta demanda en t+3")
    axes[0].set_ylabel("Cantón-días")
    clipped = series.clip(upper=series.quantile(0.99))
    sns.histplot(clipped, bins=45, ax=axes[1], color="#194f78")
    axes[1].set_title("Volumen futuro (recortado al P99 para visualizar)")
    axes[1].set_xlabel("Emergencias en la fecha objetivo")
    fig.tight_layout()
    fig.savefig(PLOTS / "01_target_distribucion.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    sns.boxplot(x=series, ax=ax, color="#e4aa33", showfliers=True)
    ax.set_xlim(0, float(series.quantile(0.995)))
    ax.set_title("Boxplot del volumen futuro (eje limitado al P99.5)")
    ax.set_xlabel("Emergencias")
    fig.tight_layout()
    fig.savefig(PLOTS / "02_boxplot_outliers.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    sns.heatmap(spearman, cmap="vlag", center=0, vmin=-1, vmax=1, ax=ax, square=False)
    ax.set_title("Correlación de Spearman")
    fig.tight_layout()
    fig.savefig(PLOTS / "03_correlacion_spearman.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    monthly = df.groupby("periodo", observed=True)["n_emergencias"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(11, 4.0))
    sns.lineplot(data=monthly, x="periodo", y="n_emergencias", marker="o", markersize=3, ax=ax)
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.axvspan("2024-01", "2024-01", color="#cc4c4c", alpha=0.25, label="Periodo anómalo")
    ax.set_title("Volumen mensual del ECU 911")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Emergencias")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOTS / "04_volumen_mensual.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    seasonal_profile = (
        data.train.groupby("temporada_objetivo", observed=True)
        .agg(
            filas=(target, "size"),
            positivos=(target, "sum"),
            prevalencia=(target, "mean"),
            emergencias_futuras_promedio=(target_count, "mean"),
        )
        .reset_index()
        .sort_values("prevalencia", ascending=False)
    )
    seasonal_profile.to_csv(DOCS / "eda_temporadas.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    sns.barplot(data=seasonal_profile, x="temporada_objetivo", y="prevalencia", ax=axes[0], color="#2a9d8f")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set_title("Prevalencia de alta demanda por temporada")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Proporción positiva")
    holiday_profile = (
        data.train.loc[data.train["es_feriado_objetivo"]]
        .groupby("nombre_feriado_objetivo", observed=True)[target]
        .agg(["mean", "size"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    holiday_profile.to_csv(DOCS / "eda_feriados.csv", index=False)
    sns.barplot(data=holiday_profile, y="nombre_feriado_objetivo", x="mean", ax=axes[1], color="#e9c46a")
    axes[1].set_title("Alta demanda por feriado")
    axes[1].set_xlabel("Proporción positiva")
    axes[1].set_ylabel("")
    fig.tight_layout()
    fig.savefig(PLOTS / "08_eda_feriados.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    prevalence_train = float(data.train[target].mean())
    prevalence_test = float(data.test[target].mean())
    strongest = (
        spearman[target]
        .drop([target, target_count, "n_emergencias"], errors="ignore")
        .abs()
        .sort_values(ascending=False)
        .head(3)
    )
    summary = {
        "dataset": asdict(info),
        "train_rows": len(data.train),
        "test_rows": len(data.test),
        "train_date_range": [str(data.train.fecha.min().date()), str(data.train.fecha.max().date())],
        "test_date_range": [str(data.test.fecha.min().date()), str(data.test.fecha.max().date())],
        "target_prevalence_train": prevalence_train,
        "target_prevalence_test": prevalence_test,
        "iqr_upper": float(upper),
        "iqr_outliers": int((series > upper).sum()),
        "iqr_outliers_pct": float((series > upper).mean() * 100),
        "zscore_ge_3": int((z.abs() >= 3).sum()),
        "zscore_ge_3_pct": float((z.abs() >= 3).mean() * 100),
        "strongest_spearman_with_target": {key: float(value) for key, value in strongest.items()},
        "findings": [
            {
                "hallazgo": "El target está desbalanceado",
                "evidencia": f"La clase positiva representa {prevalence_train:.2%} del entrenamiento.",
                "consecuencia": "Usar PR-AUC, ponderación de clases y un umbral basado en costo.",
            },
            {
                "hallazgo": "El volumen futuro tiene cola larga y valores atípicos reales",
                "evidencia": f"IQR detecta {(series > upper).mean():.2%} y Z>=3 detecta {(z.abs() >= 3).mean():.2%}.",
                "consecuencia": "No borrar picos; representan precisamente el evento de negocio.",
            },
            {
                "hallazgo": "El historial reciente y estacional contiene señal",
                "evidencia": "Las mayores asociaciones de Spearman aparecen en lags y medias previas.",
                "consecuencia": "Crear promedio del mismo día de semana, tendencia y volatilidad previa.",
            },
            {
                "hallazgo": "Feriados y temporadas cambian la tasa de alta demanda",
                "evidencia": "La prevalencia se revisó por fin de año, Carnaval, Semana Santa y otros feriados.",
                "consecuencia": "Agregar calendario objetivo, rezagos anuales y backtests estacionales completos.",
            },
        ],
    }
    guardar_json(DOCS / "eda_resumen.json", summary)
    return summary


def _pesos_estacionales(frame: pd.DataFrame) -> np.ndarray:
    """Da más importancia a fechas operativamente críticas sin duplicar filas."""
    weights = np.ones(len(frame), dtype=float)
    if "es_ventana_feriado_objetivo" in frame:
        weights[np.asarray(frame["es_ventana_feriado_objetivo"].fillna(False), dtype=bool)] = 2.0
    if "temporada_objetivo" in frame:
        weights[np.asarray(frame["temporada_objetivo"].eq("fin_anio"), dtype=bool)] = 2.5
    return weights


def _safe_pr_auc(y_true: pd.Series | np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, scores))


def _segment_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    holiday = np.asarray(frame["es_feriado_objetivo"].fillna(False), dtype=bool)
    holiday_window = np.asarray(frame["es_ventana_feriado_objetivo"].fillna(False), dtype=bool)
    end_of_year = np.asarray(frame["temporada_objetivo"].eq("fin_anio"), dtype=bool)
    mobile_holidays = np.asarray(frame["temporada_objetivo"].isin(["carnaval", "semana_santa"]), dtype=bool)
    december = np.asarray(frame["mes_objetivo"].eq(12), dtype=bool)
    return {
        "todos": np.ones(len(frame), dtype=bool),
        "feriado": holiday,
        "ventana_feriado": holiday_window,
        "fin_anio": end_of_year,
        "carnaval_semana_santa": mobile_holidays,
        "diciembre": december,
        "regular": ~holiday_window,
    }


def _metricas_segmentos(
    frame: pd.DataFrame,
    y_true: pd.Series | np.ndarray,
    scores: np.ndarray,
    threshold: float,
    modelo: str,
    alcance: str,
    fold: int | None = None,
) -> list[dict[str, Any]]:
    y = np.asarray(y_true, dtype=int)
    rows: list[dict[str, Any]] = []
    for segmento, mask in _segment_masks(frame).items():
        if mask.sum() == 0 or len(np.unique(y[mask])) < 2:
            continue
        values = metricas(y[mask], scores[mask], threshold=threshold)
        rows.append(
            {
                "modelo": modelo,
                "alcance": alcance,
                "fold": fold,
                "segmento": segmento,
                "filas": int(mask.sum()),
                "positivos": int(y[mask].sum()),
                "prevalencia": float(y[mask].mean()),
                **values,
            }
        )
    return rows


def _evaluar_cv(model, data: ModelData) -> tuple[dict[str, float], Any, list[dict[str, float]]]:
    X = data.train[data.feature_columns]
    y = data.train[data.target_column]
    splits = cortes_temporales(data.train["fecha"], n_splits=CV_SPLITS, horizon=data.horizon)
    fold_rows: list[dict[str, float]] = []
    last_fold_model = None
    start = time.perf_counter()
    for fold, (train_idx, validation_idx) in enumerate(splits, 1):
        fitted = clone(model)
        fold_start = time.perf_counter()
        fitted.fit(X.iloc[train_idx], y.iloc[train_idx])
        scores = _scores(fitted, X.iloc[validation_idx])
        values = metricas(y.iloc[validation_idx], scores)
        validation_frame = data.train.iloc[validation_idx]
        weights = _pesos_estacionales(validation_frame)
        values["pr_auc_ponderado"] = float(
            average_precision_score(y.iloc[validation_idx], scores, sample_weight=weights)
        )
        masks = _segment_masks(validation_frame)
        values["pr_auc_feriados"] = _safe_pr_auc(y.iloc[validation_idx].to_numpy()[masks["feriado"]], scores[masks["feriado"]])
        values["pr_auc_ventana_feriado"] = _safe_pr_auc(y.iloc[validation_idx].to_numpy()[masks["ventana_feriado"]], scores[masks["ventana_feriado"]])
        values["pr_auc_fin_anio"] = _safe_pr_auc(y.iloc[validation_idx].to_numpy()[masks["fin_anio"]], scores[masks["fin_anio"]])
        values["pr_auc_diciembre"] = _safe_pr_auc(y.iloc[validation_idx].to_numpy()[masks["diciembre"]], scores[masks["diciembre"]])
        values["pr_auc_regular"] = _safe_pr_auc(y.iloc[validation_idx].to_numpy()[masks["regular"]], scores[masks["regular"]])
        values["fold"] = fold
        values["train_start"] = str(data.train.iloc[train_idx]["fecha"].min().date())
        values["train_end"] = str(data.train.iloc[train_idx]["fecha"].max().date())
        values["validation_start"] = str(validation_frame["fecha"].min().date())
        values["validation_end"] = str(validation_frame["fecha"].max().date())
        values["train_rows"] = int(len(train_idx))
        values["validation_rows"] = int(len(validation_idx))
        values["fit_seconds"] = time.perf_counter() - fold_start
        fold_rows.append(values)
        last_fold_model = fitted
    elapsed = time.perf_counter() - start
    cv_values = [row["pr_auc"] for row in fold_rows]
    weighted_values = [row["pr_auc_ponderado"] for row in fold_rows]
    summary = {
        "cv_pr_auc_mean": float(np.mean(cv_values)),
        "cv_pr_auc_std": float(np.std(cv_values, ddof=0)),
        "cv_pr_auc_min": float(np.min(cv_values)),
        "cv_pr_auc_ponderado_mean": float(np.mean(weighted_values)),
        "cv_pr_auc_ponderado_std": float(np.std(weighted_values, ddof=0)),
        "cv_pr_auc_feriados_mean": float(np.nanmean([row["pr_auc_feriados"] for row in fold_rows])),
        "cv_pr_auc_ventana_feriado_mean": float(np.nanmean([row["pr_auc_ventana_feriado"] for row in fold_rows])),
        "cv_pr_auc_fin_anio_mean": float(np.nanmean([row["pr_auc_fin_anio"] for row in fold_rows])),
        "cv_pr_auc_diciembre_mean": float(np.nanmean([row["pr_auc_diciembre"] for row in fold_rows])),
        "cv_pr_auc_regular_mean": float(np.nanmean([row["pr_auc_regular"] for row in fold_rows])),
        "cv_fit_seconds": float(elapsed),
    }
    return summary, last_fold_model, fold_rows


def _fit_full(model, data: ModelData) -> tuple[Any, dict[str, float]]:
    X = data.train[data.feature_columns]
    y = data.train[data.target_column]
    start = time.perf_counter()
    fitted = clone(model).fit(X, y)
    fit_seconds = time.perf_counter() - start
    start_pred = time.perf_counter()
    scores = _scores(fitted, X)
    inference_ms_per_1000 = (time.perf_counter() - start_pred) * 1000 / max(len(X), 1) * 1000
    values = metricas(y, scores)
    values = {f"train_{key}": value for key, value in values.items()}
    values["train_fit_seconds"] = fit_seconds
    values["inference_ms_per_1000"] = inference_ms_per_1000
    return fitted, values


def ejecutar_baselines_y_candidatos(data: ModelData, info: DatasetInfo, force: bool = False) -> pd.DataFrame:
    configurar_mlflow()
    output = DOCS / "candidatos.csv"
    if output.exists() and not force:
        return pd.read_csv(output)

    rows: list[dict[str, Any]] = []
    backtest_rows: list[dict[str, Any]] = []
    X = data.train[data.feature_columns]
    y = data.train[data.target_column]

    dummy = construir_pipeline("dummy", data.numeric_columns, data.categorical_columns)
    cv, _, folds = _evaluar_cv(dummy, data)
    backtest_rows.extend({"modelo": "dummy", "familia": "baseline_ingenua", **fold} for fold in folds)
    fitted, train_metrics = _fit_full(dummy, data)
    with mlflow.start_run(run_name="baseline_dummy") as run:
        mlflow.set_tags(_common_tags("baseline", "ingenua", data.horizon))
        _log_data(info, data)
        mlflow.log_param("modelo", "DummyClassifier_prior")
        mlflow.log_metrics({**cv, **train_metrics, "generalization_gap": train_metrics["train_pr_auc"] - cv["cv_pr_auc_mean"]})
        mlflow.log_dict({"folds": folds}, "cv_folds.json")
        _log_model(fitted, X)
        rows.append(
            {
                "modelo": "dummy",
                "familia": "baseline_ingenua",
                "run_id": run.info.run_id,
                **cv,
                **train_metrics,
                "generalization_gap": train_metrics["train_pr_auc"] - cv["cv_pr_auc_mean"],
            }
        )

    seasonal = SeasonalRuleClassifier().fit(data.train[["score_regla_negocio"]], y)
    seasonal_scores_train = seasonal.predict_proba(data.train[["score_regla_negocio"]])[:, 1]
    seasonal_folds: list[dict[str, Any]] = []
    for fold, (train_idx, validation_idx) in enumerate(
        cortes_temporales(data.train["fecha"], n_splits=CV_SPLITS, horizon=data.horizon), 1
    ):
        validation_frame = data.train.iloc[validation_idx]
        validation_scores = seasonal_scores_train[validation_idx]
        validation_y = y.iloc[validation_idx]
        values = metricas(validation_y, validation_scores, threshold=2 / 3)
        values["pr_auc_ponderado"] = float(
            average_precision_score(validation_y, validation_scores, sample_weight=_pesos_estacionales(validation_frame))
        )
        masks = _segment_masks(validation_frame)
        values["pr_auc_feriados"] = _safe_pr_auc(validation_y.to_numpy()[masks["feriado"]], validation_scores[masks["feriado"]])
        values["pr_auc_ventana_feriado"] = _safe_pr_auc(validation_y.to_numpy()[masks["ventana_feriado"]], validation_scores[masks["ventana_feriado"]])
        values["pr_auc_fin_anio"] = _safe_pr_auc(validation_y.to_numpy()[masks["fin_anio"]], validation_scores[masks["fin_anio"]])
        values["pr_auc_diciembre"] = _safe_pr_auc(validation_y.to_numpy()[masks["diciembre"]], validation_scores[masks["diciembre"]])
        values["pr_auc_regular"] = _safe_pr_auc(validation_y.to_numpy()[masks["regular"]], validation_scores[masks["regular"]])
        values.update(
            {
                "fold": fold,
                "train_start": str(data.train.iloc[train_idx]["fecha"].min().date()),
                "train_end": str(data.train.iloc[train_idx]["fecha"].max().date()),
                "validation_start": str(validation_frame["fecha"].min().date()),
                "validation_end": str(validation_frame["fecha"].max().date()),
                "train_rows": int(len(train_idx)),
                "validation_rows": int(len(validation_idx)),
                "fit_seconds": 0.0,
            }
        )
        seasonal_folds.append(values)
        backtest_rows.append({"modelo": "regla_estacional", "familia": "baseline_negocio", **values})
    seasonal_train = metricas(y, seasonal_scores_train, threshold=2 / 3)
    seasonal_cv = {
        "cv_pr_auc_mean": float(np.mean([row["pr_auc"] for row in seasonal_folds])),
        "cv_pr_auc_std": float(np.std([row["pr_auc"] for row in seasonal_folds])),
        "cv_pr_auc_min": float(np.min([row["pr_auc"] for row in seasonal_folds])),
        "cv_pr_auc_ponderado_mean": float(np.mean([row["pr_auc_ponderado"] for row in seasonal_folds])),
        "cv_pr_auc_ponderado_std": float(np.std([row["pr_auc_ponderado"] for row in seasonal_folds])),
        "cv_pr_auc_feriados_mean": float(np.nanmean([row["pr_auc_feriados"] for row in seasonal_folds])),
        "cv_pr_auc_ventana_feriado_mean": float(np.nanmean([row["pr_auc_ventana_feriado"] for row in seasonal_folds])),
        "cv_pr_auc_fin_anio_mean": float(np.nanmean([row["pr_auc_fin_anio"] for row in seasonal_folds])),
        "cv_pr_auc_diciembre_mean": float(np.nanmean([row["pr_auc_diciembre"] for row in seasonal_folds])),
        "cv_pr_auc_regular_mean": float(np.nanmean([row["pr_auc_regular"] for row in seasonal_folds])),
        "cv_fit_seconds": 0.0,
    }
    with mlflow.start_run(run_name="baseline_regla_estacional") as run:
        mlflow.set_tags(_common_tags("baseline", "regla_negocio", data.horizon))
        _log_data(info, data)
        mlflow.log_param("regla", "media(lag_7,lag_14,lag_21,lag_28)>=P90_canton_dia_semana")
        metrics_to_log = {
            **seasonal_cv,
            **{f"train_{key}": value for key, value in seasonal_train.items()},
            "generalization_gap": seasonal_train["pr_auc"] - seasonal_cv["cv_pr_auc_mean"],
        }
        mlflow.log_metrics(metrics_to_log)
        _log_model(seasonal, data.train[["score_regla_negocio"]])
        rows.append(
            {
                "modelo": "regla_estacional",
                "familia": "baseline_negocio",
                "run_id": run.info.run_id,
                **seasonal_cv,
                **{f"train_{key}": value for key, value in seasonal_train.items()},
                "train_fit_seconds": 0.0,
                "inference_ms_per_1000": 0.0,
                "generalization_gap": seasonal_train["pr_auc"] - seasonal_cv["cv_pr_auc_mean"],
            }
        )

    for name, family in FAMILIES.items():
        model = construir_pipeline(name, data.numeric_columns, data.categorical_columns)
        cv, _, folds = _evaluar_cv(model, data)
        backtest_rows.extend({"modelo": name, "familia": family, **fold} for fold in folds)
        fitted, train_metrics = _fit_full(model, data)
        gap = train_metrics["train_pr_auc"] - cv["cv_pr_auc_mean"]
        with mlflow.start_run(run_name=f"candidato_{name}") as run:
            mlflow.set_tags(_common_tags("candidato", family, data.horizon))
            _log_data(info, data)
            mlflow.log_params({"modelo": name, **{key: str(value) for key, value in fitted.named_steps["modelo"].get_params(deep=False).items()}})
            mlflow.log_metrics({**cv, **train_metrics, "generalization_gap": gap})
            mlflow.log_dict({"folds": folds}, "cv_folds.json")
            _log_model(fitted, X)
            rows.append(
                {
                    "modelo": name,
                    "familia": family,
                    "run_id": run.info.run_id,
                    **cv,
                    **train_metrics,
                    "generalization_gap": gap,
                }
            )

    result = pd.DataFrame(rows).sort_values("cv_pr_auc_ponderado_mean", ascending=False)
    result.to_csv(output, index=False)
    pd.DataFrame(backtest_rows).to_csv(DOCS / "backtests_candidatos.csv", index=False)
    return result


def ejecutar_tuning(data: ModelData, info: DatasetInfo, force: bool = False) -> pd.DataFrame:
    configurar_mlflow()
    output = DOCS / "tuning_resumen.csv"
    candidates = ejecutar_baselines_y_candidatos(data, info, force=False)
    top_models = (
        candidates.loc[
            candidates["familia"].isin(FAMILIES.values())
            & candidates["generalization_gap"].le(0.20)
        ]
        .nlargest(3, "cv_pr_auc_ponderado_mean")["modelo"]
        .tolist()
    )
    if output.exists() and not force:
        existing = pd.read_csv(output)
        if (
            set(existing["modelo"]) == set(top_models)
            and "best_cv_pr_auc_ponderado_mean" in existing.columns
        ):
            return existing

    summary_rows: list[dict[str, Any]] = []
    X_example = data.train[data.feature_columns]
    log_models = os.getenv("LOG_TUNING_MODELS", "true").lower() == "true"

    for model_name in top_models:
        detail_path = DOCS / f"tuning_{model_name}.csv"
        base_cv = float(candidates.loc[candidates["modelo"] == model_name, "cv_pr_auc_mean"].iloc[0])
        base_cv_weighted = float(
            candidates.loc[candidates["modelo"] == model_name, "cv_pr_auc_ponderado_mean"].iloc[0]
        )
        sampler = list(ParameterSampler(SEARCH_SPACES[model_name], n_iter=TUNING_CONFIGS, random_state=SEED))
        detail_rows: list[dict[str, Any]] = []

        # Recuperar checkpoints nuevos desde CSV o directamente desde MLflow.
        # Así una pausa no obliga a repetir configuraciones ya terminadas.
        if detail_path.exists():
            checkpoint = pd.read_csv(detail_path)
            if "cv_pr_auc_ponderado_mean" in checkpoint.columns:
                detail_rows = checkpoint.to_dict("records")
        recovered_indices = {int(row["config_index"]) for row in detail_rows}
        tracked = mlflow.search_runs(
            experiment_names=[EXPERIMENT_NAME],
            filter_string=f"tags.fase = 'tuning' and params.modelo = '{model_name}'",
            output_format="pandas",
        )
        if len(tracked):
            metric_names = [
                "cv_pr_auc_mean", "cv_pr_auc_std", "cv_pr_auc_min",
                "cv_pr_auc_ponderado_mean", "cv_pr_auc_ponderado_std",
                "cv_pr_auc_feriados_mean", "cv_pr_auc_ventana_feriado_mean",
                "cv_pr_auc_fin_anio_mean", "cv_pr_auc_diciembre_mean",
                "cv_pr_auc_regular_mean", "cv_fit_seconds",
            ]
            tracked = tracked.sort_values("start_time")
            for _, tracked_row in tracked.iterrows():
                raw_index = tracked_row.get("params.config_index")
                if pd.isna(raw_index):
                    continue
                config_index = int(raw_index)
                if config_index in recovered_indices or not (1 <= config_index <= len(sampler)):
                    continue
                metrics = {
                    name: float(tracked_row[f"metrics.{name}"])
                    for name in metric_names
                    if f"metrics.{name}" in tracked_row and pd.notna(tracked_row[f"metrics.{name}"])
                }
                if "cv_pr_auc_ponderado_mean" not in metrics:
                    continue
                detail_rows.append(
                    {
                        "modelo": model_name,
                        "config_index": config_index,
                        "run_id": tracked_row["run_id"],
                        **metrics,
                        **sampler[config_index - 1],
                    }
                )
                recovered_indices.add(config_index)
        if detail_rows:
            pd.DataFrame(detail_rows).sort_values("config_index").to_csv(detail_path, index=False)

        with mlflow.start_run(run_name=f"tuning_{model_name}") as parent:
            mlflow.set_tags(_common_tags("tuning_parent", FAMILIES[model_name], data.horizon))
            _log_data(info, data)
            mlflow.log_params({"modelo": model_name, "n_configuraciones": len(sampler), "search": "ParameterSampler"})
            for config_index, params in enumerate(sampler, 1):
                if config_index in recovered_indices:
                    continue
                model = construir_pipeline(model_name, data.numeric_columns, data.categorical_columns)
                model.set_params(**params)
                cv, fold_model, folds = _evaluar_cv(model, data)
                with mlflow.start_run(run_name=f"{model_name}_config_{config_index:02d}", nested=True) as child:
                    mlflow.set_tags(_common_tags("tuning", FAMILIES[model_name], data.horizon))
                    mlflow.set_tag("parent_run_id", parent.info.run_id)
                    _log_data(info, data)
                    mlflow.log_params({"modelo": model_name, "config_index": config_index, **{key: str(value) for key, value in params.items()}})
                    mlflow.log_metrics(cv)
                    mlflow.log_dict({"folds": folds}, "cv_folds.json")
                    if log_models and fold_model is not None:
                        _log_model(fold_model, X_example)
                    detail_rows.append(
                        {
                            "modelo": model_name,
                            "config_index": config_index,
                            "run_id": child.info.run_id,
                            **cv,
                            **{key: value for key, value in params.items()},
                        }
                    )
                    recovered_indices.add(config_index)
                    pd.DataFrame(detail_rows).sort_values("config_index").to_csv(detail_path, index=False)

            details = pd.DataFrame(detail_rows).sort_values(
                ["cv_pr_auc_ponderado_mean", "cv_pr_auc_mean"], ascending=False
            )
            details.to_csv(detail_path, index=False)
            best = details.iloc[0]
            # Recuperar el objeto original evita que tuplas o enteros se conviertan a texto/float en CSV.
            best_params = sampler[int(best["config_index"]) - 1].copy()
            best_model = construir_pipeline(model_name, data.numeric_columns, data.categorical_columns)
            best_model.set_params(**best_params)
            fitted, train_metrics = _fit_full(best_model, data)
            improvement = float(best["cv_pr_auc_ponderado_mean"] - base_cv_weighted)
            mlflow.log_metrics(
                {
                    "best_cv_pr_auc_mean": float(best["cv_pr_auc_mean"]),
                    "best_cv_pr_auc_std": float(best["cv_pr_auc_std"]),
                    "best_cv_pr_auc_ponderado_mean": float(best["cv_pr_auc_ponderado_mean"]),
                    "best_cv_pr_auc_ponderado_std": float(best["cv_pr_auc_ponderado_std"]),
                    "improvement_vs_default": improvement,
                    **train_metrics,
                }
            )
            mlflow.log_params({f"best_{key}": str(value) for key, value in best_params.items()})
            mlflow.log_artifact(str(detail_path), artifact_path="resultados")
            _log_model(fitted, X_example)
            joblib.dump(fitted, CACHE / f"tuned_{model_name}.joblib", compress=3)
            guardar_json(CACHE / f"tuned_{model_name}_params.json", best_params)
            summary_rows.append(
                {
                    "modelo": model_name,
                    "familia": FAMILIES[model_name],
                    "parent_run_id": parent.info.run_id,
                    "base_cv_pr_auc": base_cv,
                    "base_cv_pr_auc_ponderado": base_cv_weighted,
                    "best_cv_pr_auc_mean": float(best["cv_pr_auc_mean"]),
                    "best_cv_pr_auc_std": float(best["cv_pr_auc_std"]),
                    "best_cv_pr_auc_ponderado_mean": float(best["cv_pr_auc_ponderado_mean"]),
                    "best_cv_pr_auc_ponderado_std": float(best["cv_pr_auc_ponderado_std"]),
                    "best_cv_pr_auc_feriados_mean": float(best["cv_pr_auc_feriados_mean"]),
                    "best_cv_pr_auc_fin_anio_mean": float(best["cv_pr_auc_fin_anio_mean"]),
                    "improvement_vs_default": improvement,
                    "mejora_supera_desviacion": improvement > float(best["cv_pr_auc_ponderado_std"]),
                    "configuraciones": len(details),
                    "best_params_json": json.dumps(best_params, default=_json_default, ensure_ascii=False),
                    **train_metrics,
                }
            )

    result = pd.DataFrame(summary_rows).sort_values("best_cv_pr_auc_ponderado_mean", ascending=False)
    result.to_csv(output, index=False)
    return result


def _oof_scores(model, data: ModelData) -> tuple[np.ndarray, np.ndarray]:
    X = data.train[data.feature_columns]
    y = data.train[data.target_column]
    all_scores, all_targets = [], []
    for train_idx, validation_idx in cortes_temporales(
        data.train["fecha"], n_splits=CV_SPLITS, horizon=data.horizon
    ):
        fitted = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
        all_scores.append(_scores(fitted, X.iloc[validation_idx]))
        all_targets.append(y.iloc[validation_idx].to_numpy())
    return np.concatenate(all_targets), np.concatenate(all_scores)


def _model_from_selection(model_name: str, params_json: str | None, data: ModelData):
    model = construir_pipeline(model_name, data.numeric_columns, data.categorical_columns)
    if params_json and isinstance(params_json, str) and params_json.strip():
        params = json.loads(params_json)
        class_weight = params.get("modelo__class_weight")
        if isinstance(class_weight, dict):
            params["modelo__class_weight"] = {int(key): value for key, value in class_weight.items()}
        model.set_params(**params)
    return model


def _register_alias(run_id: str, model_name: str, alias: str) -> int:
    result = mlflow.register_model(f"runs:/{run_id}/modelo", model_name)
    client = MlflowClient()
    client.set_registered_model_alias(model_name, alias, result.version)
    return int(result.version)


def ejecutar_seleccion(data: ModelData, info: DatasetInfo, full_df: pd.DataFrame, force: bool = False) -> dict[str, Any]:
    configurar_mlflow()
    output = DOCS / "seleccion_resumen.json"
    if output.exists() and not force:
        return json.loads(output.read_text(encoding="utf-8"))

    candidates = ejecutar_baselines_y_candidatos(data, info, force=False)
    tuning = ejecutar_tuning(data, info, force=False)
    candidate_only = candidates.loc[candidates["modelo"].isin(FAMILIES)].copy()
    leaderboard = candidate_only[
        [
            "modelo",
            "familia",
            "cv_pr_auc_mean",
            "cv_pr_auc_std",
            "cv_pr_auc_min",
            "cv_pr_auc_ponderado_mean",
            "cv_pr_auc_ponderado_std",
            "cv_pr_auc_feriados_mean",
            "cv_pr_auc_fin_anio_mean",
            "cv_pr_auc_diciembre_mean",
            "train_fit_seconds",
            "generalization_gap",
        ]
    ].copy()
    leaderboard["params_json"] = None
    leaderboard["origen"] = "default"
    for _, row in tuning.iterrows():
        idx = leaderboard.index[leaderboard["modelo"] == row["modelo"]][0]
        if row["best_cv_pr_auc_ponderado_mean"] >= leaderboard.loc[idx, "cv_pr_auc_ponderado_mean"]:
            leaderboard.loc[idx, "cv_pr_auc_mean"] = row["best_cv_pr_auc_mean"]
            leaderboard.loc[idx, "cv_pr_auc_std"] = row["best_cv_pr_auc_std"]
            leaderboard.loc[idx, "cv_pr_auc_ponderado_mean"] = row["best_cv_pr_auc_ponderado_mean"]
            leaderboard.loc[idx, "cv_pr_auc_ponderado_std"] = row["best_cv_pr_auc_ponderado_std"]
            leaderboard.loc[idx, "cv_pr_auc_feriados_mean"] = row["best_cv_pr_auc_feriados_mean"]
            leaderboard.loc[idx, "cv_pr_auc_fin_anio_mean"] = row["best_cv_pr_auc_fin_anio_mean"]
            leaderboard.loc[idx, "params_json"] = row["best_params_json"]
            leaderboard.loc[idx, "origen"] = "tuned"
    leaderboard = leaderboard.sort_values(
        ["cv_pr_auc_ponderado_mean", "cv_pr_auc_mean", "cv_pr_auc_std"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    leaderboard.to_csv(DOCS / "leaderboard_final.csv", index=False)
    finalists = leaderboard.head(2)

    seasonal_test_scores = np.clip(data.test["score_regla_negocio"].to_numpy(dtype=float) / 1.5, 0, 1)
    seasonal_test_metrics = metricas(data.test[data.target_column], seasonal_test_scores, threshold=2 / 3)
    final_rows = []
    run_ids: dict[str, str] = {}
    thresholds: dict[str, float] = {}
    fitted_models: dict[str, Any] = {}
    seasonal_rows: list[dict[str, Any]] = []

    seasonal_rows.extend(
        _metricas_segmentos(
            data.test,
            data.test[data.target_column],
            seasonal_test_scores,
            threshold=2 / 3,
            modelo="regla_estacional",
            alcance="holdout_final",
        )
    )

    for position, row in finalists.iterrows():
        name = str(row["modelo"])
        model = _model_from_selection(name, row.get("params_json"), data)
        oof_y, oof_scores = _oof_scores(model, data)
        baseline_oof_scores = []
        for _, validation_idx in cortes_temporales(
            data.train["fecha"], n_splits=CV_SPLITS, horizon=data.horizon
        ):
            baseline_oof_scores.append(
                np.clip(data.train.iloc[validation_idx]["score_regla_negocio"].to_numpy(dtype=float) / 1.5, 0, 1)
            )
        baseline_oof = metricas(oof_y, np.concatenate(baseline_oof_scores), threshold=2 / 3)
        threshold, threshold_table = elegir_umbral(oof_y, oof_scores, baseline_metrics=baseline_oof)
        threshold_table.to_csv(DOCS / f"umbrales_{name}.csv", index=False)
        validation_frames = [
            data.train.iloc[validation_idx]
            for _, validation_idx in cortes_temporales(
                data.train["fecha"], n_splits=CV_SPLITS, horizon=data.horizon
            )
        ]
        oof_frame = pd.concat(validation_frames, axis=0)
        seasonal_rows.extend(
            _metricas_segmentos(
                oof_frame,
                oof_y,
                oof_scores,
                threshold=threshold,
                modelo=name,
                alcance="backtests_oof",
            )
        )
        start = time.perf_counter()
        fitted = clone(model).fit(data.train[data.feature_columns], data.train[data.target_column])
        train_seconds = time.perf_counter() - start
        start = time.perf_counter()
        test_scores = _scores(fitted, data.test[data.feature_columns])
        inference_seconds = time.perf_counter() - start
        test_metrics = metricas(data.test[data.target_column], test_scores, threshold=threshold)
        test_pr_auc_weighted = float(
            average_precision_score(
                data.test[data.target_column],
                test_scores,
                sample_weight=_pesos_estacionales(data.test),
            )
        )
        seasonal_rows.extend(
            _metricas_segmentos(
                data.test,
                data.test[data.target_column],
                test_scores,
                threshold=threshold,
                modelo=name,
                alcance="holdout_final",
            )
        )
        reduction = 1.0 - test_metrics["false_negatives"] / max(seasonal_test_metrics["false_negatives"], 1)
        phase_name = "final_champion" if position == 0 else "final_challenger"
        with mlflow.start_run(run_name=f"{phase_name}_{name}") as run:
            mlflow.set_tags(_common_tags("final", FAMILIES[name], data.horizon))
            mlflow.set_tag("rol", "champion" if position == 0 else "challenger")
            _log_data(info, data)
            mlflow.log_params({"modelo": name, "threshold": threshold, "params_json": row.get("params_json") or "default"})
            mlflow.log_metrics(
                {
                    "cv_pr_auc_mean": float(row["cv_pr_auc_mean"]),
                    "cv_pr_auc_std": float(row["cv_pr_auc_std"]),
                    "cv_pr_auc_ponderado_mean": float(row["cv_pr_auc_ponderado_mean"]),
                    "test_pr_auc": test_metrics["pr_auc"],
                    "test_pr_auc_ponderado": test_pr_auc_weighted,
                    "test_accuracy": test_metrics["accuracy"],
                    "test_precision": test_metrics["precision"],
                    "test_recall": test_metrics["recall"],
                    "test_false_negatives": test_metrics["false_negatives"],
                    "test_false_positives": test_metrics["false_positives"],
                    "business_missed_peak_reduction": reduction,
                    "train_fit_seconds": train_seconds,
                    "test_inference_seconds": inference_seconds,
                }
            )
            _log_model(fitted, data.train[data.feature_columns])
            run_ids[name] = run.info.run_id
        thresholds[name] = threshold
        fitted_models[name] = fitted
        final_rows.append(
            {
                "modelo": name,
                "rol": "champion" if position == 0 else "challenger",
                "run_id": run_ids[name],
                "threshold": threshold,
                "cv_pr_auc_mean": float(row["cv_pr_auc_mean"]),
                "cv_pr_auc_std": float(row["cv_pr_auc_std"]),
                "cv_pr_auc_ponderado_mean": float(row["cv_pr_auc_ponderado_mean"]),
                "cv_pr_auc_feriados_mean": float(row["cv_pr_auc_feriados_mean"]),
                "cv_pr_auc_fin_anio_mean": float(row["cv_pr_auc_fin_anio_mean"]),
                "test_pr_auc": test_metrics["pr_auc"],
                "test_pr_auc_ponderado": test_pr_auc_weighted,
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_false_negatives": test_metrics["false_negatives"],
                "test_false_positives": test_metrics["false_positives"],
                "business_missed_peak_reduction": reduction,
                "train_fit_seconds": train_seconds,
                "test_inference_seconds": inference_seconds,
            }
        )

    final_table = pd.DataFrame(final_rows)
    final_table.to_csv(DOCS / "finalistas_test.csv", index=False)
    pd.DataFrame(seasonal_rows).to_csv(DOCS / "pruebas_estacionales.csv", index=False)
    champion_name = str(finalists.iloc[0]["modelo"])
    challenger_name = str(finalists.iloc[1]["modelo"])
    champion_version = _register_alias(run_ids[champion_name], MODEL_NAME, "champion")
    challenger_version = _register_alias(run_ids[challenger_name], MODEL_NAME, "challenger")

    loaded = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
    reproduced_scores = _scores(loaded, data.test[data.feature_columns])
    reproduced = metricas(data.test[data.target_column], reproduced_scores, thresholds[champion_name])
    expected = float(final_table.loc[final_table["rol"] == "champion", "test_pr_auc"].iloc[0])
    if not np.isclose(reproduced["pr_auc"], expected, rtol=0, atol=1e-12):
        raise AssertionError("La metrica del champion no se reprodujo exactamente desde Model Registry")

    # Sensibilidad del mismo ganador en t+7 para respetar los dos horizontes del PSet 1.
    data_t7 = preparar_modelado(full_df, horizon=7, holdout_start=data.holdout_start)
    champion_params = finalists.iloc[0].get("params_json")
    model_t7 = _model_from_selection(champion_name, champion_params, data_t7)
    fitted_t7 = clone(model_t7).fit(data_t7.train[data_t7.feature_columns], data_t7.train[data_t7.target_column])
    oof_y7, oof_s7 = _oof_scores(model_t7, data_t7)
    baseline_oof_scores_t7 = []
    for _, validation_idx in cortes_temporales(
        data_t7.train["fecha"], n_splits=CV_SPLITS, horizon=data_t7.horizon
    ):
        baseline_oof_scores_t7.append(
            np.clip(data_t7.train.iloc[validation_idx]["score_regla_negocio"].to_numpy(dtype=float) / 1.5, 0, 1)
        )
    baseline_oof_t7 = metricas(oof_y7, np.concatenate(baseline_oof_scores_t7), threshold=2 / 3)
    threshold_t7, _ = elegir_umbral(oof_y7, oof_s7, baseline_metrics=baseline_oof_t7)
    scores_t7 = _scores(fitted_t7, data_t7.test[data_t7.feature_columns])
    metrics_t7 = metricas(data_t7.test[data_t7.target_column], scores_t7, threshold_t7)
    with mlflow.start_run(run_name=f"sensibilidad_t7_{champion_name}") as run_t7:
        mlflow.set_tags(_common_tags("sensibilidad", FAMILIES[champion_name], 7))
        _log_data(info, data_t7)
        mlflow.log_params({"modelo": champion_name, "threshold": threshold_t7})
        mlflow.log_metrics({f"test_{key}": value for key, value in metrics_t7.items()})
        _log_model(fitted_t7, data_t7.train[data_t7.feature_columns])
        t7_run_id = run_t7.info.run_id
    t7_model_name = MODEL_NAME.replace("-modelo", "-t7-modelo")
    t7_version = _register_alias(t7_run_id, t7_model_name, "champion")

    runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME], output_format="pandas")
    runs.to_csv(DOCS / "corridas.csv", index=False)
    summary = {
        "champion": champion_name,
        "challenger": challenger_name,
        "champion_run_id": run_ids[champion_name],
        "challenger_run_id": run_ids[challenger_name],
        "champion_version": champion_version,
        "challenger_version": challenger_version,
        "model_registry_name": MODEL_NAME,
        "champion_threshold": thresholds[champion_name],
        "champion_test_pr_auc": expected,
        "champion_test_accuracy": float(
            final_table.loc[final_table["rol"] == "champion", "test_accuracy"].iloc[0]
        ),
        "champion_cv_pr_auc_ponderado": float(
            final_table.loc[final_table["rol"] == "champion", "cv_pr_auc_ponderado_mean"].iloc[0]
        ),
        "champion_reproduced_pr_auc": reproduced["pr_auc"],
        "business_missed_peak_reduction": float(
            final_table.loc[final_table["rol"] == "champion", "business_missed_peak_reduction"].iloc[0]
        ),
        "seasonal_baseline_test": seasonal_test_metrics,
        "validation_design": {
            "folds": CV_SPLITS,
            "block": "semestres_completos",
            "purge_days": data.purge_days,
            "seasonal_weighting": "x2 ventana feriado; x2.5 fin de año",
            "selection_metric": "cv_pr_auc_ponderado_mean",
        },
        "t7": {
            "model": champion_name,
            "run_id": t7_run_id,
            "registry_name": t7_model_name,
            "version": t7_version,
            "threshold": threshold_t7,
            **metrics_t7,
        },
        "total_mlflow_runs": int(len(runs)),
    }
    guardar_json(output, summary)
    return summary


def cargar_contexto(horizon: int | None = None) -> tuple[pd.DataFrame, ModelData, DatasetInfo]:
    selected_horizon = horizon or int(os.getenv("PRIMARY_HORIZON", "3"))
    holdout_start = os.getenv("HOLDOUT_START", "2026-04-01")
    df, info = cargar_obt()
    data = preparar_modelado(df, horizon=selected_horizon, holdout_start=holdout_start)
    return df, data, info
