"""
Preparación (imputación, codificación, escalado) y métricas del PSet 3.

Todo lo que "aprende" de los datos vive dentro de un Pipeline de scikit-learn:
se ajusta sólo con las filas de entrenamiento de cada pliegue / del train final.
Los notebooks importan desde aquí; no copian código.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.datos import FEATURES_CAT, FEATURES_NUM, SEMILLA

# Costos de negocio (PSet 1 -> PSet 3): un pico no anticipado obliga a coordinar de
# forma reactiva; una falsa alerta cuesta una revisión humana. Se asume 1 FN = 5 FP.
COSTO_FN = 5.0
COSTO_FP = 1.0


# ---------------------------------------------------------------------------
# Preparación
# ---------------------------------------------------------------------------
def preprocesador(escalar: bool = True) -> ColumnTransformer:
    """Numéricas: mediana + (opcional) estandarización. Categóricas: moda + one-hot.
    Los árboles no necesitan escalar, pero se usa la misma preparación para todos
    los modelos (requisito del PSet); escalar no cambia sus particiones."""
    num = [("imputar", SimpleImputer(strategy="median", add_indicator=True))]
    if escalar:
        num.append(("escalar", StandardScaler()))
    cat = [("imputar", SimpleImputer(strategy="most_frequent")),
           ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    return ColumnTransformer(
        [("num", Pipeline(num), FEATURES_NUM), ("cat", Pipeline(cat), FEATURES_CAT)],
        verbose_feature_names_out=False,
    )


def construir(modelo, escalar: bool = True) -> Pipeline:
    return Pipeline([("prep", preprocesador(escalar)), ("modelo", modelo)])


# ---------------------------------------------------------------------------
# Línea base de negocio: regla estacional del PSet 1
# ---------------------------------------------------------------------------
class ReglaEstacional(BaseEstimator, ClassifierMixin):
    """Alerta si la media de las 4 últimas ocurrencias del mismo día de semana
    supera el P90 histórico (ratio_regla > 1). No aprende nada: fit no mira y.
    predict_proba devuelve un puntaje monótono en ratio_regla para poder medir PR-AUC."""

    def fit(self, X, y=None):
        self.classes_ = np.array([0.0, 1.0])
        return self

    def _ratio(self, X):
        r = pd.to_numeric(pd.DataFrame(X)["ratio_regla"], errors="coerce").fillna(0).to_numpy()
        return r

    def predict_proba(self, X):
        r = self._ratio(X)
        p = r / (1.0 + r)                       # ratio 1 -> 0.5
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self._ratio(X) > 1.0).astype(float)


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def pr_auc(y, score) -> float:
    return float(average_precision_score(y, score))


def costo(y, alerta) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, alerta, labels=[0, 1]).ravel()
    c = COSTO_FN * fn + COSTO_FP * fp
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "alertas": int(tp + fp), "recall": tp / max(tp + fn, 1),
            "precision": tp / max(tp + fp, 1),
            "costo_por_1000": 1000 * c / len(y)}


def umbral_costo_minimo(y, score) -> float:
    """Umbral de probabilidad que minimiza COSTO_FN*FN + COSTO_FP*FP."""
    y = np.asarray(y)
    orden = np.argsort(-score)
    s, yy = np.asarray(score)[orden], y[orden]
    tp = np.cumsum(yy)
    fp = np.cumsum(1 - yy)
    fn = yy.sum() - tp
    c = COSTO_FN * fn + COSTO_FP * fp
    i = int(np.argmin(c))
    return float(s[i])


def umbral_presupuesto(score, n_alertas: int) -> float:
    """Umbral que emite exactamente n_alertas (mismas alertas que la regla)."""
    s = np.sort(np.asarray(score))[::-1]
    return float(s[max(n_alertas - 1, 0)])


def metricas_negocio(y, score, umbral, y_regla) -> dict:
    """Métrica de negocio del PSet 1: reducción de picos NO anticipados (FN) frente a
    la regla estacional. Para que sea justa se compara con el MISMO número de alertas
    que emite la regla (si no, alertar todo reduce FN trivialmente). Además se reporta
    el costo con el umbral de costo mínimo."""
    y = np.asarray(y)
    base = costo(y, y_regla)
    u_pres = umbral_presupuesto(score, base["alertas"])
    pres = costo(y, (score >= u_pres).astype(int))
    opt = costo(y, (score >= umbral).astype(int))
    return {
        "fn_regla": base["fn"], "fn_modelo_mismo_presupuesto": pres["fn"],
        "reduccion_fn_pct": 100 * (base["fn"] - pres["fn"]) / max(base["fn"], 1),
        "costo_regla_por_1000": base["costo_por_1000"],
        "costo_modelo_por_1000": opt["costo_por_1000"],
        "recall_umbral": opt["recall"], "precision_umbral": opt["precision"],
        "alertas_umbral": opt["alertas"], "alertas_regla": base["alertas"],
    }


# ---------------------------------------------------------------------------
# Candidatos con hiperparámetros por defecto (fase "candidato")
# ---------------------------------------------------------------------------
def candidatos() -> dict:
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from xgboost import XGBClassifier

    return {
        "logistic_regression": ("lineal", LogisticRegression(max_iter=2000, random_state=SEMILLA)),
        "random_forest": ("bagging", RandomForestClassifier(n_jobs=-1, random_state=SEMILLA)),
        "hist_gradient_boosting": ("boosting", HistGradientBoostingClassifier(random_state=SEMILLA)),
        "lightgbm": ("boosting", LGBMClassifier(random_state=SEMILLA, verbose=-1, n_jobs=-1)),
        "xgboost": ("boosting", XGBClassifier(random_state=SEMILLA, n_jobs=-1, eval_metric="aucpr")),
        "mlp": ("red_neuronal", MLPClassifier(random_state=SEMILLA)),
    }


# ---------------------------------------------------------------------------
# Espacios de búsqueda para el tuning (fase "tuning"). Claves con prefijo modelo__.
# Escalas: tasas de aprendizaje y regularización en escala log (su efecto es
# multiplicativo y el óptimo puede estar en cualquier orden de magnitud); conteos
# (árboles, hojas) en escala lineal o log según su rango; fracciones en uniforme.
# ---------------------------------------------------------------------------
def espacios() -> dict:
    from scipy.stats import loguniform, randint, uniform

    return {
        "lightgbm": {
            "modelo__learning_rate": loguniform(5e-3, 0.3),
            "modelo__n_estimators": randint(100, 800),
            "modelo__num_leaves": randint(8, 128),
            "modelo__min_child_samples": randint(10, 1000),
            "modelo__subsample": uniform(0.5, 0.5), "modelo__subsample_freq": [1],
            "modelo__colsample_bytree": uniform(0.4, 0.6),
            "modelo__reg_lambda": loguniform(1e-3, 10),
            "modelo__class_weight": [None, "balanced"],
        },
        "hist_gradient_boosting": {
            "modelo__learning_rate": loguniform(1e-2, 0.3),
            "modelo__max_iter": randint(100, 600),
            "modelo__max_leaf_nodes": randint(8, 128),
            "modelo__min_samples_leaf": randint(20, 1000),
            "modelo__l2_regularization": loguniform(1e-4, 10),
            "modelo__max_features": uniform(0.4, 0.6),
            "modelo__class_weight": [None, "balanced"],
        },
        "xgboost": {
            "modelo__learning_rate": loguniform(1e-2, 0.3),
            "modelo__n_estimators": randint(100, 800),
            "modelo__max_depth": randint(3, 10),
            "modelo__min_child_weight": loguniform(1, 100),
            "modelo__subsample": uniform(0.5, 0.5),
            "modelo__colsample_bytree": uniform(0.4, 0.6),
            "modelo__reg_lambda": loguniform(1e-3, 10),
            "modelo__scale_pos_weight": [1, 3, 9],
        },
        "mlp": {
            "modelo__hidden_layer_sizes": [(32,), (64,), (128,), (64, 32), (128, 64), (256, 128)],
            "modelo__alpha": loguniform(1e-6, 1e-1),
            "modelo__learning_rate_init": loguniform(1e-4, 1e-2),
            "modelo__batch_size": [256, 512, 1024],
            "modelo__early_stopping": [True],
        },
        "logistic_regression": {
            "modelo__C": loguniform(1e-4, 1e2),
            "modelo__class_weight": [None, "balanced"],
        },
        "random_forest": {
            "modelo__n_estimators": randint(100, 400),
            "modelo__max_depth": [None, 8, 12, 16, 24],
            "modelo__min_samples_leaf": randint(1, 300),
            "modelo__max_features": ["sqrt", 0.3, 0.5],
            "modelo__max_samples": uniform(0.3, 0.5),
        },
    }
