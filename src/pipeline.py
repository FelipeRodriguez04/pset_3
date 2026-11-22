"""Preparación de datos dentro de un Pipeline de scikit-learn.

Todo lo que aprende algo de los datos (P90 por cantón/día, imputación,
codificación, escalado) se ajusta en `fit` con los datos de entrenamiento
del pliegue y solo se aplica en `transform`. Los notebooks importan de aquí.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.datos import FEATURES_CAT, FEATURES_NUM, SEED

CICLICAS = {"dow_obj": 7, "mes_obj": 12}
NUMERICAS = [c for c in FEATURES_NUM if c not in CICLICAS] + ["ratio_nivel_p90", "p90_obj"]


class RatioP90(BaseEstimator, TransformerMixin):
    """Aprende el P90 de n_emergencias por (cod_canton, dia_semana) con las filas de
    entrenamiento y agrega:
      - p90_obj: P90 del cantón para el día de la semana de t+h;
      - ratio_nivel_p90: prom_4_mismo_dia / (p90_obj + 1).
    Cantón/día no visto en train -> mediana global de los P90."""

    def fit(self, X: pd.DataFrame, y=None):
        self.p90_ = X.groupby(["cod_canton", "dia_semana"])["n_emergencias"].quantile(0.9)
        self.mediana_ = float(self.p90_.median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        idx = pd.MultiIndex.from_arrays([X["cod_canton"].values, X["dow_obj"].values])
        p90 = self.p90_.reindex(idx).to_numpy(dtype=float)
        X["p90_obj"] = np.where(np.isnan(p90), self.mediana_, p90)
        X["ratio_nivel_p90"] = X["prom_4_mismo_dia"] / (X["p90_obj"] + 1)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(input_features) + ["p90_obj", "ratio_nivel_p90"], dtype=object)


def _seno_coseno(X: pd.DataFrame) -> np.ndarray:
    cols = []
    for c in X.columns:
        ang = 2 * np.pi * X[c].to_numpy(dtype=float) / CICLICAS[c]
        cols += [np.sin(ang), np.cos(ang)]
    return np.column_stack(cols)


def _nombres_ciclicos(_, input_features):
    return np.asarray([f"{c}_{t}" for c in input_features for t in ("sin", "cos")], dtype=object)


def construir_preprocesador(escalar: bool) -> ColumnTransformer:
    pasos_num = [("imputar", SimpleImputer(strategy="median"))]
    if escalar:
        pasos_num.append(("escalar", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(pasos_num), NUMERICAS),
            ("ciclico", FunctionTransformer(_seno_coseno, feature_names_out=_nombres_ciclicos), list(CICLICAS)),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FEATURES_CAT),
        ],
        remainder="drop",  # cod_canton y dia_semana solo sirven para aprender el P90
        verbose_feature_names_out=False,
    )


def construir_pipeline(modelo, escalar: bool) -> Pipeline:
    return Pipeline([
        ("p90", RatioP90()),
        ("prep", construir_preprocesador(escalar)),
        ("modelo", modelo),
    ])


# Familias candidatas con hiperparámetros por defecto. Únicas excepciones, por los
# 8 GB de RAM: n_jobs y max_samples del bosque, y early_stopping de la red.
MODELOS_DEFECTO = {
    "lineal": lambda: (LogisticRegression(max_iter=2000), True),
    "bagging": lambda: (RandomForestClassifier(max_samples=0.3, n_jobs=2, random_state=SEED), False),
    "boosting": lambda: (HistGradientBoostingClassifier(random_state=SEED), False),
    "red_neuronal": lambda: (MLPClassifier(early_stopping=True, random_state=SEED), True),
}
