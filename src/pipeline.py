"""Preprocesamiento y familias de modelos del PSet 3.

Se usa la misma preparacion en todas las familias. La codificacion ordinal mantiene
el uso de memoria controlado en una laptop y permite entregar una matriz densa a
todos los estimadores, incluido HistGradientBoosting.
"""

from __future__ import annotations

import os

import numpy as np
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


SEED = int(os.getenv("RANDOM_SEED", "42"))
THREADS = int(os.getenv("MODEL_THREADS", "2"))


class SeasonalRuleClassifier(BaseEstimator, ClassifierMixin):
    """Baseline de negocio basado en las ultimas cuatro semanas equivalentes."""

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        if hasattr(X, "loc"):
            raw = np.asarray(X.loc[:, "score_regla_negocio"], dtype=float)
        else:
            raw = np.asarray(X, dtype=float).reshape(-1)
        raw = np.nan_to_num(raw, nan=0.0, posinf=1.5, neginf=0.0)
        probability = np.clip(raw / 1.5, 0.0, 1.0)
        return np.column_stack([1.0 - probability, probability])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 2.0 / 3.0).astype(int)


def construir_preprocesador(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputar", SimpleImputer(strategy="most_frequent")),
            (
                "codificar",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1),
            ),
            ("escalar", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [("numericas", numeric, numeric_columns), ("categoricas", categorical, categorical_columns)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def construir_modelo(name: str):
    models = {
        "dummy": DummyClassifier(strategy="prior", random_state=SEED),
        "logistica": LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=300,
            random_state=SEED,
            solver="lbfgs",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=14,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=THREADS,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=100,
            max_leaf_nodes=31,
            min_samples_leaf=30,
            l2_regularization=1.0,
            class_weight="balanced",
            early_stopping=True,
            random_state=SEED,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(24, 12),
            alpha=0.001,
            learning_rate_init=0.001,
            batch_size=512,
            max_iter=60,
            early_stopping=True,
            n_iter_no_change=7,
            random_state=SEED,
        ),
        "catboost": CatBoostClassifier(
            iterations=120,
            depth=6,
            learning_rate=0.08,
            loss_function="Logloss",
            eval_metric="PRAUC",
            auto_class_weights="Balanced",
            verbose=False,
            random_seed=SEED,
            thread_count=THREADS,
            allow_writing_files=False,
        ),
    }
    if name not in models:
        raise KeyError(f"Modelo desconocido: {name}")
    return models[name]


def construir_pipeline(
    name: str,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> Pipeline:
    return Pipeline(
        [
            ("preparacion", construir_preprocesador(numeric_columns, categorical_columns)),
            ("modelo", construir_modelo(name)),
        ]
    )


FAMILIES = {
    "logistica": "lineal_regularizado",
    "random_forest": "bagging",
    "hist_gradient_boosting": "boosting",
    "mlp": "red_neuronal",
    "catboost": "sota_tabular",
}


SEARCH_SPACES = {
    "logistica": {
        "modelo__C": np.logspace(-3, 2, 100),
        "modelo__class_weight": ["balanced", {0: 1.0, 1: 2.0}, {0: 1.0, 1: 4.0}],
    },
    "random_forest": {
        "modelo__n_estimators": [40, 60, 80, 100, 120],
        "modelo__max_depth": [8, 10, 12, 14, 16, None],
        "modelo__min_samples_leaf": [2, 5, 10, 20, 40],
        "modelo__max_features": ["sqrt", 0.5, 0.8],
        "modelo__class_weight": ["balanced", "balanced_subsample"],
    },
    "hist_gradient_boosting": {
        "modelo__learning_rate": np.logspace(-2, -0.65, 100),
        "modelo__max_iter": [60, 80, 100, 130, 160],
        "modelo__max_leaf_nodes": [15, 23, 31, 47, 63],
        "modelo__min_samples_leaf": [15, 30, 60, 100],
        "modelo__l2_regularization": np.logspace(-3, 1, 50),
        "modelo__class_weight": ["balanced", {0: 1.0, 1: 3.0}, {0: 1.0, 1: 5.0}],
    },
    "mlp": {
        "modelo__hidden_layer_sizes": [(16,), (24, 12), (32, 16), (48, 24)],
        "modelo__alpha": np.logspace(-5, -1, 80),
        "modelo__learning_rate_init": np.logspace(-4, -2, 80),
        "modelo__batch_size": [256, 512, 1024],
    },
    "catboost": {
        "modelo__iterations": [60, 80, 100, 120, 160],
        "modelo__depth": [4, 5, 6, 7, 8],
        "modelo__learning_rate": np.logspace(-2, -0.5, 100),
        "modelo__l2_leaf_reg": np.logspace(-1, 1.3, 80),
        "modelo__random_strength": [0.0, 0.5, 1.0, 2.0],
    },
}
