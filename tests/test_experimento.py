import os

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from src.datos import COLUMNAS_X, construir_tabla_modelado, obt_sintetica
from src.experimento import (
    configurar_mlflow,
    evaluar_cv,
    evaluar_regla,
    metricas_negocio,
    registrar_corrida,
    score_regla_negocio,
    umbral_negocio,
)
from src.pipeline import MODELOS_DEFECTO, construir_pipeline
from src.validacion import pliegues_temporales, separar_holdout


@pytest.fixture(scope="module")
def tabla():
    t = construir_tabla_modelado(obt_sintetica(n_cantones=5, n_dias=420, seed=7), 3)
    train, _ = separar_holdout(t, 3)
    return train


@pytest.fixture(scope="module")
def pliegues(tabla):
    return pliegues_temporales(tabla, 3, n_splits=5)


def _boosting():
    m, e = MODELOS_DEFECTO["boosting"]()
    return construir_pipeline(m, e)


def test_evaluar_cv_devuelve_5_folds(tabla, pliegues):
    r = evaluar_cv(_boosting, tabla, 3, pliegues)
    assert len(r.pr_auc_folds) == 5
    assert all(0 <= v <= 1 for v in r.pr_auc_folds)
    assert r.segundos > 0
    assert r.brecha == pytest.approx(r.pr_auc_train - r.pr_auc_cv_mean)
    # OOF: una fila por fila de validación de todos los pliegues
    assert len(r.oof) == sum(len(v) for _, v in pliegues)
    assert set(r.oof.columns) >= {"idx", "fold", "y", "score"}


def test_dummy_pr_auc_igual_prevalencia(tabla, pliegues):
    r = evaluar_cv(lambda: construir_pipeline(DummyClassifier(strategy="prior"), False), tabla, 3, pliegues)
    prev = r.oof.groupby("fold").y.mean().to_numpy()
    assert np.allclose(r.pr_auc_folds, prev)


def test_regla_negocio_alerta_si_prom_supera_p90():
    df = pd.DataFrame({"prom_4_mismo_dia": [5.0, 3.0, 0.5], "p90": [4.0, 3.0, 0.0]})
    s = score_regla_negocio(df["prom_4_mismo_dia"].to_numpy(), df["p90"].to_numpy())
    assert list(s > 0) == [True, False, True]


def test_evaluar_regla_sin_fit(tabla, pliegues):
    r = evaluar_regla(score_regla_negocio, tabla, 3, pliegues)
    assert len(r.pr_auc_folds) == 5 and r.segundos == 0


def test_umbral_respeta_precision_min():
    y = np.array([0, 0, 1, 0, 1, 1, 1, 0, 1, 1])
    s = np.linspace(0, 1, 10)
    u = umbral_negocio(y, s, precision_min=0.8)
    m = metricas_negocio(y, s, u)
    assert m["precision"] >= 0.8
    # con el umbral más bajo que aún cumple la precisión, el recall es máximo
    assert m["recall"] == pytest.approx(5 / 6)
    assert m["tp"] + m["fn"] == y.sum()


def test_umbral_sin_solucion_no_alerta():
    y = np.array([0, 1, 0, 1])
    s = np.array([0.9, 0.1, 0.8, 0.2])
    m = metricas_negocio(y, s, umbral_negocio(y, s, precision_min=0.99))
    assert m["tp"] + m["fp"] == 0


def test_registrar_corrida_tags_y_metricas(tabla, pliegues, tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_experiment(experiment_id=mlflow.create_experiment("t", artifact_location=tmp_path.as_uri()))
    r = evaluar_cv(_boosting, tabla, 3, pliegues)
    pipe = _boosting().fit(tabla[COLUMNAS_X], (tabla.n_obj > tabla.n_obj.median()).astype(int))
    meta = {"tabla_origen": "X", "fecha_corte": "2022-01-01", "sha256": "a" * 64, "filas": 1}
    run_id = registrar_corrida("hgb", "candidato", "boosting", 3, {"max_iter": 100}, r, meta,
                               pipeline=pipe, X_ejemplo=tabla[COLUMNAS_X].head(20))
    run = mlflow.get_run(run_id)
    assert run.data.tags["fase"] == "candidato"
    assert run.data.tags["familia"] == "boosting"
    assert run.data.tags["horizonte"] == "t3"
    assert len(run.data.tags["git_commit"]) == 40
    assert run.data.params["seed"] == "42" and run.data.params["dataset_sha256"] == "a" * 64
    for k in ("pr_auc_train", "pr_auc_cv_mean", "pr_auc_cv_std", "brecha", "segundos_entrenamiento"):
        assert k in run.data.metrics
    m = mlflow.sklearn.load_model(f"runs:/{run_id}/modelo")
    assert m.predict_proba(tabla[COLUMNAS_X].head(3)).shape == (3, 2)


def test_configurar_mlflow_error_claro():
    with pytest.raises(RuntimeError, match="MLflow"):
        configurar_mlflow("http://127.0.0.1:9")


def test_experimento_desde_entorno(monkeypatch):
    from src.experimento import nombre_experimento
    monkeypatch.setenv("ECU911_EXPERIMENTO", "dev-x")
    assert nombre_experimento() == "dev-x"
    monkeypatch.delenv("ECU911_EXPERIMENTO")
    assert nombre_experimento() == "ecu911-demanda-alta"


def test_registrar_corrida_marca_arbol_sucio(tabla, pliegues, tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_experiment(experiment_id=mlflow.create_experiment("d", artifact_location=tmp_path.as_uri()))
    meta = {"tabla_origen": "X", "fecha_corte": "2022-01-01", "sha256": "a" * 64, "filas": 1}
    run_id = registrar_corrida("r", "baseline", "regla", 3, {}, None, meta)
    assert mlflow.get_run(run_id).data.tags["git_dirty"] in {"true", "false"}
