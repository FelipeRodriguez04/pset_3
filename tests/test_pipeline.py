import numpy as np
import pandas as pd
import pytest

from src.datos import COLUMNAS_X, construir_tabla_modelado, obt_sintetica
from src.pipeline import MODELOS_DEFECTO, RatioP90, construir_pipeline
from src.validacion import etiquetar, separar_holdout, tabla_p90


@pytest.fixture(scope="module")
def datos():
    tabla = construir_tabla_modelado(obt_sintetica(n_cantones=6, n_dias=450, seed=5), 3)
    train, test = separar_holdout(tabla, 3)
    p90 = tabla_p90(train)
    return train, test, etiquetar(train, p90), etiquetar(test, p90)


def test_ratio_no_inf_con_p90_cero(datos):
    train, *_ = datos
    X = train[COLUMNAS_X].copy()
    X["n_emergencias"] = 0  # todos los P90 = 0
    out = RatioP90().fit(X).transform(X)
    assert np.isfinite(out["ratio_nivel_p90"]).all()
    assert (out["p90_obj"] == 0).all()


def test_ratio_aprende_p90_de_train(datos):
    train, *_ = datos
    r = RatioP90().fit(train[COLUMNAS_X])
    pd.testing.assert_series_equal(r.p90_.rename("p90"), tabla_p90(train), check_names=False)


def test_ratio_canton_no_visto_usa_mediana(datos):
    train, *_ = datos
    r = RatioP90().fit(train[COLUMNAS_X])
    X = train[COLUMNAS_X].head(1).copy()
    X["cod_canton"] = "9999"
    out = r.transform(X)
    assert out["p90_obj"].iloc[0] == pytest.approx(r.p90_.median())


@pytest.mark.parametrize("familia", sorted(MODELOS_DEFECTO))
def test_pipeline_fit_predict_proba(datos, familia):
    train, test, ytr, _ = datos
    modelo, escalar = MODELOS_DEFECTO[familia]()
    pipe = construir_pipeline(modelo, escalar).fit(train[COLUMNAS_X], ytr)
    p = pipe.predict_proba(test[COLUMNAS_X])[:, 1]
    assert p.shape == (len(test),) and np.all((p >= 0) & (p <= 1))


def test_preprocesador_ajusta_solo_train(datos):
    train, test, ytr, _ = datos
    modelo, _ = MODELOS_DEFECTO["lineal"]()
    pipe = construir_pipeline(modelo, escalar=True).fit(train[COLUMNAS_X], ytr)
    escalador = pipe.named_steps["prep"].named_transformers_["num"].named_steps["escalar"]
    assert escalador.mean_[0] == pytest.approx(train["n_emergencias"].mean())
