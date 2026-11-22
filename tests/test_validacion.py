import numpy as np
import pandas as pd
import pytest

from src.datos import construir_tabla_modelado, obt_sintetica
from src.validacion import etiquetar, pliegues_temporales, separar_holdout, tabla_p90


@pytest.fixture(scope="module")
def tabla3():
    # 500 días desde 2021-07-01 -> termina 2022-11-12 (fecha de corte != 2026-08-31)
    return construir_tabla_modelado(obt_sintetica(n_cantones=4, n_dias=500, seed=3), 3)


def test_holdout_derivado_de_fecha_corte(tabla3):
    train, test = separar_holdout(tabla3, 3, meses=6)
    corte = tabla3.fecha.max()
    inicio = (corte.to_period("M") - 5).to_timestamp()
    assert test.fecha.min() == inicio
    assert test.fecha.max() == corte


def test_purga_target_en_holdout(tabla3):
    train, test = separar_holdout(tabla3, 3, meses=6)
    assert train.fecha_obj.max() < test.fecha.min()
    assert len(train) + len(test) < len(tabla3)  # se purgaron filas


@pytest.mark.parametrize("h", [3, 7])
def test_pliegues_misma_fecha_mismo_lado_y_gap(h):
    tabla = construir_tabla_modelado(obt_sintetica(n_cantones=3, n_dias=400, seed=4), h)
    pl = pliegues_temporales(tabla, h, n_splits=5)
    assert len(pl) == 5
    for tr, va in pl:
        ftr, fva = set(tabla.fecha.iloc[tr]), set(tabla.fecha.iloc[va])
        assert not ftr & fva
        # el target del último día de train (t+h) es anterior al primer día de validación
        assert (min(fva) - max(ftr)).days > h
        # todas las filas de una fecha de validación están en validación
        assert len(va) == tabla.fecha.isin(fva).sum()


def test_p90_solo_train(tabla3):
    train, test = separar_holdout(tabla3, 3)
    p_antes = tabla_p90(train)
    test2 = test.copy()
    test2["n_emergencias"] = 10_000
    p_despues = tabla_p90(train)
    pd.testing.assert_series_equal(p_antes, p_despues)
    assert p_antes.index.names == ["cod_canton", "dia_semana"]


def test_p90_es_cuantil_90_por_canton_y_dia(tabla3):
    p = tabla_p90(tabla3)
    c, d = p.index[0]
    sub = tabla3[(tabla3.cod_canton == c) & (tabla3.dia_semana == d)].n_emergencias
    assert p.loc[(c, d)] == pytest.approx(np.quantile(sub, 0.9))


def test_etiqueta_usa_dow_del_objetivo():
    df = pd.DataFrame({"cod_canton": ["A", "A"], "dow_obj": [1, 2], "n_obj": [5, 5]})
    p = pd.Series([4.0, 6.0], index=pd.MultiIndex.from_tuples([("A", 1), ("A", 2)], names=["cod_canton", "dia_semana"]))
    assert list(etiquetar(df, p)) == [1, 0]


def test_p90_cero_define_target():
    df = pd.DataFrame({"cod_canton": ["Z"] * 3, "dow_obj": [1] * 3, "n_obj": [0, 1, 0]})
    p = pd.Series([0.0], index=pd.MultiIndex.from_tuples([("Z", 1)], names=["cod_canton", "dia_semana"]))
    assert list(etiquetar(df, p)) == [0, 1, 0]


def test_etiqueta_canton_no_visto_es_cero():
    df = pd.DataFrame({"cod_canton": ["NUEVO"], "dow_obj": [1], "n_obj": [99]})
    p = pd.Series([1.0], index=pd.MultiIndex.from_tuples([("A", 1)], names=["cod_canton", "dia_semana"]))
    assert list(etiquetar(df, p)) == [0]


def test_holdout_igual_para_ambos_horizontes_con_corte_a_mitad_de_mes():
    # 432 días desde 2021-07-01 -> corte 2022-09-05: con max(fecha) por horizonte,
    # h=3 (max 09-02) y h=7 (max 08-29) darían inicios distintos.
    obt = obt_sintetica(n_cantones=3, n_dias=432, seed=11)
    assert obt.fecha.max() == pd.Timestamp("2022-09-05")
    inicios = {h: separar_holdout(construir_tabla_modelado(obt, h), h)[1].fecha.min() for h in (3, 7)}
    assert inicios[3] == inicios[7] == pd.Timestamp("2022-04-01")
