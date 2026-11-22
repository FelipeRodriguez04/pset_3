import numpy as np
import pandas as pd
import pytest

from src.datos import (
    COLUMNAS_OBT,
    FEATURES_CAT,
    FEATURES_NUM,
    construir_tabla_modelado,
    metadatos_dataset,
    obt_sintetica,
)


@pytest.fixture(scope="module")
def obt():
    return obt_sintetica(n_cantones=4, n_dias=200, seed=1)


def test_sintetica_tiene_columnas_obt(obt):
    assert len(COLUMNAS_OBT) == 37
    assert list(obt.columns) == COLUMNAS_OBT


def _valor(obt, canton, fecha):
    fila = obt[(obt.cod_canton == canton) & (obt.fecha == fecha)]
    return float(fila.n_emergencias.iloc[0])


@pytest.mark.parametrize("h,desfases", [(3, [4, 11, 18, 25]), (7, [0, 7, 14, 21])])
def test_prom_4_mismo_dia_sin_futuro(obt, h, desfases):
    tabla = construir_tabla_modelado(obt, h)
    fila = tabla.iloc[len(tabla) // 2]
    esperado = np.mean([_valor(obt, fila.cod_canton, fila.fecha - pd.Timedelta(days=d)) for d in desfases])
    assert fila.prom_4_mismo_dia == pytest.approx(esperado)
    # el día de la semana objetivo es el de t+h
    assert fila.fecha_obj == fila.fecha + pd.Timedelta(days=h)
    assert fila.dow_obj == fila.fecha_obj.dayofweek + 1


@pytest.mark.parametrize("h", [3, 7])
def test_n_obj_es_volumen_en_t_mas_h(obt, h):
    tabla = construir_tabla_modelado(obt, h)
    fila = tabla.iloc[10]
    assert fila.n_obj == _valor(obt, fila.cod_canton, fila.fecha_obj)


@pytest.mark.parametrize("h", [3, 7])
def test_tabla_sin_nulos_en_features_y_target(obt, h):
    tabla = construir_tabla_modelado(obt, h)
    assert len(tabla) > 0
    assert not tabla[FEATURES_NUM + FEATURES_CAT + ["n_obj"]].isna().any().any()


def test_tabla_no_contiene_columnas_de_target_como_feature():
    prohibidas = {"obj_n_emergencias_t3", "obj_n_emergencias_t7", "n_obj"}
    assert not prohibidas & set(FEATURES_NUM + FEATURES_CAT)


def test_metadatos_hash_estable(obt, tmp_path):
    ruta = tmp_path / "obt.parquet"
    obt.to_parquet(ruta, index=False)
    a, b = metadatos_dataset(ruta), metadatos_dataset(ruta)
    assert a["sha256"] == b["sha256"] and len(a["sha256"]) == 64
    assert a["fecha_corte"] == obt.fecha.max().date().isoformat()
    assert a["filas"] == len(obt)
    assert a["tabla_origen"] == "ECU911.OBT.OBT_EMERGENCIAS_CANTON_DIA"


def test_cargar_obt_modo_sintetico(monkeypatch):
    from src.datos import cargar_obt
    monkeypatch.setenv("ECU911_SINTETICO", "1")
    obt, meta = cargar_obt()
    assert list(obt.columns) == COLUMNAS_OBT
    assert meta["tabla_origen"].startswith("SINTETICO")


def test_cargar_obt_sin_snapshot_falla_claro(monkeypatch, tmp_path):
    from src.datos import cargar_obt
    monkeypatch.delenv("ECU911_SINTETICO", raising=False)
    with pytest.raises(FileNotFoundError, match="descargar_snapshot"):
        cargar_obt(tmp_path / "no.parquet")


def test_periodo_anomalo_no_es_feature():
    # la marca de 2024-01 se puso a posteriori en el PSet 2: no existe al momento de predecir
    assert "es_periodo_anomalo" not in FEATURES_NUM + FEATURES_CAT
