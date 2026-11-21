import numpy as np
import pandas as pd
import pytest

from src.datos import construir_tabla_modelado, obt_sintetica
from src.fundacional import contextos_por_origen
from src.llm import construir_prompt, parsear_respuesta


@pytest.mark.parametrize("texto", [
    "La probabilidad es alta",           # texto libre
    '{"probabilidad": 1.5}',            # fuera de rango
    '{"probabilidad": -0.1}',
    '{"probabilidad": 0.4',             # JSON roto
    '{"prob": 0.4}',                    # clave equivocada
    '{"probabilidad": "alta"}',
])
def test_parsear_respuesta_invalida(texto):
    assert parsear_respuesta(texto) is None


@pytest.mark.parametrize("texto,esperado", [
    ('{"probabilidad": 0.35}', 0.35),
    ('```json\n{"probabilidad": 0}\n```', 0.0),
    ('Respuesta: {"probabilidad": 1}', 1.0),
])
def test_parsear_respuesta_valida(texto, esperado):
    assert parsear_respuesta(texto) == pytest.approx(esperado)


def test_prompt_no_incluye_target():
    fila = {"canton": "QUITO", "dia_obj": "lunes", "feriado_obj": False, "h": 3, "p90": 812.0,
            "historia": list(range(28)), "n_obj": 123456}
    p = construir_prompt(fila)
    assert "123456" not in p
    assert "812" in p and "lunes" in p and "27" in p


def test_contextos_por_origen_solo_pasado():
    obt = obt_sintetica(n_cantones=2, n_dias=200, seed=9)
    fechas = pd.to_datetime(["2021-10-01", "2021-11-15"])
    claves, ctx = contextos_por_origen(obt, fechas, contexto=64)
    assert len(claves) == len(ctx) == 4
    c, f = claves.iloc[0]["cod_canton"], claves.iloc[0]["fecha"]
    serie = obt[(obt.cod_canton == c) & (obt.fecha <= f)].n_emergencias.to_numpy(float)
    np.testing.assert_array_equal(ctx[0], serie[-64:])  # termina en t (incluido), nada de t+1
