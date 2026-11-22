"""Familia "State of the Art": TimesFM 2.5 (200M), modelo fundacional de series de tiempo.

Zero-shot: no se entrena. Para cada cantón y fecha de origen t recibe los últimos
`contexto` días de n_emergencias (hasta t inclusive) y pronostica t+1..t+7.
Del pronóstico se toman t+3 y t+7; el score de alerta se calcula igual que la
regla de negocio: (pronóstico - P90) / (P90 + 1).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

MODELO_ID = "google/timesfm-2.5-200m-pytorch"
_MODELO = None


def contextos_por_origen(obt: pd.DataFrame, fechas_origen, contexto: int = 128):
    """Devuelve (claves[cod_canton, fecha], lista de arrays) con la historia de cada
    cantón que termina en la fecha de origen (incluida). Solo pasado."""
    mat = obt.pivot(index="fecha", columns="cod_canton", values="n_emergencias").sort_index()
    pos = pd.Series(np.arange(len(mat)), index=mat.index)
    valores = mat.to_numpy(dtype=float)
    claves, ctx = [], []
    for f in pd.to_datetime(fechas_origen):
        i = pos[f]
        bloque = valores[max(0, i - contexto + 1): i + 1]
        for j, c in enumerate(mat.columns):
            claves.append((c, f))
            ctx.append(bloque[:, j])
    return pd.DataFrame(claves, columns=["cod_canton", "fecha"]), ctx


def cargar_modelo(contexto: int = 128, lote: int = 256):
    global _MODELO
    if _MODELO is None:
        import timesfm
        import torch

        torch.set_float32_matmul_precision("high")
        m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(MODELO_ID)
        m.compile(timesfm.ForecastConfig(max_context=contexto, max_horizon=8, normalize_inputs=True,
                                         per_core_batch_size=lote, infer_is_positive=True,
                                         fix_quantile_crossing=True))
        _MODELO = m
    return _MODELO


def pronosticar_fundacional(obt: pd.DataFrame, fechas_origen, contexto: int = 128,
                            bloque: int = 8192) -> tuple[pd.DataFrame, dict]:
    """Pronóstico puntual en t+3 y t+7 para cada (cantón, fecha de origen).
    Devuelve (predicciones, info) con el modelo usado y los segundos de inferencia."""
    modelo = cargar_modelo(contexto)
    claves, ctx = contextos_por_origen(obt, fechas_origen, contexto)
    preds = []
    t0 = time.perf_counter()
    for i in range(0, len(ctx), bloque):
        punto, _ = modelo.forecast(horizon=7, inputs=list(ctx[i:i + bloque]))
        preds.append(punto[: len(ctx[i:i + bloque])])
    seg = time.perf_counter() - t0
    p = np.vstack(preds)
    claves["pred_t3"] = p[:, 2]
    claves["pred_t7"] = p[:, 6]
    info = {"modelo_usado": MODELO_ID, "contexto": contexto, "segundos_inferencia": seg,
            "series": len(ctx)}
    return claves, info


def liberar_modelo() -> None:
    """Saca TimesFM de memoria (8 GB de RAM: no debe convivir con el entrenamiento)."""
    global _MODELO
    import gc
    _MODELO = None
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        pass
