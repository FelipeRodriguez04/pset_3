"""División temporal, pliegues de CV y target sin fuga.

- Hold-out: últimos `meses` meses calendario según la fecha de corte del snapshot.
- Purga: se quitan de entrenamiento las filas cuyo objetivo (t+h) cae en el hold-out.
- CV: TimeSeriesSplit sobre FECHAS únicas (todos los cantones de un día van juntos)
  con gap = h días, para que el target de train no se solape con validación.
- P90 por (cantón, día de la semana) calculado solo con las fechas de train.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def inicio_holdout(tabla: pd.DataFrame, meses: int = 6) -> pd.Timestamp:
    corte = tabla["fecha"].max()
    return (corte.to_period("M") - (meses - 1)).to_timestamp()


def separar_holdout(tabla: pd.DataFrame, h: int, meses: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (train, test). `h` se recibe para dejar explícito el horizonte de la purga."""
    inicio = inicio_holdout(tabla, meses)
    test = tabla[tabla["fecha"] >= inicio]
    train = tabla[(tabla["fecha"] < inicio) & (tabla["fecha_obj"] < inicio)]
    assert (train["fecha_obj"] - train["fecha"]).dt.days.eq(h).all()
    return train.reset_index(drop=True), test.reset_index(drop=True)


def pliegues_temporales(tabla: pd.DataFrame, h: int, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Índices posicionales (train, validación) de cada pliegue."""
    fechas = np.sort(tabla["fecha"].unique())
    pos_fecha = pd.Series(np.arange(len(fechas)), index=fechas)
    fila_a_fecha = pos_fecha.loc[tabla["fecha"].values].values
    pliegues = []
    for tr_f, va_f in TimeSeriesSplit(n_splits=n_splits, gap=h).split(fechas):
        tr = np.flatnonzero(np.isin(fila_a_fecha, tr_f))
        va = np.flatnonzero(np.isin(fila_a_fecha, va_f))
        pliegues.append((tr, va))
    return pliegues


def tabla_p90(train: pd.DataFrame) -> pd.Series:
    """Cuantil 0.9 de n_emergencias por (cod_canton, dia_semana) con las fechas de train."""
    return (train.groupby(["cod_canton", "dia_semana"])["n_emergencias"]
            .quantile(0.9).rename("p90"))


def p90_objetivo(df: pd.DataFrame, p90: pd.Series) -> np.ndarray:
    """P90 correspondiente al cantón y al día de la semana de t+h (NaN si no visto)."""
    idx = pd.MultiIndex.from_arrays([df["cod_canton"].values, df["dow_obj"].values],
                                    names=["cod_canton", "dia_semana"])
    return p90.reindex(idx).to_numpy(dtype=float)


def etiquetar(df: pd.DataFrame, p90: pd.Series) -> np.ndarray:
    """y = 1 si el volumen en t+h supera el P90 de ese cantón y día de la semana."""
    umbral = p90_objetivo(df, p90)
    return (df["n_obj"].to_numpy(dtype=float) > umbral).astype(int)  # NaN -> False
