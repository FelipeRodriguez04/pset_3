"""
Datos del PSet 3: lectura de la OBT desde Snowflake, construcción del target,
features derivadas y división temporal.

Convención temporal (heredada de build_obt.py del PSet 2):
  - cada fila = información disponible al CIERRE del día t para un cantón;
  - la predicción es para la fecha objetivo d = t + h (h = 3 por defecto);
  - todo lo que se usa como feature debe ser conocido en t.

Target (PSet 1, ajustado en el EDA del PSet 3):
  y_h = 1 si n_emergencias(d) > P90 de las 52 ocurrencias anteriores del mismo
  cantón y el mismo día de la semana de d (rolling, sólo fechas <= d - 7 <= t).
  Un P90 fijo calculado en entrenamiento deja de ser comparable cuando el volumen
  cambia de nivel entre años (ver notebooks/01_eda.ipynb, hallazgo H2).
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
CACHE = RAIZ / "data" / "obt.parquet"
TABLA_OBT = "ECU911.OBT.OBT_EMERGENCIAS_CANTON_DIA"

HORIZONTE = 3              # días entre t (ejecución) y d (fecha objetivo)
SEMANAS_P90 = 52           # ventana del percentil histórico (1 año de ese día de semana)
SEMILLA = 42
INICIO_TEST = "2026-03-01"  # fechas objetivo >= esto son hold-out (6 meses)

SERVICIOS = ["n_seguridad_ciudadana", "n_gestion_sanitaria", "n_transito_movilidad",
             "n_servicios_municipales", "n_gestion_siniestros", "n_servicio_militar",
             "n_gestion_riesgos", "n_sin_servicio"]


# ---------------------------------------------------------------------------
# Snowflake
# ---------------------------------------------------------------------------
def conectar():
    """Conexión con el usuario del pipeline. Credenciales sólo desde .env / entorno."""
    import snowflake.connector
    from dotenv import load_dotenv

    load_dotenv(RAIZ / ".env")
    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        role=os.environ.get("SNOWFLAKE_ROLE", "ECU911_ROLE"),
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "ECU911_WH"),
        database=os.environ.get("SNOWFLAKE_DATABASE", "ECU911"),
    )


def cargar_obt(refrescar: bool = False) -> pd.DataFrame:
    """Lee la OBT completa. La primera vez la guarda en data/obt.parquet (snapshot):
    así todos los notebooks usan exactamente los mismos datos y el hash no cambia."""
    if CACHE.exists() and not refrescar:
        return pd.read_parquet(CACHE)
    with conectar() as con:
        df = con.cursor().execute(f"SELECT * FROM {TABLA_OBT}").fetch_pandas_all()
    df.columns = [c.lower() for c in df.columns]
    df["fecha"] = pd.to_datetime(df["fecha"])
    for c in df.columns:
        if df[c].dtype == "object" and c not in ("cod_canton", "canton", "cod_provincia",
                                                  "provincia", "region", "periodo",
                                                  "nombre_dia", "nombre_feriado"):
            df[c] = df[c].astype(float)
    df = df.sort_values(["cod_canton", "fecha"]).reset_index(drop=True)
    CACHE.parent.mkdir(exist_ok=True)
    df.to_parquet(CACHE, index=False)
    return pd.read_parquet(CACHE)     # mismos tipos que en lecturas posteriores -> mismo hash


def hash_dataset(df: pd.DataFrame) -> str:
    """Hash estable del contenido (independiente del orden de filas)."""
    h = pd.util.hash_pandas_object(df.sort_values(["cod_canton", "fecha"]), index=False)
    return hashlib.sha256(h.values.tobytes()).hexdigest()[:16]


@dataclass
class MetaDatos:
    tabla: str
    fecha_corte: str
    hash: str
    filas: int

    def como_params(self) -> dict:
        return {"datos_tabla": self.tabla, "datos_fecha_corte": self.fecha_corte,
                "datos_hash": self.hash, "datos_filas": self.filas}


def meta(obt: pd.DataFrame) -> MetaDatos:
    return MetaDatos(TABLA_OBT, str(obt["fecha"].max().date()), hash_dataset(obt), len(obt))


# ---------------------------------------------------------------------------
# Target y features
# ---------------------------------------------------------------------------
def p90_mismo_dia(df: pd.DataFrame, adelanto: int, semanas: int = SEMANAS_P90) -> pd.Series:
    """P90 de las `semanas` ocurrencias anteriores del día de semana de la fecha t+adelanto,
    calculado desde la fila t con lags (7k - adelanto), k = 1..semanas.
    Con adelanto <= 7 todos los lags son >= 0: sólo usa días <= t (sin futuro).
    El spine del hecho no tiene huecos, así que 1 fila = 1 día. NaN si falta algún lag."""
    g = df.groupby("cod_canton", sort=False)["n_emergencias"]
    lags = np.column_stack([g.shift(7 * k - adelanto).to_numpy(dtype=float)
                            for k in range(1, semanas + 1)])
    completo = ~np.isnan(lags).any(axis=1)
    out = np.full(len(df), np.nan)
    out[completo] = np.quantile(lags[completo], 0.9, axis=1)   # interpolación lineal
    return pd.Series(out, index=df.index)


def construir_dataset(obt: pd.DataFrame, h: int = HORIZONTE) -> pd.DataFrame:
    """Una fila por (cantón, t) con features conocidas en t y el target de d = t + h."""
    df = obt.sort_values(["cod_canton", "fecha"]).reset_index(drop=True).copy()
    g = df.groupby("cod_canton", sort=False)
    n = df["n_emergencias"]

    # --- target: volumen en d comparado con el P90 de su día de semana ---
    df["p90_dia"] = p90_mismo_dia(df, 0)                     # umbral del propio día t
    df["fecha_objetivo"] = df["fecha"] + pd.Timedelta(days=h)
    df["n_objetivo"] = g["n_emergencias"].shift(-h)          # = obj_n_emergencias_t3
    df["umbral_objetivo"] = p90_mismo_dia(df, h)             # usa fechas <= d-7 <= t
    df["y"] = (df["n_objetivo"] > df["umbral_objetivo"]).astype(float)
    df.loc[df["n_objetivo"].isna() | df["umbral_objetivo"].isna(), "y"] = np.nan

    # --- features derivadas (todas sólo con información hasta t) ---
    # F1 regla estacional como feature: media de las 4 últimas ocurrencias del día
    #    de semana objetivo. Para h=3 son los lags 4, 11, 18, 25 (todos <= t).
    lags_sem = [7 * k - h for k in range(1, 5)]
    df["media_4_mismo_dia"] = sum(g["n_emergencias"].shift(l) for l in lags_sem) / 4
    df["ratio_regla"] = df["media_4_mismo_dia"] / df["umbral_objetivo"].clip(lower=0.5)
    # F2 tendencia de corto plazo vs nivel del mes (hallazgo H3: autocorrelación)
    df["tendencia_7_28"] = df["media_prev_7d"] / df["media_prev_28d"].clip(lower=0.5)
    # F3 pico de hoy relativo al nivel reciente
    df["ratio_hoy_28"] = n / df["media_prev_28d"].clip(lower=0.5)
    # F4 frecuencia reciente de picos (sólo picos cuya fecha ya pasó: d' <= t)
    pico_dia = (n > df["p90_dia"]).astype(float).where(df["p90_dia"].notna())
    df["tasa_picos_28d"] = (pico_dia.groupby(df["cod_canton"]).transform(
        lambda s: s.rolling(28, min_periods=28).mean()))
    # F5 volumen del cantón en escala log (hallazgo H4: cola larga entre cantones)
    df["log_media_28d"] = np.log1p(df["media_prev_28d"])
    # F6 calendario de la fecha objetivo (futuro conocido)
    df["dia_semana_obj"] = df["fecha_objetivo"].dt.dayofweek.astype(str)
    df["mes_obj"] = df["fecha_objetivo"].dt.month
    df["es_feriado_obj"] = df[f"es_feriado_t{h}"].astype(float)
    # F7 mezcla de servicios del día t (proporción de seguridad ciudadana y tránsito)
    tot = n.clip(lower=1)
    df["prop_seguridad"] = df["n_seguridad_ciudadana"] / tot
    df["prop_transito"] = df["n_transito_movilidad"] / tot
    df["region"] = df["region"].astype(str)
    return df


FEATURES_NUM = [
    "n_emergencias", "lag_1d", "lag_7d", "lag_14d", "lag_28d",
    "media_prev_7d", "media_prev_28d",
    "umbral_objetivo", "media_4_mismo_dia", "ratio_regla", "tendencia_7_28",
    "ratio_hoy_28", "tasa_picos_28d", "log_media_28d", "mes_obj", "es_feriado_obj",
    "prop_seguridad", "prop_transito",
]
FEATURES_CAT = ["dia_semana_obj", "region"]
FEATURES = FEATURES_NUM + FEATURES_CAT


# ---------------------------------------------------------------------------
# División temporal
# ---------------------------------------------------------------------------
def modelable(df: pd.DataFrame, excluir_anomalo: bool = True) -> pd.DataFrame:
    """Filas con target definido. Se excluyen fechas objetivo del periodo anómalo
    (2024-01, marcado en dbt): su volumen no es comparable (ver EDA)."""
    m = df["y"].notna()
    if excluir_anomalo:
        per_obj = df["fecha_objetivo"].dt.strftime("%Y-%m")
        m &= per_obj != "2024-01"
        m &= ~df["es_periodo_anomalo"].astype(bool)
    out = df.loc[m].sort_values(["fecha", "cod_canton"]).reset_index(drop=True)
    out["y"] = out["y"].astype(int)
    return out


def dividir(df: pd.DataFrame, inicio_test: str = INICIO_TEST, h: int = HORIZONTE):
    """Train: fecha objetivo < inicio_test. Test: fecha de ejecución >= inicio_test.
    Las h filas intermedias se descartan (su target cae en el test)."""
    ini = pd.Timestamp(inicio_test)
    train = df[df["fecha_objetivo"] < ini].reset_index(drop=True)
    test = df[df["fecha"] >= ini].reset_index(drop=True)
    return train, test


class CVTemporal:
    """TimeSeriesSplit sobre FECHAS (no filas): todos los cantones de un día quedan
    en el mismo pliegue. `gap` = días descartados entre train y validación para
    que ninguna fecha objetivo del train sea posterior al inicio de validación."""

    def __init__(self, fechas, n_splits: int = 5, gap: int = HORIZONTE, dias_val: int = 120):
        self.fechas = pd.to_datetime(pd.Series(fechas)).values
        self.n_splits, self.gap, self.dias_val = n_splits, gap, dias_val

    def split(self, X=None, y=None, groups=None):
        fechas = self.fechas
        unicas = np.unique(fechas)
        fin = len(unicas)
        for k in range(self.n_splits, 0, -1):
            ini_val = fin - k * self.dias_val
            val_d = unicas[ini_val: ini_val + self.dias_val]
            tr_d = unicas[: ini_val - self.gap]
            yield (np.flatnonzero(np.isin(fechas, tr_d)),
                   np.flatnonzero(np.isin(fechas, val_d)))

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
