"""Acceso a la OBT del PSet 2 y construcción de la tabla de modelado.

- `descargar_snapshot` lee la OBT de Snowflake UNA vez y la guarda en parquet.
- Los notebooks trabajan siempre sobre ese snapshot (`cargar_snapshot`).
- `construir_tabla_modelado(obt, h)` arma, para el horizonte h, las features
  y el volumen objetivo `n_obj` (= emergencias en t+h). Todas las features usan
  solo información disponible al cierre del día t.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RUTA_SNAPSHOT = RAIZ / "data" / "obt_snapshot.parquet"
TABLA_ORIGEN = "ECU911.OBT.OBT_EMERGENCIAS_CANTON_DIA"
HORIZONTES = (3, 7)
SEED = 42

# Mismo orden que spark/jobs/build_obt.py
COLUMNAS_OBT = [
    "cod_canton", "fecha",
    "canton", "cod_provincia", "provincia", "region", "es_zona_no_delimitada",
    "anio", "trimestre", "mes", "periodo", "dia_mes", "dia_semana", "nombre_dia",
    "es_fin_de_semana", "es_feriado", "nombre_feriado", "es_periodo_anomalo",
    "n_emergencias", "n_seguridad_ciudadana", "n_gestion_sanitaria", "n_transito_movilidad",
    "n_servicios_municipales", "n_gestion_siniestros", "n_servicio_militar",
    "n_gestion_riesgos", "n_sin_servicio",
    "lag_1d", "lag_7d", "lag_14d", "lag_28d", "media_prev_7d", "media_prev_28d",
    "es_feriado_t3", "es_feriado_t7",
    "obj_n_emergencias_t3", "obj_n_emergencias_t7",
]
SERVICIOS = [
    "n_seguridad_ciudadana", "n_gestion_sanitaria", "n_transito_movilidad",
    "n_servicios_municipales", "n_gestion_siniestros", "n_servicio_militar",
    "n_gestion_riesgos", "n_sin_servicio",
]
BOOLEANAS = ["es_zona_no_delimitada", "es_fin_de_semana", "es_feriado", "es_periodo_anomalo",
             "es_feriado_t3", "es_feriado_t7"]

# Columnas que el Pipeline necesita para aprender el P90 (no son features directas)
COLUMNAS_ID = ["cod_canton", "dia_semana"]
FEATURES_NUM = (
    ["n_emergencias"] + SERVICIOS
    + ["lag_1d", "lag_7d", "lag_14d", "lag_28d", "media_prev_7d", "media_prev_28d",
       "prom_4_mismo_dia", "tendencia_7_28",
       "dow_obj", "mes_obj", "es_feriado_obj", "es_feriado"]
)
FEATURES_CAT = ["region"]
COLUMNAS_X = COLUMNAS_ID + FEATURES_NUM + FEATURES_CAT


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------
def descargar_snapshot(ruta: Path = RUTA_SNAPSHOT) -> Path:
    """Lee la OBT completa de Snowflake y la guarda en `ruta` (parquet).
    Credenciales desde variables de entorno / `.env` (nunca en el código)."""
    import snowflake.connector
    from dotenv import load_dotenv

    load_dotenv(RAIZ / ".env")
    faltan = [v for v in ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD") if not os.getenv(v)]
    if faltan:
        raise RuntimeError(f"Faltan variables en .env: {', '.join(faltan)}")
    con = snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        role=os.getenv("SNOWFLAKE_ROLE"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE", "ECU911"),
    )
    try:
        cur = con.cursor()
        cur.execute(f"SELECT * FROM {TABLA_ORIGEN} ORDER BY COD_CANTON, FECHA")
        df = cur.fetch_pandas_all()
    finally:
        con.close()
    df.columns = [c.lower() for c in df.columns]
    df = _tipar(df[COLUMNAS_OBT])
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ruta, index=False)
    return ruta


def _tipar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["cod_canton"] = df["cod_canton"].astype(str)
    df["cod_provincia"] = df["cod_provincia"].astype(str)
    for c in BOOLEANAS:
        df[c] = df[c].astype("boolean")
    return df.sort_values(["cod_canton", "fecha"]).reset_index(drop=True)


def cargar_snapshot(ruta: Path = RUTA_SNAPSHOT) -> pd.DataFrame:
    if not Path(ruta).exists():
        raise FileNotFoundError(f"No existe {ruta}. Ejecuta primero src.datos.descargar_snapshot().")
    return _tipar(pd.read_parquet(ruta))


def cargar_obt(ruta: Path = RUTA_SNAPSHOT) -> tuple[pd.DataFrame, dict]:
    """OBT + metadatos para MLflow. Con ECU911_SINTETICO=1 usa datos sintéticos
    (solo para desarrollar los notebooks; nunca para la entrega)."""
    if os.getenv("ECU911_SINTETICO") == "1":
        obt = obt_sintetica(n_cantones=12, n_dias=900)
        meta = {"tabla_origen": "SINTETICO (desarrollo)", "fecha_corte": obt.fecha.max().date().isoformat(),
                "sha256": "0" * 64, "filas": len(obt), "ruta": "sintetico"}
        return obt, meta
    return cargar_snapshot(ruta), metadatos_dataset(ruta)


def metadatos_dataset(ruta: Path = RUTA_SNAPSHOT) -> dict:
    """Metadatos que se registran en cada corrida de MLflow."""
    ruta = Path(ruta)
    sha = hashlib.sha256(ruta.read_bytes()).hexdigest()
    fechas = pd.read_parquet(ruta, columns=["fecha"])["fecha"]
    return {
        "tabla_origen": TABLA_ORIGEN,
        "fecha_corte": pd.to_datetime(fechas).max().date().isoformat(),
        "sha256": sha,
        "filas": int(len(fechas)),
        "ruta": str(ruta.name),
    }


# ---------------------------------------------------------------------------
# Tabla de modelado
# ---------------------------------------------------------------------------
def construir_tabla_modelado(obt: pd.DataFrame, h: int) -> pd.DataFrame:
    """Una fila por cantón y día de ejecución t, con features al cierre de t
    y el volumen objetivo `n_obj` en t+h. El P90 y el target binario NO se
    calculan aquí: dependen de las fechas de entrenamiento (ver src.validacion)."""
    if h not in HORIZONTES:
        raise ValueError(f"h debe ser uno de {HORIZONTES}")
    df = obt.sort_values(["cod_canton", "fecha"]).reset_index(drop=True).copy()
    g = df.groupby("cod_canton", sort=False)["n_emergencias"]

    # Promedio de las 4 ocurrencias anteriores del día de la semana de t+h:
    # t+h-7, t+h-14, t+h-21, t+h-28  ->  desfases (7k - h) días, todos >= 0.
    # El spine no tiene huecos, así que 1 fila = 1 día.
    desfases = [7 * k - h for k in (1, 2, 3, 4)]
    df["prom_4_mismo_dia"] = np.mean([g.shift(d) for d in desfases], axis=0)
    df["tendencia_7_28"] = (df["media_prev_7d"] + 1) / (df["media_prev_28d"] + 1)

    df["fecha_obj"] = df["fecha"] + pd.Timedelta(days=h)
    df["dow_obj"] = df["fecha_obj"].dt.isocalendar().day.astype("int64")        # 1 = lunes, como dim_fecha
    df["mes_obj"] = df["fecha_obj"].dt.month
    df["es_feriado_obj"] = df[f"es_feriado_t{h}"]
    df["n_obj"] = df[f"obj_n_emergencias_t{h}"]

    # es_periodo_anomalo NO es feature: la marca de 2024-01 se puso a posteriori (PSet 2)
    for c in ["es_feriado_obj", "es_feriado"]:
        df[c] = df[c].astype("float").astype("int8")

    df = df.dropna(subset=FEATURES_NUM + FEATURES_CAT + ["n_obj"]).reset_index(drop=True)
    cols = ["cod_canton", "canton", "provincia", "fecha", "fecha_obj", "dia_semana", "n_obj"]
    resto = [c for c in COLUMNAS_X if c not in cols]
    return df[cols + resto]


# ---------------------------------------------------------------------------
# Fixture sintético (tests y desarrollo sin Snowflake)
# ---------------------------------------------------------------------------
def obt_sintetica(n_cantones: int = 6, n_dias: int = 400, seed: int = SEED) -> pd.DataFrame:
    """OBT falsa con las 37 columnas y la misma convención temporal que build_obt.py."""
    rng = np.random.default_rng(seed)
    fechas = pd.date_range("2021-07-01", periods=n_dias, freq="D")
    calendario = pd.date_range(fechas[0], periods=n_dias + 7, freq="D")
    feriados = pd.Series(rng.random(len(calendario)) < 0.03, index=calendario)
    regiones = ["Costa", "Sierra", "Amazonía", "Insular"]
    filas = []
    for i in range(n_cantones):
        escala = [0.3, 3, 15, 60, 150, 400][i % 6] * (1 + 0.1 * (i // 6))
        dow = fechas.dayofweek.values
        lam = escala * (1 + 0.25 * (dow >= 4)) * (1 + 0.6 * feriados[fechas].values)
        n = rng.poisson(lam) + rng.poisson(escala * 0.8, n_dias) * (rng.random(n_dias) < 0.03)
        base = pd.DataFrame({"cod_canton": f"{(i % 24) + 1:02d}{i + 1:02d}", "fecha": fechas, "n_emergencias": n})
        partes = rng.multinomial(1, [1 / 8] * 8, size=(n_dias,)).argmax(1)
        for j, s in enumerate(SERVICIOS):
            base[s] = np.where(partes == j, n, 0)
        base["canton"] = f"CANTON {i + 1}"
        base["cod_provincia"] = base["cod_canton"].str[:2]
        base["provincia"] = f"PROVINCIA {base['cod_provincia'].iloc[0]}"
        base["region"] = regiones[i % len(regiones)]
        base["es_zona_no_delimitada"] = False
        filas.append(base)
    df = pd.concat(filas, ignore_index=True)
    f = df["fecha"]
    df["anio"] = f.dt.year
    df["trimestre"] = f.dt.quarter
    df["mes"] = f.dt.month
    df["periodo"] = f.dt.strftime("%Y-%m")
    df["dia_mes"] = f.dt.day
    df["dia_semana"] = f.dt.isocalendar().day.astype("int64")
    df["nombre_dia"] = f.dt.day_name()
    df["es_fin_de_semana"] = df["dia_semana"] >= 6
    df["es_feriado"] = feriados[f].values
    df["nombre_feriado"] = np.where(df["es_feriado"], "Feriado", None)
    df["es_periodo_anomalo"] = df["periodo"] == "2024-01"
    g = df.groupby("cod_canton")["n_emergencias"]
    for k in (1, 7, 14, 28):
        df[f"lag_{k}d"] = g.shift(k)
    for k in (7, 28):
        df[f"media_prev_{k}d"] = g.transform(lambda s, k=k: s.shift(1).rolling(k, min_periods=k).mean())
    for k in (3, 7):
        df[f"es_feriado_t{k}"] = feriados[f + pd.Timedelta(days=k)].values
        df[f"obj_n_emergencias_t{k}"] = g.shift(-k)
    return _tipar(df[COLUMNAS_OBT])
