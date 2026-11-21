"""
Carga de la OBT del PSet 2 y construcción del target del PSet 1.

Fuente principal: Snowflake, tabla ECU911.OBT.OBT_EMERGENCIAS_CANTON_DIA (la escribe
el job de Spark del PSet 2). Respaldo: el CSV exportado de esa misma tabla (data/).
Ambas fuentes se normalizan al mismo esquema, así que producen el mismo hash.

Uso en los notebooks:
    from src.datos import cargar_obt, agregar_target, dividir_train_test
    obt, meta = cargar_obt()            # meta = fuente, tabla, fecha de corte, hash
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
TABLA_OBT = "OBT.OBT_EMERGENCIAS_CANTON_DIA"
CLAVES = ["cod_canton", "fecha"]

# -----------------------------------------------------------------------------
# Decisiones del experimento (se documentan en el memo y se registran en MLflow)
# -----------------------------------------------------------------------------
HORIZONTE = 3                               # t+3: modelo principal (PSet 1)
PERCENTIL = 0.90                            # y = 1 si el volumen en t+3 supera el P90
FECHA_INICIO_TEST = pd.Timestamp("2025-09-01")  # hold-out: 12 meses de fechas objetivo
PERIODOS_ANOMALOS = ["2024-01"]             # mes con subregistro detectado en el PSet 2

# Esquema de la OBT (37 columnas). Fijar los tipos hace que Snowflake y el CSV
# entreguen exactamente el mismo DataFrame (y por lo tanto el mismo hash).
TEXTO = ["cod_canton", "canton", "cod_provincia", "provincia", "region",
         "periodo", "nombre_dia", "nombre_feriado"]
BOOLEANAS = ["es_zona_no_delimitada", "es_fin_de_semana", "es_feriado",
             "es_periodo_anomalo", "es_feriado_t3", "es_feriado_t7"]
ENTERAS = ["anio", "trimestre", "mes", "dia_mes", "dia_semana", "n_emergencias",
           "n_seguridad_ciudadana", "n_gestion_sanitaria", "n_transito_movilidad",
           "n_servicios_municipales", "n_gestion_siniestros", "n_servicio_militar",
           "n_gestion_riesgos", "n_sin_servicio"]
# Pueden tener nulos (inicio o final de la serie): se guardan como float
CON_NULOS = ["lag_1d", "lag_7d", "lag_14d", "lag_28d", "media_prev_7d", "media_prev_28d",
             "obj_n_emergencias_t3", "obj_n_emergencias_t7"]
COLUMNAS = (["cod_canton", "fecha"] + TEXTO[1:] + BOOLEANAS + ENTERAS + CON_NULOS)


def _cargar_env() -> None:
    """Lee .env de la raíz del proyecto (sin pisar variables ya definidas)."""
    try:
        from dotenv import load_dotenv
        load_dotenv(RAIZ / ".env", override=False)
    except ImportError:
        pass


def _normalizar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=str.lower)
    faltan = set(COLUMNAS) - set(df.columns)
    if faltan:
        raise ValueError(f"La OBT no tiene las columnas esperadas: {sorted(faltan)}")
    df = df[COLUMNAS].copy()
    df["fecha"] = pd.to_datetime(df["fecha"]).astype("datetime64[ns]")
    for c in TEXTO:
        df[c] = df[c].astype("string")
    for c in ["cod_canton", "cod_provincia"]:          # el CSV puede perder ceros iniciales
        ancho = 4 if c == "cod_canton" else 2
        df[c] = df[c].str.zfill(ancho)
    for c in BOOLEANAS:
        df[c] = df[c].map(lambda v: str(v).strip().lower() in ("true", "1")).astype(bool)
    for c in ENTERAS:
        df[c] = pd.to_numeric(df[c]).astype("int64")
    for c in CON_NULOS:
        df[c] = pd.to_numeric(df[c]).astype("float64")
    # Las medias se guardan con 4 decimales en Snowflake; se redondean igual en ambas fuentes
    for c in ["media_prev_7d", "media_prev_28d"]:
        df[c] = df[c].round(4)
    return df.sort_values(CLAVES).reset_index(drop=True)


def _leer_snowflake() -> pd.DataFrame:
    import snowflake.connector

    con = snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        role=os.environ.get("SNOWFLAKE_ROLE"),
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE"),
        database=os.environ.get("SNOWFLAKE_DATABASE", "ECU911"),
        login_timeout=30,
        enable_server_session_keep_alive_auto_detection=False,
    )
    try:
        cur = con.cursor()
        cur.execute(f"SELECT * FROM {TABLA_OBT}")
        return cur.fetch_pandas_all()
    finally:
        con.close()


def hash_dataset(df: pd.DataFrame) -> str:
    """SHA-256 del contenido (independiente del índice). Identifica el snapshot exacto."""
    filas = pd.util.hash_pandas_object(df, index=False).to_numpy()
    return hashlib.sha256(filas.tobytes()).hexdigest()


def cargar_obt(fuente: str = "auto") -> tuple[pd.DataFrame, dict]:
    """Devuelve (obt, metadatos).

    fuente = "snowflake" | "csv" | "auto" (Snowflake y, si falla, el CSV de respaldo).
    Los metadatos son los que se registran en MLflow como datos de cada corrida.
    """
    _cargar_env()
    error = None
    if fuente in ("auto", "snowflake"):
        try:
            df = _leer_snowflake()
            origen = f"snowflake://{os.environ['SNOWFLAKE_DATABASE']}.{TABLA_OBT}"
            usada = "snowflake"
        except Exception as e:                       # sin red, sin credenciales, etc.
            if fuente == "snowflake":
                raise
            error = f"{type(e).__name__}: {e}"
    if fuente == "csv" or error is not None:
        ruta = RAIZ / os.environ.get("OBT_CSV_RESPALDO", "data/obt_emergencias_canton_dia.csv")
        df = pd.read_csv(ruta, dtype={"cod_canton": str, "cod_provincia": str})
        origen = f"csv://{ruta.relative_to(RAIZ).as_posix()}"
        usada = "csv"
        if error:
            print(f"[datos] Snowflake no disponible ({error[:120]}). Se usa el respaldo CSV.")

    df = _normalizar(df)
    meta = {
        "fuente": usada,
        "tabla_origen": origen,
        "fecha_corte_snapshot": df["fecha"].max().date().isoformat(),
        "dataset_hash": hash_dataset(df),
        "n_filas": len(df),
        "n_columnas": df.shape[1],
    }
    print(f"[datos] {usada}: {meta['n_filas']:,} filas x {meta['n_columnas']} columnas | "
          f"corte {meta['fecha_corte_snapshot']} | hash {meta['dataset_hash'][:12]}")
    return df, meta


# =============================================================================
# Dataset de modelado: features derivadas del historial + fechas del objetivo
# =============================================================================
DIAS_HISTORIA_MINIMA = 28   # lag_28d y media_prev_28d necesitan 28 días previos
LAGS_MISMO_DIA = (4, 11, 18, 25)   # mismo día de la semana que t+3, en las 4 semanas previas
LAGS_ANIO_PREVIO = tuple(range(358, 365))   # 7 días centrados en t+3-364 (mismo día, un año antes)


def construir_dataset(obt: pd.DataFrame) -> pd.DataFrame:
    """Agrega a la OBT las columnas que dependen de la fecha objetivo t+3 y del historial.

    Todas las features usan SOLO información disponible al cierre del día t:
      - fecha_obj / dia_semana_obj / mes_obj / es_fin_de_semana_obj: calendario de t+3
        (se conoce de antemano, igual que es_feriado_t3 de la OBT).
      - media_mismo_dia_4s: promedio de los últimos 4 días con el mismo día de la semana
        que t+3 (t-4, t-11, t-18, t-25). Es la estimación de la regla de negocio del PSet 1.
      - media_anio_previo: promedio de los 7 días centrados en t+3-364 (fiestas cantonales y
        estacionalidad anual). NaN durante el primer año del histórico (se imputa en el pipeline).
      - tendencia_7_28: media de los 7 días previos / media de los 28 días previos
        (candidata evaluada en el notebook 02 y descartada: no mejora la CV).
    No se usa obj_n_emergencias_t3 (es el insumo del target) ni obj_*_t7.
    """
    df = obt.sort_values(CLAVES).reset_index(drop=True)

    # El spine de Gold es completo (test fct_spine_completo del PSet 2): 1 fila = 1 día,
    # así que desplazar k filas dentro del cantón = mirar k días atrás.
    dias = df.groupby("cod_canton")["fecha"].diff().dropna()
    if not (dias == pd.Timedelta(days=1)).all():
        raise ValueError("La serie cantón-día tiene huecos: los desplazamientos no serían días.")

    df["fecha_obj"] = df["fecha"] + pd.Timedelta(days=HORIZONTE)
    df["periodo_obj"] = df["fecha_obj"].dt.strftime("%Y-%m")
    df["dia_semana_obj"] = (df["fecha_obj"].dt.dayofweek + 1).astype("int64")  # 1 = lunes ... 7 = domingo
    df["mes_obj"] = df["fecha_obj"].dt.month.astype("int64")
    df["es_fin_de_semana_obj"] = df["dia_semana_obj"] >= 6

    g = df.groupby("cod_canton")["n_emergencias"]
    df["media_mismo_dia_4s"] = sum(g.shift(k) for k in LAGS_MISMO_DIA) / len(LAGS_MISMO_DIA)
    df["media_anio_previo"] = sum(g.shift(k) for k in LAGS_ANIO_PREVIO) / len(LAGS_ANIO_PREVIO)
    df["tendencia_7_28"] = np.where(df["media_prev_28d"] > 0,
                                    df["media_prev_7d"] / df["media_prev_28d"], np.nan)

    # Filas sin historia completa (primeros 28 días del histórico) o sin target observado
    inicio = df["fecha"].min() + pd.Timedelta(days=DIAS_HISTORIA_MINIMA)
    df = df[(df["fecha"] >= inicio) & df["obj_n_emergencias_t3"].notna()]

    # Mes anómalo (subregistro): se excluye del entrenamiento si toca la fila o su objetivo
    df["en_periodo_anomalo"] = (df["periodo"].isin(PERIODOS_ANOMALOS)
                                | df["periodo_obj"].isin(PERIODOS_ANOMALOS))
    return df.reset_index(drop=True)


def dividir_train_test(ds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """División única por fecha OBJETIVO (no aleatoria).

    test : fechas objetivo desde FECHA_INICIO_TEST (2025-09-01 .. 2026-08-31, 12 meses).
    train: fechas objetivo anteriores, sin el mes anómalo. Como se corta por la fecha
           objetivo, ningún target de train usa un día que pertenezca al test.
    """
    es_test = ds["fecha_obj"] >= FECHA_INICIO_TEST
    train = ds[~es_test & ~ds["en_periodo_anomalo"]].reset_index(drop=True)
    test = ds[es_test].reset_index(drop=True)
    return train, test


def umbrales_p90(df: pd.DataFrame, percentil: float = PERCENTIL) -> pd.Series:
    """P90 histórico de n_emergencias por (cantón, día de la semana).

    Se calcula SOLO con las filas que recibe (entrenamiento o el pliegue de entrenamiento
    de la CV): es la definición del target del PSet 1, sin mirar el futuro.
    """
    return (df.groupby(["cod_canton", "dia_semana"])["n_emergencias"]
              .quantile(percentil).rename("umbral_p90"))


def etiquetar(df: pd.DataFrame, umbrales: pd.Series) -> pd.DataFrame:
    """y_t3 = 1 si el volumen real en t+3 SUPERA el P90 del cantón para ese día de la semana."""
    u = umbrales.rename_axis(["cod_canton", "dia_semana_obj"]).reset_index()
    out = df.drop(columns=["umbral_p90"], errors="ignore").merge(
        u, on=["cod_canton", "dia_semana_obj"], how="left", validate="many_to_one")
    if out["umbral_p90"].isna().any():
        raise ValueError("Hay cantón-días sin umbral P90 (cantón ausente del entrenamiento).")
    out["y_t3"] = (out["obj_n_emergencias_t3"] > out["umbral_p90"]).astype(int)
    return out
