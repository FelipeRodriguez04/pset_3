"""Carga, validacion y construccion de datos para el proyecto ECU 911.

La fuente local y Snowflake entregan el mismo contrato: una fila por canton-dia.
El cambio de fuente se controla con ``DATA_SOURCE`` y no modifica los notebooks.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

BOOLEAN_COLUMNS = [
    "es_zona_no_delimitada",
    "es_fin_de_semana",
    "es_feriado",
    "es_periodo_anomalo",
    "es_feriado_t3",
    "es_feriado_t7",
]


@dataclass(frozen=True)
class DatasetInfo:
    source: str
    snapshot_end: str
    dataset_hash: str
    rows: int
    columns: int


@dataclass
class ModelData:
    frame: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    target_column: str
    target_count_column: str
    horizon: int
    holdout_start: str
    purge_days: int


def _resolve_local_path(value: str | None = None) -> Path:
    configured = value or os.getenv(
        "OBT_LOCAL_PATH",
        "../../../Data_science_end_to_end/laboratorios/semana-09-sesion-01/data/obt_emergencias_canton_dia.csv",
    )
    path = Path(configured)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"No se encontro la OBT local: {path}")
    return path


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_local(path: Path) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(
        path,
        dtype={"cod_canton": "string", "cod_provincia": "string"},
        parse_dates=["fecha"],
        low_memory=False,
    )
    return df, sha256_file(path)


def _load_snowflake() -> tuple[pd.DataFrame, str]:
    try:
        import snowflake.connector
    except ImportError as exc:  # pragma: no cover - solo se activa con Snowflake
        raise RuntimeError("Instala requirements.txt para usar Snowflake") from exc

    required = ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise ValueError(f"Faltan variables de Snowflake: {', '.join(missing)}")

    database = os.getenv("SNOWFLAKE_DATABASE", "ECU911")
    schema = os.getenv("SNOWFLAKE_SCHEMA", "OBT")
    table = os.getenv("SNOWFLAKE_TABLE", "OBT_EMERGENCIAS_CANTON_DIA")
    qualified = f'"{database}"."{schema}"."{table}"'
    connection = snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        role=os.getenv("SNOWFLAKE_ROLE", "ECU911_ROLE"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "ECU911_WH"),
        database=database,
        schema=schema,
    )
    try:
        cursor = connection.cursor()
        try:
            cursor.execute(f"SELECT * FROM {qualified} ORDER BY FECHA, COD_CANTON")
            df = cursor.fetch_pandas_all()
            df.columns = [column.lower() for column in df.columns]
            df["fecha"] = pd.to_datetime(df["fecha"])
            snapshot = str(df["fecha"].max().date())
            fingerprint = hashlib.sha256(
                f"snowflake:{qualified}:{len(df)}:{snapshot}".encode("utf-8")
            ).hexdigest()
            return df, fingerprint
        finally:
            cursor.close()
    finally:
        connection.close()


def cargar_obt(source: str | None = None) -> tuple[pd.DataFrame, DatasetInfo]:
    selected = (source or os.getenv("DATA_SOURCE", "local")).strip().lower()
    if selected == "local":
        path = _resolve_local_path()
        df, fingerprint = _load_local(path)
        source_name = str(path)
    elif selected == "snowflake":
        df, fingerprint = _load_snowflake()
        source_name = ".".join(
            [
                os.getenv("SNOWFLAKE_DATABASE", "ECU911"),
                os.getenv("SNOWFLAKE_SCHEMA", "OBT"),
                os.getenv("SNOWFLAKE_TABLE", "OBT_EMERGENCIAS_CANTON_DIA"),
            ]
        )
    else:
        raise ValueError("DATA_SOURCE debe ser 'local' o 'snowflake'")

    required = {
        "cod_canton",
        "fecha",
        "provincia",
        "region",
        "n_emergencias",
        "lag_1d",
        "lag_7d",
        "lag_14d",
        "lag_28d",
        "media_prev_7d",
        "media_prev_28d",
        "obj_n_emergencias_t3",
        "obj_n_emergencias_t7",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"La OBT no cumple el contrato; faltan: {missing}")
    if df.duplicated(["cod_canton", "fecha"]).any():
        raise ValueError("La OBT contiene claves canton-fecha duplicadas")

    for column in BOOLEAN_COLUMNS:
        if column in df:
            if df[column].dtype == object or str(df[column].dtype).startswith("string"):
                df[column] = df[column].astype("string").str.lower().map({"true": True, "false": False})
            df[column] = df[column].astype("boolean")
    df = df.sort_values(["fecha", "cod_canton"]).reset_index(drop=True)
    info = DatasetInfo(
        source=source_name,
        snapshot_end=str(df["fecha"].max().date()),
        dataset_hash=fingerprint,
        rows=len(df),
        columns=len(df.columns),
    )
    return df, info


def agregar_features(df: pd.DataFrame, horizon: int = 3) -> pd.DataFrame:
    if horizon not in (3, 7):
        raise ValueError("El horizonte debe ser 3 o 7 dias")
    work = df.sort_values(["cod_canton", "fecha"]).copy()
    group = work.groupby("cod_canton", sort=False)
    work["lag_21d"] = group["n_emergencias"].shift(21)
    # 364 días conserva el día de semana; 365 aproxima la fecha calendario y
    # 371 permite suavizar con la semana contigua del año anterior.
    work["lag_364d"] = group["n_emergencias"].shift(364)
    work["lag_365d"] = group["n_emergencias"].shift(365)
    work["lag_371d"] = group["n_emergencias"].shift(371)
    work["media_mismo_periodo_anio"] = work[["lag_364d", "lag_365d", "lag_371d"]].mean(axis=1)
    same_weekday = ["lag_7d", "lag_14d", "lag_21d", "lag_28d"]
    work["media_mismo_dia_4"] = work[same_weekday].mean(axis=1)
    work["tendencia_7_28"] = (
        (work["media_prev_7d"] - work["media_prev_28d"]) / (work["media_prev_28d"].abs() + 1.0)
    )
    work["cambio_1_7"] = (work["lag_1d"] - work["lag_7d"]) / (work["lag_7d"].abs() + 1.0)
    work["volatilidad_prev_28d"] = group["n_emergencias"].transform(
        lambda values: values.shift(1).rolling(28, min_periods=7).std()
    )
    work["fecha_objetivo"] = work["fecha"] + pd.to_timedelta(horizon, unit="D")
    work["dia_semana_objetivo"] = work["fecha_objetivo"].dt.dayofweek
    work["mes_objetivo"] = work["fecha_objetivo"].dt.month
    work["trimestre_objetivo"] = work["fecha_objetivo"].dt.quarter
    work["dia_mes_objetivo"] = work["fecha_objetivo"].dt.day
    work["dia_anio_objetivo"] = work["fecha_objetivo"].dt.dayofyear
    work["semana_anio_objetivo"] = work["fecha_objetivo"].dt.isocalendar().week.astype("int16")
    work["es_fin_semana_objetivo"] = work["dia_semana_objetivo"].isin([5, 6])
    work["mes_sin"] = np.sin(2 * np.pi * work["mes_objetivo"] / 12.0)
    work["mes_cos"] = np.cos(2 * np.pi * work["mes_objetivo"] / 12.0)
    work["dia_anio_sin"] = np.sin(2 * np.pi * work["dia_anio_objetivo"] / 365.25)
    work["dia_anio_cos"] = np.cos(2 * np.pi * work["dia_anio_objetivo"] / 365.25)
    work["indice_tiempo"] = (work["fecha"] - work["fecha"].min()).dt.days

    # El calendario de feriados es información conocida al momento de inferencia.
    holiday_calendar = (
        work.loc[work["es_feriado"].fillna(False), ["fecha", "nombre_feriado"]]
        .drop_duplicates("fecha")
        .set_index("fecha")["nombre_feriado"]
    )
    work["nombre_feriado_objetivo"] = (
        work["fecha_objetivo"].map(holiday_calendar).astype("string").fillna("No feriado")
    )
    holiday_dates = np.array(sorted(pd.to_datetime(holiday_calendar.index).unique()), dtype="datetime64[ns]")
    target_dates = work["fecha_objetivo"].to_numpy(dtype="datetime64[ns]")
    if len(holiday_dates):
        positions = np.searchsorted(holiday_dates, target_dates)
        previous_positions = np.clip(positions - 1, 0, len(holiday_dates) - 1)
        next_positions = np.clip(positions, 0, len(holiday_dates) - 1)
        previous_distance = (target_dates - holiday_dates[previous_positions]) / np.timedelta64(1, "D")
        next_distance = (holiday_dates[next_positions] - target_dates) / np.timedelta64(1, "D")
        use_previous = previous_distance <= next_distance
        signed_distance = np.where(use_previous, -previous_distance, next_distance)
        work["dias_al_feriado_mas_cercano"] = np.clip(signed_distance, -30, 30)
    else:  # pragma: no cover - la OBT real siempre contiene calendario
        work["dias_al_feriado_mas_cercano"] = 30.0

    distance = work["dias_al_feriado_mas_cercano"].abs()
    work["es_feriado_objetivo"] = work["nombre_feriado_objetivo"].ne("No feriado")
    work["es_ventana_feriado_objetivo"] = distance <= 2
    work["fase_feriado"] = np.select(
        [
            distance.eq(0),
            work["dias_al_feriado_mas_cercano"].between(-2, -1),
            work["dias_al_feriado_mas_cercano"].between(1, 2),
            distance.between(3, 7),
        ],
        ["feriado", "despues_feriado", "antes_feriado", "cercano_3_7_dias"],
        default="regular",
    )
    holiday_name = work["nombre_feriado_objetivo"].str.lower()
    end_of_year = (
        ((work["mes_objetivo"] == 12) & (work["dia_mes_objetivo"] >= 20))
        | ((work["mes_objetivo"] == 1) & (work["dia_mes_objetivo"] <= 7))
    )
    work["temporada_objetivo"] = np.select(
        [
            end_of_year,
            holiday_name.str.contains("carnaval", na=False),
            holiday_name.str.contains("viernes santo", na=False),
            work["es_feriado_objetivo"],
        ],
        ["fin_anio", "carnaval", "semana_santa", "otro_feriado"],
        default="regular",
    )
    canton_name = work["canton"].astype("string").str.upper()
    work["es_feriado_local_objetivo"] = (
        (holiday_name.str.contains("guayaquil", na=False) & canton_name.str.contains("GUAYAQUIL", na=False))
        | (holiday_name.str.contains("cuenca", na=False) & canton_name.str.contains("CUENCA", na=False))
    )
    work["horizonte"] = horizon
    return work.sort_values(["fecha", "cod_canton"]).reset_index(drop=True)


def _threshold_map(train: pd.DataFrame, target_count: str) -> tuple[pd.Series, pd.Series, float]:
    by_group = train.groupby(["cod_canton", "dia_semana_objetivo"], observed=True)[target_count].quantile(0.90)
    by_weekday = train.groupby("dia_semana_objetivo", observed=True)[target_count].quantile(0.90)
    overall = float(train[target_count].quantile(0.90))
    return by_group, by_weekday, overall


def _apply_thresholds(
    frame: pd.DataFrame,
    by_group: pd.Series,
    by_weekday: pd.Series,
    overall: float,
) -> pd.Series:
    index = pd.MultiIndex.from_frame(frame[["cod_canton", "dia_semana_objetivo"]])
    values = pd.Series(by_group.reindex(index).to_numpy(), index=frame.index, dtype="float64")
    values = values.fillna(frame["dia_semana_objetivo"].map(by_weekday)).fillna(overall)
    return values


def preparar_modelado(
    df: pd.DataFrame,
    horizon: int = 3,
    holdout_start: str = "2026-04-01",
) -> ModelData:
    work = agregar_features(df, horizon=horizon)
    target_count = f"obj_n_emergencias_t{horizon}"
    work = work.loc[work[target_count].notna()].copy()
    holdout_timestamp = pd.Timestamp(holdout_start)
    # Una fila de entrenamiento solo se admite si su fecha objetivo también
    # termina antes del holdout. Esto crea la purga t+h necesaria en el borde.
    train_mask = work["fecha_objetivo"] < holdout_timestamp
    test_mask = work["fecha"] >= holdout_timestamp
    if not train_mask.any() or not test_mask.any():
        raise ValueError("HOLDOUT_START debe dejar observaciones en entrenamiento y prueba")

    by_group, by_weekday, overall = _threshold_map(work.loc[train_mask], target_count)
    work["p90_entrenamiento"] = _apply_thresholds(work, by_group, by_weekday, overall)
    target_column = f"alta_demanda_t{horizon}"
    work[target_column] = (work[target_count] > work["p90_entrenamiento"]).astype("int8")
    work["score_regla_negocio"] = work["media_mismo_dia_4"] / (work["p90_entrenamiento"] + 1.0)
    # Normalizar por el nivel historico de cada canton evita exigir al modelo que
    # descubra por si solo que 20 eventos pueden ser altos en un canton y bajos en otro.
    denominator = work["p90_entrenamiento"] + 1.0
    work["ratio_estacional_p90"] = work["score_regla_negocio"]
    work["ratio_lag_1_p90"] = work["lag_1d"] / denominator
    work["ratio_media_7_p90"] = work["media_prev_7d"] / denominator
    work["ratio_max_lag_p90"] = work[["lag_1d", "lag_7d", "lag_14d", "lag_21d", "lag_28d"]].max(axis=1) / denominator
    work["ratio_anual_p90"] = work["media_mismo_periodo_anio"] / denominator

    numeric = [
        "lag_1d",
        "lag_7d",
        "lag_14d",
        "lag_21d",
        "lag_28d",
        "lag_364d",
        "lag_365d",
        "lag_371d",
        "media_prev_7d",
        "media_prev_28d",
        "media_mismo_dia_4",
        "media_mismo_periodo_anio",
        "tendencia_7_28",
        "cambio_1_7",
        "volatilidad_prev_28d",
        "p90_entrenamiento",
        "ratio_estacional_p90",
        "ratio_lag_1_p90",
        "ratio_media_7_p90",
        "ratio_max_lag_p90",
        "ratio_anual_p90",
        "mes_objetivo",
        "trimestre_objetivo",
        "dia_mes_objetivo",
        "semana_anio_objetivo",
        "mes_sin",
        "mes_cos",
        "dia_anio_sin",
        "dia_anio_cos",
        "indice_tiempo",
        "dias_al_feriado_mas_cercano",
        "dia_semana",
        "dia_semana_objetivo",
        "es_fin_de_semana",
        "es_fin_semana_objetivo",
        "es_feriado",
        "es_feriado_objetivo",
        "es_ventana_feriado_objetivo",
        "es_feriado_local_objetivo",
        "es_periodo_anomalo",
        f"es_feriado_t{horizon}",
    ]
    categorical = [
        "cod_canton",
        "provincia",
        "region",
        "nombre_dia",
        "nombre_feriado_objetivo",
        "fase_feriado",
        "temporada_objetivo",
    ]
    features = numeric + categorical
    for column in numeric:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    for column in categorical:
        work[column] = work[column].astype("string")

    return ModelData(
        frame=work,
        train=work.loc[train_mask].copy(),
        test=work.loc[test_mask].copy(),
        feature_columns=features,
        numeric_columns=numeric,
        categorical_columns=categorical,
        target_column=target_column,
        target_count_column=target_count,
        horizon=horizon,
        holdout_start=holdout_start,
        purge_days=horizon,
    )


def cortes_temporales(
    fechas: Iterable[pd.Timestamp],
    n_splits: int = 4,
    horizon: int = 3,
    validation_months: int = 6,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Backtests expansivos con semestres completos y una purga de ``horizon`` días."""
    dates = pd.to_datetime(pd.Series(fechas)).reset_index(drop=True)
    unique_dates = np.array(sorted(dates.unique()))
    if len(unique_dates) < (n_splits + 1) * 30:
        raise ValueError("No hay suficientes fechas para la validacion temporal")

    minimum_date = pd.Timestamp(unique_dates.min())
    maximum_date = pd.Timestamp(unique_dates.max())
    half_year_starts = []
    for year in range(minimum_date.year, maximum_date.year + 1):
        for month in (1, 7):
            start = pd.Timestamp(year=year, month=month, day=1)
            end = start + pd.DateOffset(months=validation_months)
            if start >= minimum_date + pd.DateOffset(years=1) and end - pd.Timedelta(days=1) <= maximum_date:
                half_year_starts.append(start)

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    if len(half_year_starts) >= n_splits:
        for start in half_year_starts[-n_splits:]:
            end = start + pd.DateOffset(months=validation_months)
            train_limit = start - pd.Timedelta(days=horizon)
            train_idx = np.flatnonzero((dates < train_limit).to_numpy())
            validation_idx = np.flatnonzero(((dates >= start) & (dates < end)).to_numpy())
            if len(train_idx) and len(validation_idx):
                splits.append((train_idx, validation_idx))
    else:
        # Fallback para pruebas unitarias pequeñas: bloques contiguos igualmente
        # espaciados, conservando la misma purga temporal.
        blocks = np.array_split(unique_dates, n_splits + 1)
        for index in range(1, n_splits + 1):
            validation_dates = blocks[index]
            validation_start = pd.Timestamp(validation_dates.min())
            train_limit = validation_start - pd.Timedelta(days=horizon)
            train_idx = np.flatnonzero((dates < train_limit).to_numpy())
            validation_idx = np.flatnonzero(dates.isin(validation_dates).to_numpy())
            if len(train_idx) and len(validation_idx):
                splits.append((train_idx, validation_idx))
    if len(splits) != n_splits:
        raise ValueError(f"Solo se pudieron construir {len(splits)} de {n_splits} backtests")
    return splits
