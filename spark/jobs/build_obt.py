"""
One Big Table (OBT) con Spark.

Lee el star schema de GOLD (Snowflake), une el hecho con sus dimensiones,
agrega features de series de tiempo por cantón y escribe OBT.OBT_EMERGENCIAS_CANTON_DIA.

Grain: 1 fila = 1 cantón x 1 día (el mismo del hecho). Los joins NO pueden cambiar
el número de filas: se valida antes de escribir y el job falla si algo no cuadra.

Convención temporal (para no filtrar información del futuro al modelo):
  - cada fila representa la información disponible al CIERRE del día t;
  - n_emergencias (y las columnas n_*) es el volumen del propio día t;
  - lags (lag_*) y medias (media_prev_*) usan SOLO días anteriores a t;
    una media con ventana incompleta (primeros k días de cada cantón) queda NULL;
  - es_feriado_t3/t7 son futuro conocido de antemano (calendario), se pueden usar;
  - las columnas obj_* son insumos del target (volumen real en t+3/t+7):
    NO se usan como features.

Ejecutar (desde la carpeta del proyecto):
  docker compose exec spark-master /opt/spark/bin/spark-submit \
    --packages net.snowflake:spark-snowflake_2.12:3.2.2-spark_3.5 \
    --conf spark.jars.ivy=/tmp/.ivy2 /opt/spark-jobs/build_obt.py
"""
import os
import sys

from pyspark.sql import SparkSession, Window, functions as F

SNOWFLAKE = "net.snowflake.spark.snowflake"


def opciones(schema):
    """Conexión a Snowflake: credenciales desde variables de entorno (.env)."""
    return {
        "sfURL": f"{os.environ['SNOWFLAKE_ACCOUNT']}.snowflakecomputing.com",
        "sfUser": os.environ["SNOWFLAKE_USER"],
        "sfPassword": os.environ["SNOWFLAKE_PASSWORD"],
        "sfRole": os.environ["SNOWFLAKE_ROLE"],
        "sfWarehouse": os.environ["SNOWFLAKE_WAREHOUSE"],
        "sfDatabase": os.environ["SNOWFLAKE_DATABASE"],
        "sfSchema": schema,
    }


def leer(spark, tabla):
    df = spark.read.format(SNOWFLAKE).options(**opciones("GOLD")).option("dbtable", tabla).load()
    # Snowflake devuelve los nombres en MAYÚSCULAS; se pasan a minúsculas
    return df.toDF(*[c.lower() for c in df.columns])


spark = SparkSession.builder.appName("pset2-obt").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# ---------------------------------------------------------------------------
# 1) Leer GOLD
# ---------------------------------------------------------------------------
fct = leer(spark, "FCT_EMERGENCIAS_CANTON_DIA")
dim_canton = leer(spark, "DIM_CANTON")
dim_fecha = leer(spark, "DIM_FECHA")

filas_fct = fct.count()
suma_fct = fct.agg(F.sum("n_emergencias")).first()[0]
print(f"[leer] hecho={filas_fct:,} filas, suma={suma_fct:,} | "
      f"dim_canton={dim_canton.count()} | dim_fecha={dim_fecha.count():,}")

# ---------------------------------------------------------------------------
# 2) Joins (LEFT: si una clave no existe en la dimensión, aparece NULL y la
#    validación lo detecta en vez de perder la fila en silencio).
#    Dimensiones pequeñas -> broadcast: se copian a cada worker, sin shuffle.
# ---------------------------------------------------------------------------
obt = (fct
       .join(F.broadcast(dim_canton), "cod_canton", "left")
       .join(F.broadcast(dim_fecha), "fecha", "left"))

# Feriados en t+3 y t+7: se buscan en DIM_FECHA (calendario hasta 2026-12-31), no con
# lead sobre el hecho. lead mira "la fila siguiente" y el hecho termina en el último día
# con datos -> los últimos 3/7 días quedarían NULL aunque el feriado se conoce de antemano.
for h in (3, 7):
    feriado_h = dim_fecha.select(F.col("fecha").alias(f"fecha_t{h}"),
                                 F.col("es_feriado").alias(f"es_feriado_t{h}"))
    obt = obt.join(F.broadcast(feriado_h),
                   F.date_add("fecha", h) == F.col(f"fecha_t{h}"), "left")

# ---------------------------------------------------------------------------
# 3) Features de series de tiempo por cantón.
#    El spine del hecho no tiene huecos (test dbt fct_spine_completo), así que
#    "1 fila atrás" = "1 día atrás".
# ---------------------------------------------------------------------------
w = Window.partitionBy("cod_canton").orderBy("fecha")
n = F.col("n_emergencias")


def media_previa(k):
    """Promedio de los k días ANTERIORES a t (sin incluir t). Si la ventana no tiene
    los k días (inicio de la serie) queda NULL: un promedio de 3 días no es una media de 7.
    n_emergencias nunca es NULL (ceros del spine), así que contar filas = contar días."""
    ventana = w.rowsBetween(-k, -1)
    return F.when(F.count(n).over(ventana) == k, F.avg(n).over(ventana))


obt = (obt
       # pasado: lags y medias móviles con días ANTERIORES a t
       .withColumn("lag_1d", F.lag(n, 1).over(w))
       .withColumn("lag_7d", F.lag(n, 7).over(w))
       .withColumn("lag_14d", F.lag(n, 14).over(w))
       .withColumn("lag_28d", F.lag(n, 28).over(w))     # 4 semanas: mismo día de la semana que t
       .withColumn("media_prev_7d", media_previa(7))
       .withColumn("media_prev_28d", media_previa(28))
       # insumos del target (NO son features): volumen real en t+3 y t+7.
       # El target y_h (> P90 del cantón y día de semana) se calcula en ML con datos de train.
       .withColumn("obj_n_emergencias_t3", F.lead(n, 3).over(w))
       .withColumn("obj_n_emergencias_t7", F.lead(n, 7).over(w)))

columnas = [
    # grain
    "cod_canton", "fecha",
    # dimensión cantón
    "canton", "cod_provincia", "provincia", "region", "es_zona_no_delimitada",
    # dimensión fecha
    "anio", "trimestre", "mes", "periodo", "dia_mes", "dia_semana", "nombre_dia",
    "es_fin_de_semana", "es_feriado", "nombre_feriado", "es_periodo_anomalo",
    # medidas del día
    "n_emergencias", "n_seguridad_ciudadana", "n_gestion_sanitaria", "n_transito_movilidad",
    "n_servicios_municipales", "n_gestion_siniestros", "n_servicio_militar",
    "n_gestion_riesgos", "n_sin_servicio",
    # features
    "lag_1d", "lag_7d", "lag_14d", "lag_28d", "media_prev_7d", "media_prev_28d",
    "es_feriado_t3", "es_feriado_t7",
    # insumos del target
    "obj_n_emergencias_t3", "obj_n_emergencias_t7",
]
obt = obt.select(*columnas).cache()

# ---------------------------------------------------------------------------
# 4) Validaciones: los joins no deben duplicar, perder ni desalinear filas.
# ---------------------------------------------------------------------------
filas_obt = obt.count()
claves_distintas = obt.select("cod_canton", "fecha").distinct().count()
suma_obt = obt.agg(F.sum("n_emergencias")).first()[0]
sin_canton = obt.filter(F.col("canton").isNull()).count()
sin_fecha = obt.filter(F.col("dia_semana").isNull()).count()
nulos_t3 = obt.filter(F.col("es_feriado_t3").isNull()).count()
nulos_t7 = obt.filter(F.col("es_feriado_t7").isNull()).count()
# Si hay nulos es porque dim_fecha no llega a (última fecha + 7): hay que extenderla en dbt
fecha_max = obt.agg(F.max("fecha")).first()[0]
dim_hasta = dim_fecha.agg(F.max("fecha")).first()[0]
cobertura = f"dim_fecha llega a {dim_hasta}, se necesita hasta {fecha_max} + 7 días"

checks = {
    f"filas OBT = filas hecho ({filas_obt:,} vs {filas_fct:,})": filas_obt == filas_fct,
    f"sin duplicados: claves distintas = filas ({claves_distintas:,})": claves_distintas == filas_obt,
    f"suma n_emergencias igual ({suma_obt:,} vs {suma_fct:,})": suma_obt == suma_fct,
    f"todas las filas encontraron su cantón (sin match: {sin_canton})": sin_canton == 0,
    f"todas las filas encontraron su fecha (sin match: {sin_fecha})": sin_fecha == 0,
    f"es_feriado_t3 sin nulos (nulos: {nulos_t3:,}; {cobertura})": nulos_t3 == 0,
    f"es_feriado_t7 sin nulos (nulos: {nulos_t7:,}; {cobertura})": nulos_t7 == 0,
}
for descripcion, ok in checks.items():
    print(f"[validar] {'OK   ' if ok else 'FALLA'} {descripcion}")
if not all(checks.values()):
    print("La OBT no se escribe: hay validaciones fallidas.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 5) Escribir en Snowflake (overwrite: re-ejecutar el job es idempotente)
# ---------------------------------------------------------------------------
(obt.write.format(SNOWFLAKE).options(**opciones("OBT"))
    .option("dbtable", "OBT_EMERGENCIAS_CANTON_DIA")
    .mode("overwrite").save())

escritas = (spark.read.format(SNOWFLAKE).options(**opciones("OBT"))
            .option("query", "SELECT COUNT(*) AS N FROM OBT_EMERGENCIAS_CANTON_DIA").load().first()[0])
print(f"[escribir] OBT.OBT_EMERGENCIAS_CANTON_DIA = {escritas:,} filas")
if escritas != filas_fct:
    print("Las filas escritas no coinciden con las del hecho.")
    sys.exit(1)

print("OBT OK")
spark.stop()
