-- Regla: el volumen diario promedio de un mes no se aleja más de 20% de la mediana
-- de los meses vecinos (±3). Detecta archivos incompletos o cargas parciales.
-- Por día (no por mes) para que febrero no parezca anómalo.
-- Los meses ya revisados y aceptados se excluyen con var('periodos_anomalos').
-- Umbral: el mes normal más alejado está a ~12%; 2024-01 a ~28.2%.
with mensual as (
    select periodo, count(*) / count(distinct fecha) as por_dia
    from {{ ref('emergencias') }}
    group by periodo
),
comparado as (
    select a.periodo, a.por_dia, median(b.por_dia) as mediana_vecinos
    from mensual a
    join mensual b
      on b.periodo <> a.periodo
     and abs(datediff(month, to_date(a.periodo || '-01'), to_date(b.periodo || '-01'))) <= 3
    group by a.periodo, a.por_dia
)
select periodo, round(por_dia) as por_dia, round(mediana_vecinos) as mediana_vecinos,
       round(100 * (por_dia - mediana_vecinos) / mediana_vecinos, 1) as desviacion_pct
from comparado
where abs(por_dia - mediana_vecinos) / mediana_vecinos > 0.20
  and not ({{ es_periodo_anomalo('periodo') }})
