-- Regla: entre el primer y el último mes cargado no puede faltar ningún mes.
-- El spine de Gold cubre todo ese rango: un mes ausente (ingesta fallida, mes no
-- reintentado) aparecería como 224 cantones x ~30 días con 0 emergencias (ceros falsos).
-- silver_meses_completos no lo ve porque solo revisa los meses que SÍ existen.
-- Devuelve los meses de la secuencia que no tienen filas.
with limites as (
    select to_date(min(periodo) || '-01') as desde,
           to_date(max(periodo) || '-01') as hasta
    from {{ ref('emergencias') }}
),
secuencia as (
    -- 1 fila por mes desde "desde" hasta "hasta" (1200 meses = 100 años de margen)
    select to_char(dateadd(month, row_number() over (order by seq4()) - 1, l.desde), 'YYYY-MM') as periodo,
           l.hasta
    from table(generator(rowcount => 1200))
    cross join limites l
    qualify to_date(periodo || '-01') <= l.hasta
),
con_datos as (
    select distinct periodo from {{ ref('emergencias') }}
)
select s.periodo as periodo_faltante
from secuencia s
left join con_datos c on s.periodo = c.periodo
where c.periodo is null
