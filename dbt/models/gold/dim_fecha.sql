-- =============================================================================
-- GOLD.DIM_FECHA — 1 fila = 1 día calendario, años completos 2021-2026.
-- Cubre días futuros (sep-dic 2026) para poder usarla al predecir.
-- =============================================================================

with dias as (

    select dateadd(day, row_number() over (order by seq4()) - 1, '2021-01-01'::date) as fecha
    from table(generator(rowcount => 2200))
    qualify fecha <= '2026-12-31'::date

),

feriados as (

    select * from {{ ref('feriados_ecuador') }}

)

select
    d.fecha,
    year(d.fecha)                                   as anio,
    quarter(d.fecha)                                as trimestre,
    month(d.fecha)                                  as mes,
    to_char(d.fecha, 'YYYY-MM')                     as periodo,
    day(d.fecha)                                    as dia_mes,
    dayofweekiso(d.fecha)                           as dia_semana,      -- 1 = lunes ... 7 = domingo
    decode(dayofweekiso(d.fecha), 1, 'Lunes', 2, 'Martes', 3, 'Miércoles', 4, 'Jueves',
                                  5, 'Viernes', 6, 'Sábado', 7, 'Domingo') as nombre_dia,
    dayofweekiso(d.fecha) >= 6                      as es_fin_de_semana,
    f.fecha is not null                             as es_feriado,      -- día de descanso obligatorio nacional
    f.nombre_feriado,
    {{ es_periodo_anomalo("to_char(d.fecha, 'YYYY-MM')") }}  as es_periodo_anomalo

from dias d
left join feriados f on d.fecha = f.fecha
