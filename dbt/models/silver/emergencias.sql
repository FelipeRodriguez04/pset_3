-- =============================================================================
-- SILVER.EMERGENCIAS
-- Grain: 1 fila = 1 incidente (= 1 fila de BRONZE). No se elimina ninguna fila:
-- los problemas se corrigen o se marcan con banderas (es_*), nunca se borran.
-- Las decisiones de limpieza se documentan en docs/data_quality.sql.
-- =============================================================================

with bronze as (

    select * from {{ source('bronze', 'emergencias_raw') }}

),

limpio as (

    select
        -- ID técnico: la fuente no trae ID. Mes + nº de fila del archivo es único
        -- porque cada mes se carga completo con DELETE + COPY (idempotente).
        _periodo || '-' || _fila_archivo                                   as incidente_id,

        -- Dos formatos: texto d/m/yyyy (CSV) o número serial de Excel (XLSX)
        coalesce(
            try_to_date(trim(fecha), 'DD/MM/YYYY'),
            dateadd(day, try_to_number(trim(fecha), 10, 1)::int, '1899-12-30'::date)
        )                                                                  as fecha,

        -- Vacíos/'NULL'/'0' -> NULL, 'Ð' -> 'Ñ', TRIM
        {{ limpiar_texto('provincia') }}                                   as provincia,
        {{ limpiar_texto('canton') }}                                      as canton,
        {{ limpiar_texto('parroquia') }}                                   as parroquia,
        {{ limpiar_texto('servicio') }}                                    as servicio,
        {{ limpiar_texto('subtipo') }}                                     as subtipo,

        -- El código DPA tiene 6 dígitos; 7.2% de filas perdió el cero inicial
        iff(trim(cod_parroquia) rlike '^[0-9]{5,6}$',
            lpad(trim(cod_parroquia), 6, '0'), null)                       as cod_parroquia,

        -- linaje (de dónde vino cada fila)
        _periodo                                                           as periodo,
        _fila_archivo                                                      as fila_archivo,
        _url_origen                                                        as url_origen,
        _cargado_en                                                        as cargado_en,
        _ejecucion_kestra                                                  as ejecucion_kestra

    from bronze

),

-- Clave del cantón = código DPA de 4 dígitos MÁS FRECUENTE para (provincia, cantón).
--   - evita juntar homónimos (BOLIVAR de Carchi y de Manabí) -> no se usa solo el nombre
--   - corrige códigos por defecto (090150 en filas de MORONA SANTIAGO)
codigo_canton as (

    select
        provincia,
        canton,
        left(cod_parroquia, 4)  as cod_canton
    from limpio
    where provincia is not null
      and canton is not null
      and cod_parroquia is not null
    group by 1, 2, 3
    qualify row_number() over (
        partition by provincia, canton
        order by count(*) desc, cod_canton
    ) = 1

),

-- Cambios de límites: parroquias que hoy son otro cantón (seed cantones_reasignados).
--     Se usa la geografía VIGENTE para que la serie de cada cantón sea comparable en el tiempo.
reasignados as (

    select * from {{ ref('cantones_reasignados') }}

)

select
    l.incidente_id,
    l.fecha,
    l.periodo,
    l.provincia,
    coalesce(r.canton_vigente, l.canton)                            as canton,
    l.parroquia,
    left(c.cod_canton, 2)                                           as cod_provincia,
    coalesce(r.cod_canton_vigente, c.cod_canton)                    as cod_canton,
    l.cod_parroquia,
    l.servicio,
    l.subtipo,

    -- banderas de calidad
    c.cod_canton is not null                                        as es_ubicacion_valida,
    coalesce(left(l.cod_parroquia, 4) <> c.cod_canton, false)       as es_codigo_corregido,
    r.cod_canton_vigente is not null                                as es_canton_reasignado,
    {{ es_periodo_anomalo("l.periodo") }}                          as es_periodo_anomalo,

    l.fila_archivo,
    l.url_origen,
    l.cargado_en,
    l.ejecucion_kestra

from limpio l
left join codigo_canton c
    on  l.provincia = c.provincia
    and l.canton    = c.canton
left join reasignados r
    on  l.provincia = r.provincia
    and l.canton    = r.canton_fuente
    and l.parroquia = r.parroquia
