-- =============================================================================
-- GOLD.FCT_EMERGENCIAS_CANTON_DIA
-- Grain: 1 fila = 1 cantón en 1 día. Incluye los días SIN emergencias (0):
-- el date spine (todos los cantones x todos los días con datos) crea esas filas.
-- =============================================================================

with conteos as (

    select
        cod_canton,
        fecha,
        count(*)                                        as n_emergencias,
        count_if(servicio = 'Seguridad Ciudadana')      as n_seguridad_ciudadana,
        count_if(servicio = 'Gestión Sanitaria')        as n_gestion_sanitaria,
        count_if(servicio = 'Tránsito y Movilidad')     as n_transito_movilidad,
        count_if(servicio = 'Servicios Municipales')    as n_servicios_municipales,
        count_if(servicio = 'Gestión de Siniestros')    as n_gestion_siniestros,
        count_if(servicio = 'Servicio Militar')         as n_servicio_militar,
        count_if(servicio = 'Gestión de Riesgos')       as n_gestion_riesgos,
        count_if(servicio is null)                      as n_sin_servicio
    from {{ ref('emergencias') }}
    where es_ubicacion_valida           -- sin cantón no se puede asignar
    group by cod_canton, fecha

),

-- Rango de días con datos publicados (no se inventan ceros en meses aún no publicados)
rango as (

    select min(fecha) as desde, max(fecha) as hasta from conteos

),

spine as (

    select c.cod_canton, f.fecha
    from {{ ref('dim_canton') }} c
    cross join {{ ref('dim_fecha') }} f
    join rango r on f.fecha between r.desde and r.hasta

)

select
    s.cod_canton,
    s.fecha,
    coalesce(x.n_emergencias, 0)            as n_emergencias,
    coalesce(x.n_seguridad_ciudadana, 0)    as n_seguridad_ciudadana,
    coalesce(x.n_gestion_sanitaria, 0)      as n_gestion_sanitaria,
    coalesce(x.n_transito_movilidad, 0)     as n_transito_movilidad,
    coalesce(x.n_servicios_municipales, 0)  as n_servicios_municipales,
    coalesce(x.n_gestion_siniestros, 0)     as n_gestion_siniestros,
    coalesce(x.n_servicio_militar, 0)       as n_servicio_militar,
    coalesce(x.n_gestion_riesgos, 0)        as n_gestion_riesgos,
    coalesce(x.n_sin_servicio, 0)           as n_sin_servicio

from spine s
left join conteos x
    on  s.cod_canton = x.cod_canton
    and s.fecha      = x.fecha
