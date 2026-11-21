-- Agregar no debe perder ni inventar emergencias:
-- suma del hecho = incidentes de Silver con ubicación válida.
select
    (select sum(n_emergencias) from {{ ref('fct_emergencias_canton_dia') }})  as total_gold,
    (select count(*) from {{ ref('emergencias') }} where es_ubicacion_valida)  as total_silver
where total_gold <> total_silver
