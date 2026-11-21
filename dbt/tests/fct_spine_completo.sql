-- Date spine completo: filas del hecho = nº de cantones x nº de días del rango.
-- Si falta un cantón-día, el 0 de ese día se perdería y el P90 quedaría sesgado.
with h as (
    select count(*)                                   as filas,
           count(distinct cod_canton)                 as cantones,
           datediff(day, min(fecha), max(fecha)) + 1  as dias
    from {{ ref('fct_emergencias_canton_dia') }}
)
select * from h
where filas <> cantones * dias
   or cantones <> (select count(*) from {{ ref('dim_canton') }})
