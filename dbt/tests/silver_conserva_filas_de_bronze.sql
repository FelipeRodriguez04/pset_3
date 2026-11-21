-- Silver no debe perder ni duplicar filas: la limpieza corrige y marca, no borra.
-- Falla (devuelve una fila) si los conteos difieren.
select
    (select count(*) from {{ source('bronze', 'emergencias_raw') }}) as filas_bronze,
    (select count(*) from {{ ref('emergencias') }})                  as filas_silver
where filas_bronze <> filas_silver
