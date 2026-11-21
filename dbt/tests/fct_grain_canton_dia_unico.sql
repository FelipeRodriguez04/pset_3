-- Grain del hecho: no puede haber dos filas para el mismo cantón y día.
select cod_canton, fecha, count(*) as filas
from {{ ref('fct_emergencias_canton_dia') }}
group by cod_canton, fecha
having count(*) > 1
