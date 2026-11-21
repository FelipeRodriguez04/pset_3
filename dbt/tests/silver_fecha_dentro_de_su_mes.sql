-- Cada incidente debe caer dentro del mes del archivo que lo trajo.
-- Detecta errores de interpretación de fecha (ej. confundir día y mes: 1/7 vs 7/1).
select incidente_id, fecha, periodo
from {{ ref('emergencias') }}
where to_char(fecha, 'YYYY-MM') <> periodo
