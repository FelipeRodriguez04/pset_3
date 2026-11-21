-- Regla: las filas sin cantón asignable son marginales (hoy 0.012%).
-- Si superan el 0.1%, la fuente cambió (columnas corridas, códigos nuevos...) y el conteo
-- cantón-día estaría subestimado.
select count_if(not es_ubicacion_valida) as sin_ubicacion,
       count(*)                          as total
from {{ ref('emergencias') }}
having count_if(not es_ubicacion_valida) / count(*) > 0.001
