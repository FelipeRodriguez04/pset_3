-- Regla: cada archivo mensual trae TODOS los días de su mes.
-- Si falta un día, el hecho tendría ceros falsos en todos los cantones ese día.
select periodo, count(distinct fecha) as dias_con_datos
from {{ ref('emergencias') }}
group by periodo
having count(distinct fecha) <> day(last_day(to_date(periodo || '-01')))
