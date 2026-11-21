-- Las columnas por servicio deben sumar el total de cada cantón-día
-- (detecta un servicio nuevo o renombrado en la fuente que no tenga columna).
select *
from {{ ref('fct_emergencias_canton_dia') }}
where n_seguridad_ciudadana + n_gestion_sanitaria + n_transito_movilidad + n_servicios_municipales
    + n_gestion_siniestros + n_servicio_militar + n_gestion_riesgos + n_sin_servicio
    <> n_emergencias
