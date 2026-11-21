-- Condición "el periodo está en var('periodos_anomalos')".
-- Con la lista vacía devuelve false: "periodo in ()" no es SQL válido en Snowflake.
{% macro es_periodo_anomalo(columna) -%}
    {%- set periodos = var('periodos_anomalos') -%}
    {%- if periodos | length == 0 -%}
        false
    {%- else -%}
        {{ columna }} in ({% for p in periodos %}'{{ p }}'{% if not loop.last %}, {% endif %}{% endfor %})
    {%- endif -%}
{%- endmacro %}
