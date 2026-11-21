-- Limpieza común de columnas de texto:
--   TRIM de espacios sobrantes
--   'Ð' -> 'Ñ' (error de codificación de la fuente: LOGROÐO -> LOGROÑO)
--   '', 'NULL' (texto) y '0' se convierten en NULL real
{% macro limpiar_texto(columna) -%}
    nullif(nullif(nullif(replace(trim({{ columna }}), 'Ð', 'Ñ'), ''), 'NULL'), '0')
{%- endmacro %}
