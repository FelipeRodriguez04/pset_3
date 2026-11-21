-- =============================================================================
-- Perfilado de calidad sobre ECU911.BRONZE.EMERGENCIAS_RAW
-- Solo lectura. Ejecutar en Snowsight (warehouse ECU911_WH).
-- Resultados (29-sep-2026, 17,680,253 filas, 62 meses) en el comentario de cada bloque.
-- =============================================================================
USE DATABASE ECU911;

-- Fecha interpretada: d/m/yyyy (CSV) o serial de Excel (XLSX 2026-04 y 2026-05)
-- COALESCE(TRY_TO_DATE(FECHA,'DD/MM/YYYY'), DATEADD(day, TRY_TO_NUMBER(FECHA,10,1)::INT, '1899-12-30'::DATE))

-- 1. COMPLETITUD: nulos o vacíos por columna y cardinalidad
--    PROVINCIA/CANTON/PARROQUIA 2,127 (0.012%), COD_PARROQUIA 2,131, SERVICIO 2,221, SUBTIPO 88, FECHA 0
SELECT 'PROVINCIA' col, COUNT(*) - COUNT(NULLIF(TRIM(PROVINCIA),'')) nulos, COUNT(DISTINCT PROVINCIA) distintos FROM BRONZE.EMERGENCIAS_RAW
UNION ALL SELECT 'CANTON',        COUNT(*) - COUNT(NULLIF(TRIM(CANTON),'')),        COUNT(DISTINCT CANTON)        FROM BRONZE.EMERGENCIAS_RAW
UNION ALL SELECT 'COD_PARROQUIA', COUNT(*) - COUNT(NULLIF(TRIM(COD_PARROQUIA),'')), COUNT(DISTINCT COD_PARROQUIA) FROM BRONZE.EMERGENCIAS_RAW
UNION ALL SELECT 'SERVICIO',      COUNT(*) - COUNT(NULLIF(TRIM(SERVICIO),'')),      COUNT(DISTINCT SERVICIO)      FROM BRONZE.EMERGENCIAS_RAW
UNION ALL SELECT 'SUBTIPO',       COUNT(*) - COUNT(NULLIF(TRIM(SUBTIPO),'')),       COUNT(DISTINCT SUBTIPO)       FROM BRONZE.EMERGENCIAS_RAW
UNION ALL SELECT 'FECHA',         COUNT(*) - COUNT(NULLIF(TRIM(FECHA),'')),         COUNT(DISTINCT FECHA)         FROM BRONZE.EMERGENCIAS_RAW;

-- 2. COMPLETITUD TEMPORAL: meses con días faltantes  -> 0 filas (todos los días presentes)
SELECT _PERIODO, COUNT(DISTINCT TRY_TO_DATE(FECHA,'DD/MM/YYYY')) dias
FROM BRONZE.EMERGENCIAS_RAW WHERE _PERIODO NOT IN ('2026-04','2026-05')
GROUP BY 1 HAVING dias <> DAY(LAST_DAY(TO_DATE(_PERIODO||'-01')));

-- 3. VALIDEZ FECHA: formato, interpretables y dentro del mes del archivo
--    d/m/yyyy 17,120,516 filas; serial 559,737; no interpretables 0; fuera de su mes 0
--    2,066 textos distintos = 1,888 días reales (1,308,320 filas con cero inicial "01/07/2021")
SELECT CASE WHEN FECHA RLIKE '^[0-9]{1,2}/[0-9]{1,2}/[0-9]{4}$' THEN 'd/m/yyyy'
            WHEN FECHA RLIKE '^[0-9]{5}(\\.0)?$' THEN 'serial excel' ELSE 'otro' END formato,
       COUNT(*) filas,
       SUM(IFF(COALESCE(TRY_TO_DATE(FECHA,'DD/MM/YYYY'), DATEADD(day, TRY_TO_NUMBER(FECHA,10,1)::INT, '1899-12-30'::DATE)) IS NULL,1,0)) no_interpretables
FROM BRONZE.EMERGENCIAS_RAW GROUP BY 1;

-- 4. VALIDEZ PROVINCIA: 24 oficiales + ZONA NO DELIMITADA (3,757) + '' (2,127) + 'NULL' texto (17) + '0' (16)
SELECT PROVINCIA, COUNT(*) filas, COUNT(DISTINCT _PERIODO) meses FROM BRONZE.EMERGENCIAS_RAW GROUP BY 1 ORDER BY 2 DESC;

-- 5. VALIDEZ COD_PARROQUIA: 6 dígitos 16,412,546; 5 dígitos 1,265,543 (7.2%, cero inicial perdido, 10 meses)
SELECT LENGTH(COD_PARROQUIA) largo, COUNT(*) filas FROM BRONZE.EMERGENCIAS_RAW GROUP BY 1 ORDER BY 2 DESC;

-- 6. CONSISTENCIA código DPA vs nombre (tras LPAD a 6):
--    prefijo de provincia coincide con el nombre salvo '09' que incluye 44 filas de MORONA SANTIAGO (código 090150 por defecto)
SELECT LEFT(LPAD(COD_PARROQUIA,6,'0'),2) cod_prov, LISTAGG(DISTINCT PROVINCIA,' | ') provincias, COUNT(*) filas
FROM BRONZE.EMERGENCIAS_RAW WHERE COD_PARROQUIA RLIKE '^[0-9]{5,6}$' GROUP BY 1 ORDER BY 1;

-- 7. CONSISTENCIA nombres de cantón: homónimos (BOLIVAR 0402/1302, OLMEDO 1116/1318)
--    y variantes (LOGROÑO 234 filas / LOGROÐO 4,637 filas, error de codificación en la fuente)
WITH b AS (SELECT PROVINCIA, CANTON, LEFT(LPAD(COD_PARROQUIA,6,'0'),4) cod_canton
           FROM BRONZE.EMERGENCIAS_RAW WHERE COD_PARROQUIA RLIKE '^[0-9]{5,6}$')
SELECT CANTON, LISTAGG(DISTINCT cod_canton||' ('||PROVINCIA||')',' | ') codigos, COUNT(*) filas
FROM b GROUP BY 1 HAVING COUNT(DISTINCT cod_canton) > 1 ORDER BY 3 DESC;

-- 8. DUPLICADOS EXACTOS (7 columnas): 11,747,431 filas repetidas (66.4%) -> NO son errores (no hay ID)
SELECT COUNT(*) - COUNT(DISTINCT FECHA,PROVINCIA,CANTON,COD_PARROQUIA,PARROQUIA,SERVICIO,SUBTIPO) repetidas
FROM BRONZE.EMERGENCIAS_RAW;

-- 9. PRECISIÓN / ANOMALÍA DE VOLUMEN: 2024-01 = 208,194 filas (~30% bajo sus vecinos),
--    20 de 31 días bajo el 70% del promedio diario de 2023-12, todas las provincias afectadas
SELECT _PERIODO, COUNT(*) filas FROM BRONZE.EMERGENCIAS_RAW GROUP BY 1 ORDER BY 1;

-- 10. COBERTURA cantón-día: 394,903 combinaciones con datos de 424,800 esperadas (225 x 1,888): 7.0% sin filas
--     13 cantones con < 1 emergencia/día (en promedio 1,074 de 1,888 días sin filas)
WITH cd AS (SELECT DISTINCT PROVINCIA, CANTON,
              COALESCE(TRY_TO_DATE(FECHA,'DD/MM/YYYY'), DATEADD(day, TRY_TO_NUMBER(FECHA,10,1)::INT, '1899-12-30'::DATE)) f
            FROM BRONZE.EMERGENCIAS_RAW WHERE NULLIF(TRIM(CANTON),'') IS NOT NULL AND PROVINCIA NOT IN ('0','NULL'))
SELECT COUNT(*) con_datos, COUNT(DISTINCT PROVINCIA||CANTON) * 1888 esperadas FROM cd;
