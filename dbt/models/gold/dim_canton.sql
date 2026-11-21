-- =============================================================================
-- GOLD.DIM_CANTON — 1 fila = 1 cantón (código DPA del INEC, 4 dígitos).
-- Si un código tuviera dos nombres, saldrían 2 filas y el test unique lo detecta.
-- =============================================================================

select
    cod_canton,
    canton,
    cod_provincia,
    provincia,
    case
        when cod_provincia in ('07', '08', '09', '12', '13', '23', '24')             then 'Costa'
        when cod_provincia in ('01', '02', '03', '04', '05', '06', '10', '11', '17', '18') then 'Sierra'
        when cod_provincia in ('14', '15', '16', '19', '21', '22')                   then 'Amazonía'
        when cod_provincia = '20'                                                    then 'Insular'
        when cod_provincia = '90'                                                    then 'No delimitada'
    end                                 as region,
    cod_provincia = '90'                as es_zona_no_delimitada

from {{ ref('emergencias') }}
where es_ubicacion_valida
group by cod_canton, canton, cod_provincia, provincia
