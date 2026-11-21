#!/bin/sh
# -----------------------------------------------------------------------------
# Despliega kestra/flows/*.yml en Kestra usando su API y luego sube la
# carpeta dbt/ y los jobs de spark/ como namespace files (flows dbt_build y spark_obt).
# Lo ejecuta el servicio "kestra-init" de docker-compose al levantar el stack.
#
# Por qué no usamos --flow-path: en Kestra 1.3 ese loader corre ANTES de que
# se registren los plugins externos (python, dbt, snowflake...) y rechaza los
# flows con "Invalid type". La importación por API se realiza después del arranque.
#
# Ojo: la API /flows/import responde HTTP 200 incluso si rechaza un flow;
# los rechazados vienen listados en el cuerpo. Por eso revisamos el cuerpo.
# -----------------------------------------------------------------------------
set -eu

URL="${KESTRA_URL:-http://kestra:8080}"
AUTH="${KESTRA_ADMIN_USER}:${KESTRA_ADMIN_PASSWORD}"

echo "Esperando a que la API de Kestra responda en $URL ..."
i=0
until [ "$(curl -s -o /dev/null -w '%{http_code}' -u "$AUTH" "$URL/api/v1/main/flows/search")" = "200" ]; do
  i=$((i + 1))
  if [ "$i" -ge 60 ]; then
    echo "ERROR: Kestra no respondió en 5 minutos." >&2
    exit 1
  fi
  sleep 5
done
echo "Kestra responde."

fallidos=0
for f in /kestra/flows/*.yml; do
  intento=1
  while :; do
    body=$(curl -s -u "$AUTH" -X POST -F "fileUpload=@$f" "$URL/api/v1/main/flows/import")
    if [ "$body" = "[]" ]; then
      echo "OK         $f"
      break
    fi
    # Reintento corto por si los plugins aún se están registrando.
    if [ "$intento" -ge 3 ]; then
      echo "RECHAZADO  $f -> $body" >&2
      fallidos=$((fallidos + 1))
      break
    fi
    intento=$((intento + 1))
    sleep 10
  done
done

if [ "$fallidos" -gt 0 ]; then
  echo "ERROR: $fallidos flow(s) rechazados. Revisar el YAML o los logs de Kestra." >&2
  exit 1
fi
echo "Todos los flows desplegados."

# -----------------------------------------------------------------------------
# Proyecto dbt -> "namespace files" de ecu911 (los usa el flow dbt_build).
# Se omite lo que genera dbt al correr (target, logs, dbt_packages, .user.yml).
# -----------------------------------------------------------------------------
subidos=0
for f in $(find /dbt -type f \( -name '*.sql' -o -name '*.yml' -o -name '*.csv' \) \
             -not -path '*/target/*' -not -path '*/logs/*' -not -path '*/dbt_packages/*' -not -name '.user.yml'); do
  code=$(curl -s -o /dev/null -w '%{http_code}' -u "$AUTH" -X POST \
    -F "fileContent=@$f" "$URL/api/v1/main/namespaces/ecu911/files?path=$f")
  if [ "$code" != "200" ]; then
    echo "ERROR subiendo $f (HTTP $code)" >&2
    exit 1
  fi
  subidos=$((subidos + 1))
done
echo "Proyecto dbt subido a namespace files: $subidos archivos."

# Jobs de Spark -> namespace files (los usa el flow spark_obt)
for f in /spark/jobs/*.py; do
  code=$(curl -s -o /dev/null -w '%{http_code}' -u "$AUTH" -X POST \
    -F "fileContent=@$f" "$URL/api/v1/main/namespaces/ecu911/files?path=$f")
  if [ "$code" != "200" ]; then
    echo "ERROR subiendo $f (HTTP $code)" >&2
    exit 1
  fi
done
echo "Jobs de Spark subidos a namespace files."
