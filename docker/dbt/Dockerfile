FROM python:3.11-slim-bookworm

# dbt debug y las dependencias de proyectos dbt necesitan git.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# La imagen base tiene variantes AMD64 y ARM64; Docker elige la del equipo.
COPY requirements.txt /tmp/dbt-requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/dbt-requirements.txt \
    && python -m pip check

WORKDIR /usr/app/dbt
ENV DBT_PROFILES_DIR=/usr/app/dbt
ENTRYPOINT ["dbt"]
CMD ["--version"]
