"""LLM por prompting con Ollama local (qwen2.5:3b por defecto).

El modelo recibe la misma información que tendría un operador al cierre del día t
(últimas 4 semanas del cantón, P90 histórico del día de la semana objetivo, feriado)
y devuelve una probabilidad en JSON. Nunca recibe el volumen real de t+h.
"""
from __future__ import annotations

import json
import os
import re
import time

import pandas as pd
import requests

MODELO = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
ESQUEMA = {"type": "object", "properties": {"probabilidad": {"type": "number"}},
           "required": ["probabilidad"]}

PLANTILLA = """Eres analista del ECU 911 (Ecuador). Debes estimar si el cantón {canton} tendrá \
demanda inusualmente alta de emergencias dentro de {h} días.

Definición: hay demanda alta si las emergencias de ese día superan el percentil 90 histórico \
del cantón para ese día de la semana. Para el día objetivo ({dia_obj}) ese umbral es {p90:.0f} emergencias.
El día objetivo {feriado}.

Emergencias diarias de los últimos {n} días (de la más antigua a la más reciente; el último valor es hoy):
{historia}

Responde SOLO con un JSON: {{"probabilidad": p}}, con p entre 0 y 1."""


def construir_prompt(fila: dict) -> str:
    historia = ", ".join(str(int(v)) for v in fila["historia"])
    feriado = "es feriado nacional" if fila["feriado_obj"] else "no es feriado"
    return PLANTILLA.format(canton=fila["canton"], h=fila["h"], dia_obj=fila["dia_obj"],
                            p90=fila["p90"], feriado=feriado, n=len(fila["historia"]),
                            historia=historia)


def parsear_respuesta(texto: str) -> float | None:
    """Probabilidad en [0, 1] o None si la respuesta no es válida."""
    m = re.search(r"\{.*?\}", texto, flags=re.S)
    if not m:
        return None
    try:
        p = json.loads(m.group(0)).get("probabilidad")
    except json.JSONDecodeError:
        return None
    if isinstance(p, bool) or not isinstance(p, (int, float)):
        return None
    return float(p) if 0 <= p <= 1 else None


def predecir_llm(filas: list[dict], modelo: str = MODELO, url: str = URL, timeout: int = 120) -> pd.DataFrame:
    salida = []
    for fila in filas:
        t0 = time.perf_counter()
        r = requests.post(f"{url}/api/generate", timeout=timeout, json={
            "model": modelo, "prompt": construir_prompt(fila), "stream": False, "format": ESQUEMA,
            "options": {"temperature": 0, "seed": 42, "num_predict": 40},
        })
        r.raise_for_status()
        texto = r.json().get("response", "")
        p = parsear_respuesta(texto)
        salida.append({"score": p, "valido": p is not None, "latencia_s": time.perf_counter() - t0,
                       "respuesta": texto[:200]})
    return pd.DataFrame(salida)


def liberar_llm(modelo: str = MODELO, url: str = URL) -> None:
    """Pide a Ollama descargar el modelo de memoria ya (por defecto lo retiene minutos)."""
    r = requests.post(f"{url}/api/generate", json={"model": modelo, "keep_alive": 0}, timeout=60)
    r.raise_for_status()
