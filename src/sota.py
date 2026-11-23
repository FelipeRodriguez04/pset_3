"""
Modelos fundacionales (familia "state of the art"), evaluados zero-shot con la
misma validación temporal y la misma métrica que los candidatos tabulares.

1) Chronos-Bolt (amazon/chronos-bolt-small): modelo fundacional generativo de series
   de tiempo. Para cada (cantón, t) recibe la serie diaria hasta t (sin el futuro),
   pronostica los cuantiles de n(t+h) y la probabilidad de pico se estima como
   P(n(t+h) > umbral) interpolando la distribución de cuantiles.
2) LLM por prompting (Qwen2.5-0.5B-Instruct, local): se describe la historia del
   cantón en texto y se lee la probabilidad del token "Sí" vs "No".
No hay API key de un LLM comercial en este entorno; los dos modelos son abiertos y
corren en CPU, de modo que el experimento es reproducible sin credenciales.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CHRONOS_ID = "amazon/chronos-bolt-small"
LLM_ID = "Qwen/Qwen2.5-0.5B-Instruct"
CONTEXTO = 512


def _series_por_canton(obt: pd.DataFrame) -> dict:
    s = obt.sort_values("fecha")
    return {c: (g["fecha"].values, g["n_emergencias"].to_numpy(dtype=np.float32))
            for c, g in s.groupby("cod_canton")}


def prob_exceder(cuantiles: np.ndarray, niveles: np.ndarray, umbral: np.ndarray) -> np.ndarray:
    """P(X > u) a partir de cuantiles (interpolación lineal de la CDF, colas acotadas)."""
    out = np.empty(len(umbral))
    for i, (q, u) in enumerate(zip(cuantiles, umbral)):
        q = np.maximum.accumulate(q)
        cdf = np.interp(u, q, niveles, left=0.0, right=1.0)
        out[i] = 1.0 - cdf
    return out


def chronos_scores(obt: pd.DataFrame, filas: pd.DataFrame, h: int, batch: int = 64,
                   dispositivo: str = "cpu") -> np.ndarray:
    """Puntaje P(pico) para cada fila (cod_canton, fecha=t, umbral_objetivo)."""
    import torch
    from chronos import BaseChronosPipeline

    pipe = BaseChronosPipeline.from_pretrained(CHRONOS_ID, device_map=dispositivo,
                                               torch_dtype=torch.float32)
    niveles = np.array(pipe.quantiles if hasattr(pipe, "quantiles")
                       else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    series = _series_por_canton(obt)
    contextos = []
    for c, t in zip(filas["cod_canton"].values, filas["fecha"].values):
        fechas, valores = series[c]
        fin = np.searchsorted(fechas, t, side="right")       # incluye t, excluye futuro
        contextos.append(torch.tensor(valores[max(0, fin - CONTEXTO):fin]))
    cuantiles = []
    with torch.no_grad():
        for i in range(0, len(contextos), batch):
            q, _ = pipe.predict_quantiles(contextos[i:i + batch], prediction_length=h,
                                          quantile_levels=list(niveles))
            cuantiles.append(q[:, h - 1, :].numpy())
    cuantiles = np.concatenate(cuantiles)
    return prob_exceder(cuantiles, niveles, filas["umbral_objetivo"].to_numpy())


# ---------------------------------------------------------------------------
# LLM por prompting
# ---------------------------------------------------------------------------
DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]


def prompt_fila(fila: pd.Series, historia: np.ndarray, h: int) -> str:
    d = pd.Timestamp(fila["fecha"]) + pd.Timedelta(days=h)
    return (
        "Eres analista del ECU 911 de Ecuador. Emergencias diarias del cantón "
        f"{fila['canton']} en los últimos 28 días (del más antiguo al más reciente, el último es hoy): "
        f"{', '.join(str(int(v)) for v in historia)}.\n"
        f"El percentil 90 histórico para un {DIAS[d.dayofweek]} en este cantón es "
        f"{fila['umbral_objetivo']:.1f} emergencias. "
        f"{'Ese día es feriado. ' if fila['es_feriado_obj'] == 1 else ''}"
        f"¿Dentro de {h} días ({DIAS[d.dayofweek]} {d.date()}) el número de emergencias "
        "superará estrictamente ese percentil 90? Responde sólo Sí o No."
    )


def llm_scores(obt: pd.DataFrame, filas: pd.DataFrame, h: int) -> np.ndarray:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(LLM_ID)
    modelo = AutoModelForCausalLM.from_pretrained(LLM_ID, torch_dtype=torch.float32)
    modelo.eval()
    id_si = tok.encode("Sí", add_special_tokens=False)[0]
    id_no = tok.encode("No", add_special_tokens=False)[0]
    series = _series_por_canton(obt)
    out = []
    for _, f in filas.iterrows():
        fechas, valores = series[f["cod_canton"]]
        fin = np.searchsorted(fechas, np.datetime64(f["fecha"], "ns"), side="right")
        msg = [{"role": "user", "content": prompt_fila(f, valores[fin - 28:fin], h)}]
        texto = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        ids = tok(texto, return_tensors="pt")
        with torch.no_grad():
            logits = modelo(**ids).logits[0, -1]
        p = torch.softmax(logits[[id_si, id_no]], dim=0)[0].item()
        out.append(p)
    return np.array(out)
