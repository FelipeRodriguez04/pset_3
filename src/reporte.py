"""Guardado de figuras y tablas de los notebooks (para el memo).
En modo sintético (desarrollo) nada se escribe en docs/."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.datos import RAIZ


def _destino(sub: str) -> Path:
    base = RAIZ / "data" / "dev" if os.getenv("ECU911_SINTETICO") == "1" else RAIZ / "docs"
    d = base / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def guardar_fig(fig, nombre: str) -> Path:
    p = _destino("figuras") / f"{nombre}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    return p


def guardar_tabla(df: pd.DataFrame, nombre: str) -> Path:
    p = _destino("resultados") / f"{nombre}.csv"
    df.to_csv(p, index=False)
    return p
