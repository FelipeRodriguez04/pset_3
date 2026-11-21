import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import reporte


def test_salidas_van_a_docs_en_modo_real(monkeypatch, tmp_path):
    monkeypatch.delenv("ECU911_SINTETICO", raising=False)
    monkeypatch.setattr(reporte, "RAIZ", tmp_path)
    fig, _ = plt.subplots()
    p = reporte.guardar_fig(fig, "x")
    t = reporte.guardar_tabla(pd.DataFrame({"a": [1]}), "y")
    assert p == tmp_path / "docs" / "figuras" / "x.png" and p.exists()
    assert t == tmp_path / "docs" / "resultados" / "y.csv" and t.exists()


def test_salidas_no_tocan_docs_en_modo_sintetico(monkeypatch, tmp_path):
    monkeypatch.setenv("ECU911_SINTETICO", "1")
    monkeypatch.setattr(reporte, "RAIZ", tmp_path)
    fig, _ = plt.subplots()
    p = reporte.guardar_fig(fig, "x")
    assert "docs" not in p.parts and p.exists()
