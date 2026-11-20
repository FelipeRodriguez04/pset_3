"""Genera las figuras de resultados usadas en el memo del PSet 3."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PLOTS = DOCS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
COLOR = "#2a6f97"


def guardar(fig: plt.Figure, nombre: str) -> None:
    fig.tight_layout()
    fig.savefig(PLOTS / nombre, dpi=180, bbox_inches="tight")
    plt.close(fig)


def candidatos() -> None:
    df = pd.read_csv(DOCS / "candidatos.csv").sort_values("cv_pr_auc_ponderado_mean")
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.barh(df["modelo"].str.replace("_", " "), df["cv_pr_auc_ponderado_mean"], color=COLOR)
    ax.errorbar(
        df["cv_pr_auc_ponderado_mean"], range(len(df)), xerr=df["cv_pr_auc_ponderado_std"],
        fmt="none", ecolor="#ef8354", capsize=4, linewidth=1.6,
    )
    ax.set(title="Comparación con cuatro backtests semestrales", xlabel="PR-AUC ponderado (± desviación)", ylabel="")
    ax.set_xlim(0, max(df["cv_pr_auc_ponderado_mean"] + df["cv_pr_auc_ponderado_std"]) * 1.18)
    guardar(fig, "05_comparacion_candidatos.png")


def tuning() -> None:
    familias = {
        "hist_gradient_boosting": "HistGradientBoosting",
        "catboost": "CatBoost",
        "mlp": "MLP",
    }
    colores = ["#2a9d8f", "#e76f51", "#6c5ce7"]
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    for (archivo, etiqueta), color in zip(familias.items(), colores):
        df = pd.read_csv(DOCS / f"tuning_{archivo}.csv").sort_values("cv_pr_auc_mean", ascending=False).reset_index(drop=True)
        ax.plot(range(1, len(df) + 1), df["cv_pr_auc_mean"], marker="o", ms=3.2, lw=1.4, label=etiqueta, color=color)
    ax.set(title="Treinta configuraciones por familia", xlabel="Configuración ordenada dentro de cada familia", ylabel="PR-AUC temporal")
    ax.legend(frameon=True)
    guardar(fig, "06_tuning_90_configuraciones.png")


def finalistas() -> None:
    df = pd.read_csv(DOCS / "finalistas_test.csv")
    selection = json.loads((DOCS / "seleccion_resumen.json").read_text(encoding="utf-8"))
    baseline_pr_auc = selection["seasonal_baseline_test"]["pr_auc"]
    nombres = df["rol"].str.capitalize() + "\n" + df["modelo"].str.replace("_", " ")
    x = range(len(df))
    fig, axes = plt.subplots(1, 2, figsize=(8.7, 4.1))
    axes[0].bar(x, df["test_pr_auc"], color=["#2a9d8f", "#e9c46a"])
    axes[0].axhline(baseline_pr_auc, color="#555", ls="--", label="Regla estacional")
    axes[0].set_xticks(list(x), nombres)
    axes[0].set(ylabel="PR-AUC en holdout", title="Desempeño técnico")
    axes[0].legend(fontsize=8)
    axes[1].bar(x, 100 * df["business_missed_peak_reduction"], color=["#2a9d8f", "#e9c46a"])
    axes[1].axhline(10, color="#b23a48", ls="--", label="Meta: 10%")
    axes[1].set_xticks(list(x), nombres)
    axes[1].set(ylabel="Reducción de picos omitidos (%)", title="Impacto de negocio")
    axes[1].legend(fontsize=8)
    guardar(fig, "07_resultado_holdout.png")


def temporadas() -> None:
    df = pd.read_csv(DOCS / "pruebas_estacionales.csv")
    df = df[(df["alcance"] == "backtests_oof") & (df["segmento"].isin(["todos", "feriado", "fin_anio", "diciembre", "regular"]))]
    pivot = df.pivot(index="segmento", columns="modelo", values="pr_auc")
    order = [name for name in ["todos", "feriado", "fin_anio", "diciembre", "regular"] if name in pivot.index]
    pivot = pivot.loc[order]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    pivot.plot(kind="bar", ax=ax, color=["#2a9d8f", "#e9c46a"])
    ax.set(title="Pruebas especiales en backtests", xlabel="Segmento temporal", ylabel="PR-AUC")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Modelo")
    guardar(fig, "09_pruebas_estacionales.png")


if __name__ == "__main__":
    candidatos()
    tuning()
    finalistas()
    temporadas()
    print(f"Gráficos guardados en {PLOTS}")
