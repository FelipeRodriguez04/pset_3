"""Ejecutor idempotente de las fases del PSet 3."""

from __future__ import annotations

import argparse
import json

from src.experimentos import (
    cargar_contexto,
    ejecutar_baselines_y_candidatos,
    ejecutar_eda,
    ejecutar_seleccion,
    ejecutar_tuning,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=["eda", "candidates", "tuning", "selection", "all"],
        help="Fase que se desea ejecutar.",
    )
    parser.add_argument("--force", action="store_true", help="Ignora resultados cacheados.")
    args = parser.parse_args()

    df, data, info = cargar_contexto()
    print(f"OBT: {info.rows:,} filas, corte {info.snapshot_end}, fuente={info.source}")
    print(
        f"Modelado t+{data.horizon}: {len(data.train):,} train / {len(data.test):,} test; "
        f"positivos train={data.train[data.target_column].mean():.2%}"
    )

    if args.phase in ("eda", "all"):
        result = ejecutar_eda(df, data, info)
        print("EDA:", json.dumps(result["findings"], ensure_ascii=False, indent=2))
    if args.phase in ("candidates", "all"):
        result = ejecutar_baselines_y_candidatos(data, info, force=args.force)
        print(result[["modelo", "cv_pr_auc_mean", "cv_pr_auc_ponderado_mean", "cv_pr_auc_feriados_mean", "generalization_gap"]].to_string(index=False))
    if args.phase in ("tuning", "all"):
        result = ejecutar_tuning(data, info, force=args.force)
        print(result[["modelo", "best_cv_pr_auc_mean", "best_cv_pr_auc_ponderado_mean", "best_cv_pr_auc_feriados_mean", "improvement_vs_default"]].to_string(index=False))
    if args.phase in ("selection", "all"):
        result = ejecutar_seleccion(data, info, df, force=args.force)
        print("Seleccion:", json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
