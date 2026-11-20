from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.datos import agregar_features, cortes_temporales, preparar_modelado
from src.pipeline import SeasonalRuleClassifier, construir_pipeline


def synthetic_obt(days: int = 500, cantons: int = 3) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    rows = []
    for canton in range(cantons):
        values = 10 + canton * 4 + np.arange(days) % 7
        values = values.astype(float)
        values[np.arange(days) % 17 == 0] += 30
        for index, date in enumerate(dates):
            is_holiday = date == pd.Timestamp("2025-02-14")
            target_date_t3 = date + pd.Timedelta(days=3)
            rows.append(
                {
                    "cod_canton": f"{canton:04d}",
                    "fecha": date,
                    "canton": f"C{canton}",
                    "cod_provincia": "01",
                    "provincia": "P1",
                    "region": "Sierra",
                    "es_zona_no_delimitada": False,
                    "anio": date.year,
                    "trimestre": date.quarter,
                    "mes": date.month,
                    "periodo": date.strftime("%Y-%m"),
                    "dia_mes": date.day,
                    "dia_semana": date.isoweekday(),
                    "nombre_dia": date.day_name(),
                    "es_fin_de_semana": date.weekday() >= 5,
                    "es_feriado": is_holiday,
                    "nombre_feriado": "Feriado de prueba" if is_holiday else None,
                    "es_periodo_anomalo": False,
                    "n_emergencias": values[index],
                    "n_seguridad_ciudadana": values[index] // 2,
                    "n_gestion_sanitaria": values[index] // 3,
                    "n_transito_movilidad": 0,
                    "n_servicios_municipales": 0,
                    "n_gestion_siniestros": 0,
                    "n_servicio_militar": 0,
                    "n_gestion_riesgos": 0,
                    "n_sin_servicio": 0,
                    "lag_1d": values[index - 1] if index >= 1 else np.nan,
                    "lag_7d": values[index - 7] if index >= 7 else np.nan,
                    "lag_14d": values[index - 14] if index >= 14 else np.nan,
                    "lag_28d": values[index - 28] if index >= 28 else np.nan,
                    "media_prev_7d": np.mean(values[max(0, index - 7) : index]) if index else np.nan,
                    "media_prev_28d": np.mean(values[max(0, index - 28) : index]) if index else np.nan,
                    "es_feriado_t3": target_date_t3 == pd.Timestamp("2025-02-14"),
                    "es_feriado_t7": False,
                    "obj_n_emergencias_t3": values[index + 3] if index + 3 < days else np.nan,
                    "obj_n_emergencias_t7": values[index + 7] if index + 7 < days else np.nan,
                }
            )
    return pd.DataFrame(rows)


class PSet3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = synthetic_obt()
        # El corte posterior al primer año también ejercita los rezagos 364/365/371.
        cls.data = preparar_modelado(cls.df, horizon=3, holdout_start="2026-03-01")

    def test_features_use_only_past(self):
        engineered = agregar_features(self.df, horizon=3)
        one = engineered.loc[engineered["cod_canton"] == "0000"].reset_index(drop=True)
        self.assertEqual(one.loc[30, "lag_21d"], one.loc[9, "n_emergencias"])
        expected = one.loc[2:29, "n_emergencias"].std(ddof=1)
        self.assertAlmostEqual(one.loc[30, "volatilidad_prev_28d"], expected)

    def test_future_columns_are_not_predictors(self):
        forbidden = {"obj_n_emergencias_t3", "obj_n_emergencias_t7", "fecha_objetivo"}
        self.assertFalse(forbidden.intersection(self.data.feature_columns))

    def test_temporal_splits_do_not_overlap(self):
        for train_idx, validation_idx in cortes_temporales(self.data.train["fecha"], n_splits=2, horizon=3):
            train_max = self.data.train.iloc[train_idx]["fecha"].max()
            validation_min = self.data.train.iloc[validation_idx]["fecha"].min()
            self.assertGreaterEqual((validation_min - train_max).days, 4)

    def test_holdout_boundary_is_purged(self):
        self.assertLess(
            self.data.train["fecha_objetivo"].max(),
            pd.Timestamp(self.data.holdout_start),
        )
        self.assertGreaterEqual(self.data.test["fecha"].min(), pd.Timestamp(self.data.holdout_start))

    def test_known_holiday_calendar_is_available(self):
        row = self.data.frame.loc[
            (self.data.frame["cod_canton"] == "0000")
            & (self.data.frame["fecha_objetivo"] == pd.Timestamp("2025-02-14"))
        ].iloc[0]
        self.assertTrue(bool(row["es_feriado_objetivo"]))
        self.assertEqual(row["nombre_feriado_objetivo"], "Feriado de prueba")

    def test_pipeline_fits_without_missing_leakage(self):
        pipeline = construir_pipeline("logistica", self.data.numeric_columns, self.data.categorical_columns)
        X = self.data.train[self.data.feature_columns]
        y = self.data.train[self.data.target_column]
        pipeline.fit(X, y)
        probabilities = pipeline.predict_proba(X.head(10))[:, 1]
        self.assertTrue(np.isfinite(probabilities).all())

    def test_business_rule_boundary(self):
        model = SeasonalRuleClassifier().fit(pd.DataFrame({"score_regla_negocio": [0.5]}), [0])
        predictions = model.predict(pd.DataFrame({"score_regla_negocio": [0.9, 1.0, 1.2]}))
        self.assertEqual(predictions.tolist(), [0, 1, 1])


if __name__ == "__main__":
    unittest.main()
