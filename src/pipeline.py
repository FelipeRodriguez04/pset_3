"""Toda preparación del modelo: features deterministas + ColumnTransformer + estimador.

Las transformaciones con estado se ajustan exclusivamente al fit de cada fold.
La allowlist impide incorporar obj_*, fecha, target o indicadores retrospectivos.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ['n_emergencias', 'lag_1d', 'lag_7d', 'lag_14d', 'lag_28d',
           'media_prev_7d', 'media_prev_28d', 'tendencia_7_28', 'cambio_semanal',
           'estacional_relativa', 'dow_sin', 'dow_cos', 'mes_sin', 'mes_cos',
           'feriado_objetivo', 'seasonal_avg']
CATEGORICAL = ['cod_canton', 'provincia', 'region', 'dow_objetivo']

def add_history(df):
    """Media de cuatro ocurrencias anteriores del weekday objetivo, conocidas al cierre t.

    t+3 usa t-4,-11,-18,-25. Nunca shift negativo.
    Se calcula antes de recortar series para conservar el contexto de los bordes.
    """
    df = df.sort_values(['cod_canton', 'fecha']).copy()
    g = df.groupby('cod_canton').n_emergencias
    for h in (3,):
        past = pd.concat([g.shift(7*k-h) for k in (1, 2, 3, 4)], axis=1)
        df[f'seasonal_t{h}'] = past.mean(axis=1).where(past.notna().all(axis=1))
    return df.sort_values(['fecha', 'cod_canton']).reset_index(drop=True)

def model_input(df, horizon=3):
    if horizon != 3:
        raise ValueError("Este proyecto solo admite t+3.")
    columns = ['fecha', 'cod_canton', 'provincia', 'region', 'n_emergencias',
               'lag_1d', 'lag_7d', 'lag_14d', 'lag_28d', 'media_prev_7d', 'media_prev_28d']
    x = df[columns].copy()
    x['fecha'] = x.fecha.dt.strftime('%Y-%m-%d')  # firma servible: fecha ISO, no Timestamp Python
    # Firma numérica flotante: permite nulos imputables también al servir el modelo.
    for column in columns:
        if column not in ('fecha', 'cod_canton', 'provincia', 'region'):
            x[column] = x[column].astype(float)
    x['seasonal_avg'] = df[f'seasonal_t{horizon}'].to_numpy()
    x['feriado_objetivo'] = df[f'es_feriado_t{horizon}'].astype(float).to_numpy()
    return x

class FeatureBuilder(TransformerMixin, BaseEstimator):
    def __init__(self, horizon=3, derived=True):
        self.horizon = horizon
        self.derived = derived

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.horizon != 3:
            raise ValueError("Este proyecto solo admite t+3.")
        x = X.copy()
        target_date = pd.to_datetime(x['fecha']) + pd.Timedelta(days=self.horizon)
        dow = target_date.dt.day_of_week
        month = target_date.dt.month
        x['dow_objetivo'] = dow.astype(str)
        x['dow_sin'], x['dow_cos'] = np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7)
        x['mes_sin'], x['mes_cos'] = np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12)
        x['tendencia_7_28'] = (x.media_prev_7d+1)/(x.media_prev_28d+1)
        x['cambio_semanal'] = (x.n_emergencias-x.lag_7d)/(x.lag_7d+1)
        x['estacional_relativa'] = (x.seasonal_avg+1)/(x.media_prev_28d+1)
        if not self.derived:
            x[['tendencia_7_28', 'cambio_semanal', 'estacional_relativa']] = 0.0
        return x[NUMERIC+CATEGORICAL].replace([np.inf, -np.inf], np.nan)

def make_pipeline(estimator, horizon=3, derived=True):
    if horizon != 3:
        raise ValueError("Este proyecto solo admite t+3.")
    numeric = Pipeline([('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
                        ('scale', StandardScaler())])
    categorical = Pipeline([('imputer', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
                            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preparation = ColumnTransformer([('num', numeric, NUMERIC), ('cat', categorical, CATEGORICAL)],
                                    remainder='drop')
    return Pipeline([('features', FeatureBuilder(horizon, derived)),
                     ('preparation', preparation), ('model', estimator)])

class SeasonalClassifier(ClassifierMixin, BaseEstimator):
    """Regla de negocio PSet1: media de cuatro weekdays > P90 cantón/weekday.

    predict_proba produce un score de ordenamiento, no una probabilidad calibrada.
    La decisión original se conserva en evaluate_business (score > 0.5).
    """
    def fit(self, X, y, **kwargs):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        ratio = X['seasonal_avg'].to_numpy() / np.maximum(X['p90'].to_numpy(), 1e-9)
        p = 1 / (1+np.exp(-np.clip(ratio-1, -30, 30)))
        return np.column_stack([1-p, p])

    def predict(self, X):
        return (X['seasonal_avg'].to_numpy() > X['p90'].to_numpy()).astype(int)
