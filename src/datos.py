"""Datos, EDA, experimentación, tracking y visualizaciones del PSet 3.

La preparación de features y estimadores se importa de src/pipeline.py.
Ejecutar fases con python -m src.datos <fase>.
"""

# CONFIG
from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env', override=False)
DATA = Path(os.getenv('PSET3_DATA_DIR', str(Path.home() / '.cache' / 'pset_3')))
DOCS = Path(os.getenv('PSET3_DOCS_DIR', str(ROOT / 'docs')))
EVIDENCE = DATA / 'evidencia_t3'
HORIZONS = (3,)
LEGACY_LLM_PROMPT_SHA256 = 'e6dc49ff14457fc5e7a59b563fa77e1ec554b46330274f9554de19c035e9c4f0'
LEGACY_T3_EXPERIMENT_HASH = '2c9b2b24d0968007e424d987f8ec9778daa82bdb7cc92bf90d9bb94b1a282f18'
LEGACY_T3_COMPATIBLE_CONTRACT = '38e30b475be00c4046e5ac01e66a3fcea47da0963f2de18a47b00b873b50f8e5'
SEED = 42
PROJECT = os.getenv('MLFLOW_EXPERIMENT_NAME', 'ecu911-alta-demanda')
TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5001')
SNAPSHOT_END = os.getenv('SNAPSHOT_END', '2026-09-30')
COST_FN = float(os.getenv('COST_FN', '5'))
COST_FP = float(os.getenv('COST_FP', '1'))
MIN_PRECISION = float(os.getenv('MIN_PRECISION', '0.30'))
LLM_BASE_URL = os.getenv('LLM_BASE_URL', 'http://127.0.0.1:11434/v1').rstrip('/')
LLM_MODEL = os.getenv('LLM_MODEL', 'qwen2.5:1.5b')


# DATOS
import hashlib
import json
import os
import re
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

TABLE_DEFAULT = 'ECU911.OBT.OBT_EMERGENCIAS_CANTON_DIA'

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def normalize(df):
    df = df.copy()
    df.columns = df.columns.str.lower()
    if df[['fecha', 'cod_canton']].isna().any().any():
        raise ValueError('Claves fecha/cantón nulas en origen.')
    df['fecha'] = pd.to_datetime(df['fecha']).dt.normalize()
    df['cod_canton'] = df['cod_canton'].astype(str).str.zfill(4)
    return df.sort_values(['fecha', 'cod_canton']).reset_index(drop=True)

def validate(df):
    required = {'fecha', 'cod_canton', 'provincia', 'region', 'n_emergencias',
                'lag_1d', 'lag_7d', 'lag_14d', 'lag_28d', 'media_prev_7d',
                'media_prev_28d', 'es_feriado_t3', 'obj_n_emergencias_t3'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Faltan columnas OBT: {sorted(missing)}')
    if df[['fecha', 'cod_canton']].isna().any().any() or df.duplicated(['fecha', 'cod_canton']).any():
        raise ValueError('Grain inválido: fecha/cantón nulos o duplicados.')
    if df['n_emergencias'].isna().any() or (df['n_emergencias'] < 0).any():
        raise ValueError('Conteos nulos/negativos.')
    if df[['es_feriado_t3']].isna().any().any():
        raise ValueError('Calendario futuro incompleto: ampliar dim_fecha/feriados en PSet 2.')
    ordered = df.sort_values(['cod_canton', 'fecha'])
    if not ordered.groupby('cod_canton')['fecha'].diff().dropna().eq(pd.Timedelta(days=1)).all():
        raise ValueError('Huecos en series: no se pueden interpretar lags por filas.')
    if not ordered.groupby('cod_canton').size().eq(df['fecha'].nunique()).all():
        raise ValueError('Cantones con cobertura desigual.')
    grouped = ordered.groupby('cod_canton')['n_emergencias']
    for k in (1, 7, 14, 28):
        expected = grouped.shift(k)
        if not np.allclose(expected, ordered[f'lag_{k}d'], equal_nan=True):
            raise ValueError(f'lag_{k}d no coincide con la serie histórica.')
    for k in (7, 28):
        expected = grouped.transform(lambda s: s.shift(1).rolling(k, min_periods=k).mean())
        if not np.allclose(expected, ordered[f'media_prev_{k}d'], atol=0.00011, equal_nan=True):
            raise ValueError(f'media_prev_{k}d inconsistente.')
    for h in HORIZONS:
        if not np.allclose(grouped.shift(-h), ordered[f'obj_n_emergencias_t{h}'], equal_nan=True):
            raise ValueError(f'Objetivo t+{h} inconsistente.')
    return {'rows': len(df), 'columns': len(df.columns), 'cantones': df.cod_canton.nunique(),
            'date_min': str(df.fecha.min().date()), 'date_max': str(df.fecha.max().date())}

def snapshot_from_snowflake():
    """Una consulta atómica al histórico; nunca ejecuta SQL de escritura."""
    import snowflake.connector
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / 'obt.parquet'
    if path.exists():
        raise FileExistsError('El snapshot ya existe. No se sobrescribe: usar otra PSET3_DATA_DIR.')
    parts = [os.getenv('SNOWFLAKE_DATABASE', 'ECU911'), os.getenv('SNOWFLAKE_SCHEMA', 'OBT'),
             os.getenv('SNOWFLAKE_TABLE', 'OBT_EMERGENCIAS_CANTON_DIA')]
    if not all(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', p) for p in parts):
        raise ValueError('Identificadores Snowflake inválidos.')
    table = '.'.join(parts)
    kwargs = {k: os.getenv('SNOWFLAKE_' + k.upper()) for k in
              ('account', 'user', 'password', 'role', 'warehouse')}
    kwargs = {k: v for k, v in kwargs.items() if v}
    kwargs['authenticator'] = os.getenv('SNOWFLAKE_AUTHENTICATOR', 'snowflake')
    if not kwargs.get('account') or not kwargs.get('user'):
        raise RuntimeError('Completar conexión Snowflake en .env; no hay datos reales locales.')
    with snowflake.connector.connect(**kwargs, login_timeout=20, network_timeout=60) as conn:
        with conn.cursor() as cur:
            cur.execute(f'SELECT * FROM {table} WHERE FECHA <= TO_DATE(%s) ORDER BY FECHA, COD_CANTON',
                        (SNAPSHOT_END,))
            query_id = cur.sfqid
            df = normalize(cur.fetch_pandas_all())
    # Los leads del último día se recortan al corte, aunque la tabla haya sido ampliada.
    for h in HORIZONS:
        df.loc[df.fecha + pd.Timedelta(days=h) > pd.Timestamp(SNAPSHOT_END),
               f'obj_n_emergencias_t{h}'] = np.nan
    info = validate(df)
    df.to_parquet(path, index=False)
    info.update(source_table=table, snapshot_end=SNAPSHOT_END, query_id=query_id,
                extracted_at=datetime.now(timezone.utc).isoformat(), sha256=sha256(path), synthetic=False)
    (DATA / 'snapshot.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    return info

def load_snapshot():
    path = DATA / 'obt.parquet'
    if not path.exists():
        raise FileNotFoundError('OBT pendiente: ejecutar python -m src.run snapshot con Snowflake configurado.')
    info = json.loads((DATA / 'snapshot.json').read_text())
    if sha256(path) != info['sha256']:
        raise ValueError('El snapshot cambió: hash SHA256 distinto.')
    df = normalize(pd.read_parquet(path))
    validate(df)
    return df, info

def partition(df, calibration_days=365, test_fraction=0.20, n_splits=3):
    """P90 en año inicial, train/CV posterior y hold-out temporal con embargo de 7 días.

    El año inicial es parte del bloque de entrenamiento y se reserva para definir la
    etiqueta, antes de ver cualquier fold. Se congela la definición para comparar
    exactamente el mismo problema entre familias/pliegues para t+3.
    """
    dates = np.sort(df.fecha.unique())
    calibration_end = pd.Timestamp(dates[0]) + pd.Timedelta(days=calibration_days - 1)
    cutoff = pd.Timestamp(dates[int(len(dates) * (1-test_fraction))])
    if cutoff <= calibration_end + pd.Timedelta(days=90):
        raise ValueError('Historia insuficiente después del año de calibración.')
    manifest = {'calibration_end': str(calibration_end.date()), 'test_start': str(cutoff.date()),
                'embargo_days': 7, 'n_splits': n_splits, 'test_fraction': test_fraction}
    path = DATA / 'split.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Split ya fijado con otra configuración. No reutilizar el hold-out.')
    path.write_text(json.dumps(manifest, indent=2))
    calibration = df[df.fecha <= calibration_end].copy()
    train = df[(df.fecha > calibration_end) & (df.fecha + pd.Timedelta(days=7) < cutoff)].copy()
    test = df[(df.fecha >= cutoff) & df.obj_n_emergencias_t3.notna()].copy()
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    # TimeSeriesSplit sobre DÍAS, no sobre las filas intercaladas de 224 cantones.
    days = np.sort(train.fecha.unique())
    folds = []
    for a, b in TimeSeriesSplit(n_splits=n_splits, gap=7).split(days):
        i = np.flatnonzero(train.fecha.isin(days[a]))
        j = np.flatnonzero(train.fecha.isin(days[b]))
        assert train.iloc[i].fecha.max() + pd.Timedelta(days=7) < train.iloc[j].fecha.min()
        folds.append((i, j))
    return calibration, train, test, folds, manifest

def thresholds(calibration, quantile=0.90):
    tmp = calibration.assign(target_dow=calibration.fecha.dt.day_of_week)
    return tmp.groupby(['cod_canton', 'target_dow']).n_emergencias.quantile(quantile)

def label(df, threshold, horizon=3):
    if horizon not in HORIZONS:
        raise ValueError('Este experimento está limitado a t+3.')
    dow = (df.fecha + pd.Timedelta(days=horizon)).dt.day_of_week
    idx = pd.MultiIndex.from_arrays([df.cod_canton, dow], names=threshold.index.names)
    q = threshold.reindex(idx).to_numpy()
    actual = df[f'obj_n_emergencias_t{horizon}'].to_numpy(dtype=float)
    if np.isnan(q).any() or np.isnan(actual).any():
        raise ValueError('Target no observado o cantón sin P90 de calibración. No se imputa.')
    return (actual > q).astype(int), q

# METRICAS
import numpy as np
from sklearn.metrics import average_precision_score, precision_score, recall_score, confusion_matrix

def score(y, prob, threshold=0.5, prediction=None):
    y = np.asarray(y)
    pred = np.asarray(prob >= threshold if prediction is None else prediction, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'ap': float(average_precision_score(y, prob)),
            'precision': float(precision_score(y, pred, zero_division=0)),
            'recall': float(recall_score(y, pred, zero_division=0)),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'cost_per_1000': float(1000*(COST_FN*fn+COST_FP*fp)/len(y)),
            'alert_rate': float(pred.mean())}

def choose_threshold(y, prob, min_precision=MIN_PRECISION):
    """Minimizar 5FN+FP en predicciones OOF, sujeto a precision >= 0.30.

    Si la restricción no es factible, se declara y se minimiza costo sin ella.
    Se consideran todos los puntos de cambio observados (no solo grilla gruesa).
    """
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y, prob)
    positive = np.sum(y)
    tp = recall[:-1] * positive
    fp = np.divide(tp, precision[:-1], out=np.zeros_like(tp), where=precision[:-1]>0)-tp
    costs = COST_FN*(positive-tp)+COST_FP*fp
    # Candidato sin alertas: precision 0, costo FN total. Solo fallback.
    thresholds = np.append(thresholds, 1.000001)
    costs = np.append(costs, COST_FN*positive)
    feasible = np.append(precision[:-1] >= min_precision, False)
    eligible = np.flatnonzero(feasible) if feasible.any() else np.arange(len(costs))
    k = eligible[np.argmin(costs[eligible])]
    return float(thresholds[k]), bool(feasible.any())

def fn_reduction(model_fn, baseline_fn):
    return None if baseline_fn == 0 else float((baseline_fn-model_fn)/baseline_fn)

# EDA
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def run_eda(calibration, train, threshold):
    out = EVIDENCE / 'eda'
    out.mkdir(parents=True, exist_ok=True)
    schema = pd.DataFrame({'dtype': train.dtypes.astype(str), 'nulls': train.isna().sum(),
                           'missing_fraction': train.isna().mean()})
    schema.to_csv(out/'schema.csv')
    train.describe(include='all').to_csv(out/'descriptive.csv')
    numeric = train.select_dtypes(include=[np.number])
    q1, q3 = numeric.quantile(.25), numeric.quantile(.75)
    iqr = q3-q1
    flagged_iqr = ((numeric < q1-1.5*iqr) | (numeric > q3+1.5*iqr)).sum()
    z = (numeric-numeric.mean())/numeric.std(ddof=0).replace(0, np.nan)
    pd.DataFrame({'iqr_count': flagged_iqr, 'z_ge_3_count': z.abs().ge(3).sum()}).to_csv(out/'outliers.csv')
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
    ax[0].hist(np.log1p(train.n_emergencias), bins=55, color='#167d9a')
    ax[0].set(xlabel='log(1 + emergencias en t)', ylabel='Cantón-días')
    train[['n_emergencias','lag_7d','media_prev_28d']].plot.box(ax=ax[1], logy=True)
    fig.tight_layout(); fig.savefig(out/'distribucion_boxplot.png', dpi=160); plt.close(fig)
    findings = []
    zero = float(train.n_emergencias.eq(0).mean())
    skew = float(train.n_emergencias.skew())
    findings.append({'hallazgo':'Escalas cantonales y cola del volumen',
                     'evidencia':f'asimetría={skew:.3f}; ceros={zero:.2%}; IQR flags={int(flagged_iqr.n_emergencias)}',
                     'consecuencia':'Usar Spearman y ratios de tendencia/cambio; no borrar picos reales como outliers.'})
    per_canton = train.groupby('cod_canton').n_emergencias.mean()
    per_canton.to_csv(out/'volumen_canton.csv')
    findings.append({'hallazgo':'Heterogeneidad cantonal',
                     'evidencia':f'media cantonal P10={per_canton.quantile(.1):.3f}, P90={per_canton.quantile(.9):.3f}',
                     'consecuencia':'Escalado dentro de fold, features relativas y evaluación por estrato de volumen.'})
    correlation_columns = ['n_emergencias','lag_1d','lag_7d','lag_14d','lag_28d',
                           'media_prev_7d','media_prev_28d','seasonal_t3']
    corr = train[correlation_columns].copy()
    prevalence = {}
    for h in HORIZONS:
        y, _ = label(train, threshold, h)
        prevalence[str(h)] = float(y.mean())
        corr[f'y_t{h}'] = y
    corr = corr.corr(method='spearman')
    corr.to_csv(out/'spearman.csv')
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap='coolwarm')
    ax.set_xticks(range(len(corr)), corr.columns, rotation=70, ha='right', fontsize=7)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=7)
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(out/'correlacion.png', dpi=160); plt.close(fig)
    findings.append({'hallazgo':'Distribución real de alta demanda',
                     'evidencia':f'prevalencia train t+3={prevalence["3"]:.2%}',
                     'consecuencia':'Comparar AP (PR-AUC no interpolada); calibrar umbral en OOF, no en test.'})
    seasonal = train.assign(dow=train.fecha.dt.day_of_week).groupby('dow').n_emergencias.mean()
    findings.append({'hallazgo':'Patrón semanal observado',
                     'evidencia':f'medias por weekday: {seasonal.round(2).to_dict()}',
                     'consecuencia':'Incluir sin/cos del día objetivo y media de cuatro ocurrencias; medir aporte con ablación.'})
    result = {'train_rows':len(train), 'train_columns':len(train.columns),
              'calibration_rows':len(calibration), 'prevalence':prevalence,
              'findings':findings, 'outliers_removed':0,
              'correlation_reason':'Spearman: conteos asimétricos, picos y relaciones monótonas; asociación no implica causalidad.'}
    (out/'findings.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    pd.DataFrame(findings).to_csv(out/'hallazgos.csv', index=False)
    return result

# EXPERIMENTOS
import hashlib
import json
import subprocess
import time
import warnings
import ast
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterSampler
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from .pipeline import add_history, model_input, make_pipeline, SeasonalClassifier

def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str))

def git_info():
    def git(*args):
        p = subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else 'UNCOMMITTED'
    source = hashlib.sha256()
    for path in sorted((ROOT/'src').glob('*.py')):
        source.update(path.name.encode()); source.update(path.read_bytes())
    return git('rev-parse','HEAD'), bool(git('status','--porcelain')), source.hexdigest()

def experiment_hash():
    """Cambios de presentación no invalidan experimentos científicos ya terminados."""
    ignored={'read_json','plot_candidates','plot_tuning','plot_selection','execute_notebooks','capture_mlflow','execute','main'}
    h=hashlib.sha256()
    for path in sorted((ROOT/'src').glob('*.py')):
        tree=ast.parse(path.read_text())
        tree.body=[node for node in tree.body if not (
            isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in ignored)]
        h.update(ast.dump(tree,include_attributes=False).encode())
    return h.hexdigest()

def training_contract_hash():
    """Huella de las partes que determinan entrenamiento/etiquetas y compatibilidad t+3."""
    names={'normalize','validate','load_snapshot','partition','thresholds','label','score',
           'choose_threshold','fn_reduction','context','models','spaces','check_train_labels',
           'evaluate','log_config','select_finalists','add_history','model_input','FeatureBuilder',
           'make_pipeline','SeasonalClassifier','HORIZONS','NUMERIC','CATEGORICAL'}
    digest=hashlib.sha256()
    for path in sorted((ROOT/'src').glob('*.py')):
        nodes=[]
        for node in ast.parse(path.read_text()).body:
            if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names:
                nodes.append(node)
            elif isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in node.targets):
                nodes.append(node)
        digest.update(ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False).encode())
    return digest.hexdigest()

def legacy_t3_compatible():
    """Compatibilidad acotada a esta transición; futuros cambios científicos la invalidan."""
    return training_contract_hash()==LEGACY_T3_COMPATIBLE_CONTRACT

def tuning_parent(c,h,name):
    """Reabre el padre de t+3 al retomar para mantener juntas sus 30 configuraciones."""
    path=DATA/'tuning_padres_t3.json'
    saved=json.loads(path.read_text()) if path.exists() else {}
    if saved and (saved.get('context_id')!=c['id'] or saved.get('experiment_sha256')!=experiment_hash()):
        raise ValueError('Los padres guardados pertenecen a otro protocolo.')
    key=f'{h}:{name}'
    if key in saved.get('parents',{}):
        return saved['parents'][key]
    parent_id=None
    legacy_path=DATA/'tuning_plan_paralelo.json'
    if legacy_path.exists() and legacy_t3_compatible():
        plan=json.loads(legacy_path.read_text())
        if plan.get('experiment_hash')==LEGACY_T3_EXPERIMENT_HASH:
            item=next((g for g in plan.get('groups',[]) if g['h']==h and g['name']==name),None)
            if item is not None:
                parent_id=item['parent_id']
    if parent_id is None:
        experiment=mlflow.get_experiment_by_name(PROJECT)
        parent=mlflow.MlflowClient().create_run(experiment.experiment_id,tags={
            'mlflow.runName':f'tuning-padre-t{h}-{name}','fase':'tuning_padre',
            'familia':name,'context_id':c['id'],'horizonte':str(h)})
        parent_id=parent.info.run_id
    saved.setdefault('parents',{})[key]=parent_id
    saved.update(context_id=c['id'],experiment_sha256=experiment_hash())
    dump(path,saved)
    return parent_id

def context():
    df, info = load_snapshot()
    # La OBT de origen puede incluir columnas de otros horizontes; no entran al experimento.
    model_df=df.drop(columns=['obj_n_emergencias_t7','es_feriado_t7'],errors='ignore')
    cal, train, test, folds, split = partition(add_history(model_df))
    DATA.mkdir(parents=True, exist_ok=True); DOCS.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    protocol={'split':split,'project':PROJECT,'seed':SEED,'cost_fn':COST_FN,
              'cost_fp':COST_FP,'min_precision':MIN_PRECISION}
    context_id = hashlib.sha256((info['sha256']+json.dumps(protocol,sort_keys=True)).encode()).hexdigest()[:16]
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(PROJECT)
    return {'horizons':HORIZONS, 'cal':cal, 'train':train, 'test':test, 'folds':folds,
            'p90':thresholds(cal), 'info':info, 'split':split, 'id':context_id}

def models():
    # Solo se fija semilla. Todos los hiperparámetros restantes son los defaults de sklearn.
    return {'logistica':LogisticRegression(random_state=SEED),
            'random_forest':RandomForestClassifier(random_state=SEED),
            'hist_boosting':HistGradientBoostingClassifier(random_state=SEED),
            'mlp':MLPClassifier(random_state=SEED),
            'gaussian_nb':GaussianNB()}

def spaces(name):
    # Al menos 30 configuraciones distintas: distribuciones continuas + ParameterSampler.
    return {
      'logistica':{'C':loguniform(1e-4,1e3), 'class_weight':[None,'balanced'], 'max_iter':[500,1000]},
      'random_forest':{'n_estimators':randint(80,401), 'max_depth':[None,6,12,20],
                       'min_samples_leaf':randint(1,31), 'max_features':['sqrt',.3,.7],
                       'class_weight':[None,'balanced']},
      'hist_boosting':{'learning_rate':loguniform(.01,.3), 'max_iter':randint(80,301),
                       'max_leaf_nodes':randint(7,64), 'l2_regularization':loguniform(1e-4,10),
                       'min_samples_leaf':randint(10,61)},
      'mlp':{'hidden_layer_sizes':[(32,),(64,),(100,),(64,32)],
             'alpha':loguniform(1e-6,.1), 'learning_rate_init':loguniform(1e-4,1e-2),
             'max_iter':[200,400], 'early_stopping':[True,False], 'shuffle':[False]},
      'gaussian_nb':{'var_smoothing':loguniform(1e-12,1e-3)}
    }[name]

def check_train_labels(y):
    if len(np.unique(y)) != 2:
        raise ValueError('Un fold no contiene ambas clases. Revisar cobertura/partición antes de experimentar.')

def evaluate(estimator, x, y, folds, rule=False):
    out = np.full(len(y), np.nan)
    fold_rows, caught = [], []
    start = time.perf_counter()
    fit_total = 0.0
    for k,(a,b) in enumerate(folds):
        check_train_labels(y[a]); check_train_labels(y[b])
        model = clone(estimator)
        tick = time.perf_counter()
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit(x.iloc[a], y[a])
        fold_fit = time.perf_counter()-tick
        fit_total += fold_fit
        caught += [str(w.message) for w in ws]
        train_score = score(y[a], model.predict_proba(x.iloc[a])[:,1])
        p = model.predict_proba(x.iloc[b])[:,1]
        out[b] = p
        valid_score = score(y[b], p, prediction=model.predict(x.iloc[b]) if rule else None)
        fold_rows.append({'fold':k, 'train_ap':train_score['ap'], 'cv_ap':valid_score['ap'],
                          'seconds':fold_fit, 'train_rows':len(a), 'valid_rows':len(b),
                          'train_end':str(x.iloc[a].fecha.max()), 'valid_start':str(x.iloc[b].fecha.min())})
    tick = time.perf_counter()
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always', ConvergenceWarning)
        estimator.fit(x, y)
    refit_seconds = time.perf_counter()-tick
    fit_total += refit_seconds
    caught += [str(w.message) for w in ws]
    train_ap = score(y, estimator.predict_proba(x)[:,1])['ap']
    aps = [row['cv_ap'] for row in fold_rows]
    return {'train_ap':train_ap, 'cv_ap_mean':float(np.mean(aps)),
            'cv_ap_std':float(np.std(aps, ddof=1)), 'gap':float(train_ap-np.mean(aps)),
            'fit_seconds':fit_total, 'refit_seconds':refit_seconds,
            'elapsed_seconds':time.perf_counter()-start}, out, fold_rows, caught

def log_config(c, h, name, estimator, phase, params=None, nested=False, derived=True):
    if h not in HORIZONS:
        raise ValueError('Solo se pueden entrenar configuraciones t+3.')
    key=hashlib.sha256(json.dumps({'context_id':c['id'],'h':h,'name':name,'phase':phase,
        'params':params or {},'derived':derived,'protocol':experiment_hash()},sort_keys=True,default=str).encode()).hexdigest()
    checkpoint=DATA/'configuraciones'/f'{key}.json'
    if checkpoint.exists():
        return json.loads(checkpoint.read_text())
    if legacy_t3_compatible():
        old_key=hashlib.sha256(json.dumps({'context_id':c['id'],'h':h,'name':name,'phase':phase,
            'params':params or {},'derived':derived,'protocol':LEGACY_T3_EXPERIMENT_HASH},
            sort_keys=True,default=str).encode()).hexdigest()
        old_checkpoint=DATA/'configuraciones'/f'{old_key}.json'
        if old_checkpoint.exists():
            return json.loads(old_checkpoint.read_text())
    print(f'INICIO {phase} t+{h} {name} {params or {}}',flush=True)
    x = model_input(c['train'], h)
    y,q = label(c['train'], c['p90'], h)
    rule = name == 'estacional'
    if rule:
        x['p90'] = q
        pipe = SeasonalClassifier()
    else:
        pipe = make_pipeline(estimator, h, derived=derived)
    with mlflow.start_run(run_name=f'{phase}-t{h}-{name}', nested=nested) as run:
        commit, dirty, source_hash = git_info()
        mlflow.set_tags({'fase':phase, 'familia':name, 'horizonte':str(h), 'git_commit':commit,
                         'git_dirty':str(dirty), 'source_sha256':source_hash,
                         'context_id':c['id'], 'synthetic':str(c['info'].get('synthetic',False)).lower()})
        mlflow.log_params({'seed':SEED, 'source_table':c['info']['source_table'],
                           'snapshot_end':c['info']['snapshot_end'], 'dataset_sha256':c['info']['sha256'],
                           'horizon':h, 'derived_features':derived, 'cv_splits':len(c['folds']),
                           'cost_fn':COST_FN, 'cost_fp':COST_FP, 'min_precision':MIN_PRECISION,
                           **{f'model__{k}':str(v) for k,v in (estimator.get_params() if estimator is not None else {}).items()}})
        mlflow.log_dict(c['split'], 'split.json')
        mlflow.log_dict(c['info'], 'snapshot.json')
        t, oof, fold_rows, ws = evaluate(pipe, x, y, c['folds'], rule)
        valid = np.isfinite(oof)
        threshold, feasible = choose_threshold(y[valid], oof[valid]) if not rule else (.5, True)
        oof_metrics = score(y[valid], oof[valid], threshold,
                            prediction=(oof[valid] > .5).astype(int) if rule else None)
        mlflow.log_metrics({**t, 'threshold':threshold,
                            **{f'oof_{k}':v for k,v in oof_metrics.items()}})
        mlflow.log_dict({'folds':fold_rows}, 'folds.json')
        mlflow.log_dict({'warnings':ws, 'precision_constraint_feasible':feasible}, 'diagnostico.json')
        # predict_proba en pyfunc; sklearn.load_model recupera el pipeline completo.
        example = x.head(3)
        model_info = mlflow.sklearn.log_model(pipe, name='pipeline', serialization_format='cloudpickle',
            signature=infer_signature(example, pipe.predict_proba(example)), input_example=example,
            pyfunc_predict_fn='predict_proba', code_paths=[str(ROOT/'src')],
            pip_requirements=[f'{p}=={__import__("importlib.metadata",fromlist=["version"]).version(p)}'
                              for p in ('scikit-learn','pandas','numpy','scipy','mlflow','python-dotenv')])
        run_id = run.info.run_id
        joblib.dump(pipe, DATA/f'{run_id}.joblib', compress=3)
        np.savez_compressed(DATA/f'{run_id}_oof.npz', y=y, p=oof)
        result = {'name':name, 'horizon':h, 'phase':phase, 'run_id':run_id,
                  'model_uri':model_info.model_uri, **t, 'threshold':threshold,
                  'precision_constraint_feasible':feasible, 'params':params or {},
                  'learning_rate':estimator.get_params().get('learning_rate_init',
                                    estimator.get_params().get('learning_rate','NA')) if estimator is not None else 'NA',
                  'warnings':len(ws), 'fold_ap':[r['cv_ap'] for r in fold_rows]}
        dump(checkpoint,result)
        print(f'FIN {phase} t+{h} {name}: AP_CV={t["cv_ap_mean"]:.5f} ± {t["cv_ap_std"]:.5f}; ajuste={t["fit_seconds"]:.1f}s',flush=True)
        return result

def stage_path(stage):
    return DATA/f'{stage}.json'

def read_stage(stage, c):
    path = stage_path(stage)
    if not path.exists():
        return None
    state = json.loads(path.read_text())
    if state['context_id'] != c['id']:
        raise ValueError('El estado pertenece a otro snapshot/split/proyecto.')
    current=state.get('experiment_sha256')==experiment_hash()
    compatible=(stage in {'candidatos','candidatos_parcial','ablacion','tuning'} and
                state.get('experiment_sha256')==LEGACY_T3_EXPERIMENT_HASH and legacy_t3_compatible())
    if not (current or compatible):
        raise ValueError('El protocolo cambió: no se pueden reutilizar estas corridas automáticamente.')
    rows=[r for r in state['results'] if r.get('horizon')==3]
    return rows or None

def save_stage(stage, c, rows):
    dump(stage_path(stage), {'context_id':c['id'], 'source_sha256':git_info()[2],
                            'experiment_sha256':experiment_hash(), 'results':rows})
    pd.DataFrame(rows).to_csv(EVIDENCE /f'{stage}.csv', index=False)

def export_runs(c):
    exp = mlflow.get_experiment_by_name(PROJECT)
    runs = mlflow.search_runs([exp.experiment_id], filter_string=f"tags.context_id = '{c['id']}' and tags.horizonte = '3'", max_results=100000)
    runs.to_csv(DOCS / 'corridas.csv', index=False)
    return runs

def candidates(c):
    cached = read_stage('candidatos',c)
    if cached is not None:
        return cached
    rows = []
    # Las dos líneas base se entrenan ANTES de los cinco candidatos de t+3.
    for h in HORIZONS:
        rows.append(log_config(c,h,'dummy',DummyClassifier(strategy='most_frequent'),'baseline'))
        rows.append(log_config(c,h,'estacional',None,'baseline'))
    for h in HORIZONS:
        for name,est in models().items():
            rows.append(log_config(c,h,name,est,'candidato'))
            save_stage('candidatos_parcial',c,rows)
            export_runs(c)
    for row in rows:
        if row['phase']=='candidato':
            bases = [r for r in rows if r['horizon']==row['horizon'] and r['phase']=='baseline']
            row['beats_both_baselines'] = bool(all(row['cv_ap_mean']>b['cv_ap_mean'] for b in bases))
    save_stage('candidatos',c,rows); export_runs(c)
    return rows

def tuning(c, n_iter=30):
    if n_iter < 30:
        raise ValueError('La entrega requiere al menos 30 configuraciones por modelo.')
    cached = read_stage('tuning',c)
    if cached is not None:
        return cached
    rows = read_stage('candidatos',c)
    if rows is None:
        raise RuntimeError('Ejecutar fase candidatos primero.')
    best = []
    all_trials = []
    for h in HORIZONS:
        top = sorted([r for r in rows if r['horizon']==h and r['phase']=='candidato'],
                     key=lambda r:r['cv_ap_mean'], reverse=True)[:3]
        for default in top:
            name = default['name']
            parent_id=tuning_parent(c,h,name)
            with mlflow.start_run(run_id=parent_id):
                mlflow.set_tags({'fase':'tuning_padre','familia':name,'context_id':c['id'],'horizonte':str(h)})
                mlflow.log_params({'seed':SEED,'n_configurations':n_iter})
                trials = []
                for params in ParameterSampler(spaces(name), n_iter=n_iter, random_state=SEED):
                    est = models()[name].set_params(**params)
                    trial = log_config(c,h,name,est,'tuning',params,nested=True)
                    trials.append(trial); all_trials.append(trial)
                    pd.DataFrame(all_trials).to_csv(EVIDENCE /'tuning_trials.csv', index=False)
                    export_runs(c)
                winner = max(trials, key=lambda r:r['cv_ap_mean'])
                improvement = winner['cv_ap_mean']-default['cv_ap_mean']
                winner['improvement'] = improvement
                winner['improvement_exceeds_std'] = bool(improvement>max(default['cv_ap_std'],winner['cv_ap_std']))
                # Una ganancia menor a la variación no se presenta como mejora demostrada.
                winner['default_run_id'] = default['run_id']
                best.append(winner)
                mlflow.log_metrics({'best_cv_ap':winner['cv_ap_mean'], 'improvement':improvement})
    save_stage('tuning',c,best); export_runs(c)
    return best

def ablation(c):
    cached = read_stage('ablacion',c)
    if cached is not None:
        return cached
    rows=[]
    # Dos ratios derivados (y estacional relativa) retirados conjuntamente.
    defaults=read_stage('candidatos',c)
    if defaults is None:
        raise RuntimeError('Ejecutar candidatos antes de la ablación.')
    for h in HORIZONS:
        with_features=next(r for r in defaults if r['name']=='logistica' and r['horizon']==h)
        without=log_config(c,h,'logistica',models()['logistica'],'ablacion',derived=False)
        rows.append({'horizon':h,'run_id':without['run_id'], 'without_ap':without['cv_ap_mean'],
                     'with_ap':with_features['cv_ap_mean'],
                     'delta':with_features['cv_ap_mean']-without['cv_ap_mean'],
                     'exceeds_std':bool(with_features['cv_ap_mean']-without['cv_ap_mean']>
                                        max(with_features['cv_ap_std'],without['cv_ap_std']))})
    save_stage('ablacion',c,rows); export_runs(c)
    return rows

def select_finalists(rows):
    """Top dos familias por AP de CV; defaults retenidos si tuning no mejora más de SD."""
    pool = {}
    for r in rows:
        pool.setdefault(r['name'],[]).append(r)
    selected=[]
    for family, configs in pool.items():
        default=next(r for r in configs if r['phase']=='candidato')
        tuned=[r for r in configs if r['phase']=='tuning' and r.get('improvement_exceeds_std')]
        selected.append(max(tuned+[default],key=lambda r:r['cv_ap_mean']))
    return sorted(selected, key=lambda r:(-r['cv_ap_mean'],r['fit_seconds']))[:2]

def explain_training(pipe, x, y, h, alias):
    """Importancia descriptiva en muestra de TRAIN; no vuelve a consultar hold-out."""
    from sklearn.inspection import permutation_importance
    rng=np.random.default_rng(SEED)
    idx=np.sort(rng.choice(len(x),size=min(2000,len(x)),replace=False))
    importance=permutation_importance(pipe,x.iloc[idx],y[idx],scoring='average_precision',
                                     n_repeats=3,random_state=SEED)
    result=pd.DataFrame({'feature':x.columns,'ap_drop_mean':importance.importances_mean,
                         'ap_drop_std':importance.importances_std}).sort_values('ap_drop_mean',ascending=False)
    path=EVIDENCE /f'explicacion_t{h}_{alias}.csv'
    result.to_csv(path,index=False)
    mlflow.log_artifact(str(path),'explicabilidad')
    model=pipe.named_steps['model']
    if hasattr(model,'coef_'):
        names=pipe.named_steps['preparation'].get_feature_names_out()
        coefficients=pd.DataFrame({'feature':names,'coefficient':model.coef_[0]})
        coef_path=EVIDENCE /f'coeficientes_t{h}_{alias}.csv'
        coefficients.to_csv(coef_path,index=False); mlflow.log_artifact(str(coef_path),'explicabilidad')
    return result.head(5).feature.tolist()

def final(c):
    cached=read_stage('seleccion',c)
    if cached is not None:
        return cached  # Nunca reabrir test para volver a seleccionar.
    defaults=read_stage('candidatos',c)
    tuned=read_stage('tuning',c)
    if defaults is None or tuned is None:
        raise RuntimeError('Completar candidatos y tuning antes del hold-out.')
    lock=DATA/'holdout_opened.json'
    if lock.exists():
        raise RuntimeError('Hold-out ya abierto: revisar fallo con sus resultados parciales, sin reseleccionar.')
    # Congelar ambos finalistas y umbrales antes de cualquier inferencia sobre test.
    plan={str(h):select_finalists([r for r in defaults+tuned if r['horizon']==h and r['phase'] in ('candidato','tuning')])
          for h in HORIZONS}
    dump(lock,plan)
    results=[]
    client=mlflow.MlflowClient()
    for h in HORIZONS:
        finalists=plan[str(h)]
        x=model_input(c['test'],h)
        y,q=label(c['test'],c['p90'],h)
        business_x=x.copy(); business_x['p90']=q
        business=SeasonalClassifier().fit(business_x,y)
        baseline=score(y,business.predict_proba(business_x)[:,1],prediction=business.predict(business_x))
        dif=finalists[0]['cv_ap_mean']-finalists[1]['cv_ap_mean']
        larger_sd=max(r['cv_ap_std'] for r in finalists)
        # El ranking se fija en CV: test confirma desempeño, nunca cambia el ganador.
        for rank,row in enumerate(finalists):
            pipe=joblib.load(DATA/f'{row["run_id"]}.joblib')
            tick=time.perf_counter(); p=pipe.predict_proba(x)[:,1]; elapsed=time.perf_counter()-tick
            metrics=score(y,p,row['threshold'])
            metrics['inference_ms_per_row']=1000*elapsed/len(x)
            reduction=fn_reduction(metrics['fn'],baseline['fn'])
            alias='champion' if rank==0 else 'challenger'
            with mlflow.start_run(run_name=f'final-t{h}-{alias}') as run:
                commit,dirty,source_hash=git_info()
                mlflow.set_tags({'fase':'final','familia':row['name'],'context_id':c['id'],
                                 'horizonte':str(h),'selected_run_id':row['run_id'],
                                 'git_commit':commit,'git_dirty':str(dirty),'source_sha256':source_hash,
                                 'synthetic':str(c['info'].get('synthetic',False)).lower()})
                mlflow.log_params({'threshold':row['threshold'],'seed':SEED,
                    'dataset_sha256':c['info']['sha256'],'source_table':c['info']['source_table'],
                    'snapshot_end':c['info']['snapshot_end'],'cost_fn':COST_FN,'cost_fp':COST_FP,
                    **{f'model__{k}':str(v) for k,v in pipe.named_steps['model'].get_params().items()}})
                mlflow.log_metrics({f'test_{k}':v for k,v in metrics.items()})
                mlflow.log_metrics({k:row[k] for k in ('train_ap','cv_ap_mean','cv_ap_std','gap','fit_seconds')})
                if reduction is not None:
                    mlflow.log_metric('fn_reduction_vs_business',reduction)
                mlflow.log_dict({'baseline_test':baseline,'fn_reduction':reduction,
                                  'goal_10pct':reduction is not None and reduction>=.10,
                                  'cv_difference':dif,'cv_difference_exceeds_sd':dif>larger_sd,
                                  'precision_constraint_feasible_oof':row['precision_constraint_feasible']},'decision.json')
                mlflow.log_dict(c['info'],'snapshot.json'); mlflow.log_dict(c['split'],'split.json')
                train_y,_=label(c['train'],c['p90'],h)
                important=explain_training(pipe,model_input(c['train'],h),train_y,h,alias)
                info=mlflow.sklearn.log_model(pipe,name='pipeline',serialization_format='cloudpickle',input_example=x.head(3),
                    signature=infer_signature(x.head(3),pipe.predict_proba(x.head(3))),
                    pyfunc_predict_fn='predict_proba',code_paths=[str(ROOT/'src')])
                # Registro exclusivo de t+3; champion y challenger son sus dos finalistas.
                registry_name=f'{PROJECT}-t{h}-modelo'
                version=mlflow.register_model(info.model_uri,registry_name)
                client.set_registered_model_alias(registry_name,alias,version.version)
                item={'horizon':h,'alias':alias,'name':row['name'],'run_id':run.info.run_id,
                      'selected_run_id':row['run_id'],'version':str(version.version),'registry_name':registry_name,
                      'threshold':row['threshold'],'cv_ap_mean':row['cv_ap_mean'],'cv_ap_std':row['cv_ap_std'],
                      'cv_difference':dif,'cv_difference_exceeds_sd':dif>larger_sd,
                      'business_baseline':baseline,'fn_reduction':reduction,
                      'important_features_train':important,**metrics}
                results.append(item)
                dump(DATA/'seleccion_parcial.json',results)
                # Predicciones y estratos guardados para inspección, sin nueva decisión.
                pred=c['test'][['fecha','cod_canton']].copy()
                pred['y']=y; pred['prob']=p; pred['pred']=p>=row['threshold']
                pred.to_parquet(DATA/f'pred_t{h}_{alias}.parquet',index=False)
                canton_means=c['cal'].groupby('cod_canton').n_emergencias.mean()
                ranks=canton_means.rank(pct=True)
                strata=pd.cut(ranks,[0,1/3,2/3,1],labels=['bajo','medio','alto'],include_lowest=True)
                pred['estrato']=pred.cod_canton.map(strata)
                strata_metrics=[]
                for s,g in pred.groupby('estrato',observed=True):
                    strata_metrics.append({'estrato':str(s),'rows':len(g),**score(g.y,g.prob,row['threshold'])})
                pd.DataFrame(strata_metrics).to_csv(EVIDENCE /f'estratos_t{h}_{alias}.csv',index=False)
                mlflow.log_dict({'strata':strata_metrics},'estratos.json')
    save_stage('seleccion',c,results); export_runs(c)
    return results

def verify_registry(c):
    results=read_stage('seleccion',c)
    if results is None:
        return {'status':'PENDIENTE: ejecutar selección con OBT real.'}
    checks=[]
    for row in results:
        h=row['horizon']
        loaded=mlflow.sklearn.load_model(f'models:/{row["registry_name"]}@{row["alias"]}')
        x=model_input(c['test'],h); y,_=label(c['test'],c['p90'],h)
        prob=loaded.predict_proba(x)[:,1]
        saved=pd.read_parquet(DATA/f'pred_t{h}_{row["alias"]}.parquet')
        np.testing.assert_array_equal(prob,saved.prob.to_numpy())
        reproduced=score(y,prob,row['threshold'])['ap']
        if reproduced != row['ap']:
            raise AssertionError('AP no idéntica al registro guardado.')
        recorded=mlflow.MlflowClient().get_run(row['run_id']).data.metrics['test_ap']
        if reproduced != recorded:
            raise AssertionError('AP distinta de la métrica registrada en MLflow.')
        checks.append({'horizon':h,'alias':row['alias'],'ap':reproduced,'exact':True})
    dump(EVIDENCE /'verificacion_registry.json',checks)
    return checks

# LLM
import json
import os
import time
import mlflow
import numpy as np
import requests
from .pipeline import model_input, FeatureBuilder

PROMPT = '''Eres un clasificador tabular de demanda de emergencias. Estima la probabilidad
de que el conteo del día objetivo supere estrictamente el P90 histórico del cantón
y weekday objetivo. El P90 se calculó en un año anterior al origen de predicción.
Solo usa la información numérica provista. Los conteos históricos se conocen al cierre
del día de ejecución; no tienes acceso a eventos futuros. No busques información externa.
Devuelve únicamente JSON válido con esta estructura: {"probability": 0.0}.
La probabilidad debe estar entre 0 y 1. No incluyas explicaciones ni markdown.
'''

def llm_probability(row):
    """Consulta al modelo local configurado; misma función para prueba y experimento."""
    headers={'Content-Type':'application/json'}
    if os.getenv('LLM_API_KEY'):
        headers['Authorization']='Bearer '+os.environ['LLM_API_KEY']
    try:
        response=requests.post(LLM_BASE_URL+'/chat/completions',headers=headers,timeout=120,json={
            'model':LLM_MODEL,'temperature':0,'seed':SEED,'max_tokens':128,
            'response_format':{'type':'json_object'},
            'messages':[{'role':'system','content':PROMPT},
                        {'role':'user','content':json.dumps(row,default=float,allow_nan=False)}]})
    except requests.ConnectionError as exc:
        raise RuntimeError('No se puede conectar con Ollama. Abrir Ollama y comprobar que el modelo esté instalado.') from exc
    if response.status_code!=200:
        raise RuntimeError(f'Endpoint LLM devolvió HTTP {response.status_code}; comprobar modelo y servicio.')
    content=response.json()['choices'][0]['message']['content']
    result=json.loads(content)
    probability=float(result['probability'])
    if not np.isfinite(probability) or not 0<=probability<=1:
        raise ValueError('Probabilidad LLM fuera de rango.')
    return {'probability':probability}

def llm_trial(c):
    url=LLM_BASE_URL
    model=LLM_MODEL
    if not url or not model:
        status={'status':'PENDIENTE','reason':'Sin endpoint/modelo LLM configurados; no se ha ejecutado prompting.',
                'applicable':True,'design':'Zero-shot sobre muestra del último fold de CV; no se considera competidor del ranking completo.'}
        dump(EVIDENCE /'llm_status.json',status)
        return status
    requested_rows=int(os.getenv('LLM_MAX_ROWS','100'))
    cache_key={'context_id':c['id'],'model':model,'base_url':url,'rows':requested_rows,
               'prompt_sha256':hashlib.sha256(PROMPT.encode()).hexdigest()}
    cached=DATA/'llm_t3.json'
    if cached.exists():
        saved=json.loads(cached.read_text())
        if saved.get('key')==cache_key:
            outputs=[r for r in saved['results'] if r.get('horizon')==3]
            dump(EVIDENCE/'llm_status.json',outputs)
            return outputs
    old_llm=DATA/'llm.json'
    if old_llm.exists() and legacy_t3_compatible():
        old=[r for r in json.loads(old_llm.read_text()) if r.get('horizon')==3]
        old_metadata=DATA/'evidencia'/'ollama_metadata.json'
        metadata=json.loads(old_metadata.read_text()) if old_metadata.exists() else {}
        old_candidates=json.loads((DATA/'candidatos.json').read_text()) if (DATA/'candidatos.json').exists() else {}
        if (old and old_candidates.get('context_id')==c['id'] and metadata.get('model')==model
                and url in {'http://127.0.0.1:11434/v1','http://localhost:11434/v1'}
                and requested_rows==old[0]['rows']
                and cache_key['prompt_sha256']==LEGACY_LLM_PROMPT_SHA256):
            dump(cached,{'key':cache_key,'results':old}); dump(EVIDENCE/'llm_status.json',old)
            dump(EVIDENCE/'ollama_metadata.json',metadata)
            export_runs(c)
            return old
    outputs=[]
    for h in HORIZONS:
        x=model_input(c['train'],h)
        y,q=label(c['train'],c['p90'],h)
        last=c['folds'][-1][1]
        n=min(requested_rows,len(last))
        rng=np.random.default_rng(SEED)
        indices=np.sort(rng.choice(last,n,replace=False))
        if len(np.unique(y[indices]))<2:
            raise ValueError('La muestra LLM no contiene ambas clases: aumentar LLM_MAX_ROWS antes de ejecutar.')
        features=FeatureBuilder(h).transform(x.iloc[indices]).reset_index(drop=True)
        p=[]; replies=[]
        start=time.perf_counter()
        with mlflow.start_run(run_name=f'prompting-t{h}') as run:
            commit,dirty,source_hash=git_info()
            mlflow.set_tags({'fase':'llm','familia':'llm_prompting','context_id':c['id'],'horizonte':str(h),
                            'git_commit':commit,'git_dirty':str(dirty),'source_sha256':source_hash})
            mlflow.log_params({'model':model,'temperature':0,'seed':SEED,'rows':n,
                              'dataset_sha256':c['info']['sha256'],'source_table':c['info']['source_table'],
                              'snapshot_end':c['info']['snapshot_end'],'cost_fn':COST_FN,'cost_fp':COST_FP})
            mlflow.log_text(PROMPT,'prompt.txt')
            for j,i in enumerate(indices):
                row=features.iloc[j].drop(['cod_canton','provincia','region']).to_dict()
                row['p90_objetivo']=float(q[i]); row['horizonte']=h
                result=llm_probability(row)
                probability=result['probability']
                p.append(probability)
                replies.append({'index':int(i),'features':row,'response':result})
            metrics=score(y[indices],np.array(p),threshold=.70)
            mlflow.log_metrics({**metrics,'elapsed_seconds':time.perf_counter()-start})
            mlflow.log_dict({'responses':replies},'responses.json')
            # Comparación justa: solo las mismas filas, usando predicciones OOF ya existentes.
            comparison=[]
            candidates=read_stage('candidatos',c)
            for r in candidates:
                if r['horizon']!=h:
                    continue
                predictions=np.load(DATA/f'{r["run_id"]}_oof.npz')['p'][indices]
                comparison.append({'name':r['name'],**score(y[indices],predictions,r['threshold'],
                    prediction=(predictions>.5).astype(int) if r['name']=='estacional' else None)})
            item={'horizon':h,'run_id':run.info.run_id,'metrics':metrics,'rows':n,
                  'comparison_same_rows':comparison,'status':'EJECUTADO',
                  'limitation':'Muestra pequeña; no sustituye CV completa, temperatura 0 no garantiza determinismo del proveedor.'}
            mlflow.log_dict(item,'comparison.json'); outputs.append(item)
    dump(cached,{'key':cache_key,'results':outputs}); dump(EVIDENCE /'llm_status.json',outputs); export_runs(c)
    return outputs

# VISUALIZACIONES PARA LOS NOTEBOOKS

def read_json(path, default=None):
    """Lee metadatos locales; no genera documentos."""
    return json.loads(path.read_text()) if path.exists() else default

def plot_candidates(rows):
    """Comparación de AP de CV de los modelos t+3; devuelve una imagen para el notebook."""
    rows=[r for r in rows if r.get('horizon')==3]
    if not rows:
        raise ValueError('Faltan resultados de candidatos t+3.')
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(9,3.5))
    ax.errorbar(range(len(rows)),[r['cv_ap_mean'] for r in rows],
                yerr=[r['cv_ap_std'] for r in rows],fmt='o',capsize=4)
    ax.set_xticks(range(len(rows)),[r['name'] for r in rows],rotation=20,ha='right')
    ax.set(ylabel='AP de CV ± desviación',title='Comparación de modelos — t+3')
    fig.tight_layout()
    path=EVIDENCE/'comparacion_modelos.png'
    fig.savefig(path,dpi=160);plt.close(fig)
    return path

def plot_tuning(trials):
    """Curva de mejor AP por configuración completada, exclusivamente para t+3."""
    trials=trials[pd.to_numeric(trials['horizon'],errors='coerce').eq(3)]
    if trials.empty:
        raise ValueError('Faltan configuraciones de tuning t+3.')
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(9,3.5))
    for name,group in trials.groupby('name'):
        ax.plot(range(1,len(group)+1),group.cv_ap_mean.cummax(),label=name)
    ax.set(xlabel='Configuraciones completadas',ylabel='Mejor AP de CV acumulada',title='Tuning — t+3')
    ax.legend();fig.tight_layout()
    path=EVIDENCE/'tuning.png'
    fig.savefig(path,dpi=160);plt.close(fig)
    return path

def plot_selection(rows):
    """Falsos negativos del baseline y de ambos finalistas, sin reevaluar el hold-out."""
    rows=[r for r in rows if r.get('horizon')==3]
    if len(rows)!=2:
        raise ValueError('Se necesitan los dos finalistas de t+3.')
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(7,3.5))
    ax.bar(['Estacional']+[r['alias'] for r in rows],
           [rows[0]['business_baseline']['fn']]+[r['fn'] for r in rows],
           color=['#6b8294','#167d9a','#ad7a43'])
    ax.set(ylabel='Días altos no anticipados',title='Falsos negativos en prueba — t+3')
    fig.tight_layout()
    path=EVIDENCE/'decision_fn.png'
    fig.savefig(path,dpi=160);plt.close(fig)
    return path


# RUN
import argparse
import json


def execute_notebooks():
    """Un kernel nuevo por notebook y salidas guardadas."""
    import nbformat
    import sys
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
        nb=nbformat.read(path,as_version=4)
        manager=KernelManager(kernel_name='python3')
        manager.kernel_spec.argv[0]=sys.executable
        NotebookClient(nb,timeout=None,km=manager,
                       resources={'metadata':{'path':str(ROOT)}}).execute()
        nbformat.write(nb,path)
    return {'notebooks':5,'status':'ejecutados en kernels nuevos'}

def capture_mlflow():
    """La captura intermedia se guarda fuera del árbol de entrega."""
    from urllib.parse import urlencode
    from playwright.sync_api import sync_playwright
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    exp=mlflow.get_experiment_by_name(PROJECT)
    if exp is None:
        raise RuntimeError('Ejecutar primero los experimentos.')
    runs=mlflow.search_runs([exp.experiment_id],filter_string="tags.fase = 'candidato' and tags.horizonte = '3'",max_results=5)
    if len(runs)<2:
        raise RuntimeError('Faltan al menos dos candidatos reales para comparar.')
    query=urlencode({'runs':json.dumps(runs.run_id.tolist()),'experiments':json.dumps([exp.experiment_id])})
    url=f'{TRACKING_URI}/#/compare-runs?{query}'
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True)
        page=browser.new_page(viewport={'width':1600,'height':1000})
        page.goto(url,wait_until='domcontentloaded',timeout=30000)
        page.get_by_text('Parameters',exact=True).first.wait_for(timeout=30000)
        page.get_by_text(runs.iloc[0]['tags.mlflow.runName'],exact=False).first.wait_for(timeout=30000)
        close=page.get_by_role('button',name='Close',exact=True)
        if close.count():
            close.click()
        page.get_by_role('button',name='Clear All',exact=True).click()
        page.get_by_role('combobox').nth(1).click(force=True)
        page.locator('[class*="select-item-option-content"]').filter(has_text='cv_ap_mean').first.click()
        page.get_by_role('combobox').nth(1).click(force=True)
        page.locator('[class*="select-item-option-content"]').filter(has_text='train_ap').first.click()
        page.get_by_text(f'Comparing {len(runs)} Runs from 1 Experiment',exact=True).click()
        page.screenshot(path=str(EVIDENCE/'mlflow_comparacion.png'),full_page=True)
        browser.close()
    return {'url':url,'captura':str(EVIDENCE/'mlflow_comparacion.png')}

def execute(stage):
    if stage=='notebooks':
        return execute_notebooks()
    if stage=='captura':
        return capture_mlflow()
    if stage=='snapshot':
        return snapshot_from_snowflake()
    c=context()
    if stage=='eda':
        result=run_eda(c['cal'],c['train'],c['p90'])
        full,_=load_snapshot()
        pd.DataFrame({'dtype':full.dtypes.astype(str),'nulls':full.isna().sum(),
                      'missing_fraction':full.isna().mean()}).to_csv(EVIDENCE/'eda/snapshot_schema.csv')
        return result
    if stage=='candidatos':
        return candidates(c)
    if stage=='tuning':
        return tuning(c)
    if stage=='ablacion':
        return ablation(c)
    if stage=='llm':
        result=llm_trial(c)
        if isinstance(result,list) and LLM_BASE_URL.startswith(('http://127.0.0.1:11434','http://localhost:11434')):
            metadata_path=EVIDENCE/'ollama_metadata.json'
            metadata=read_json(metadata_path,{})
            try:
                base=LLM_BASE_URL.removesuffix('/v1')
                tags=requests.get(base+'/api/tags',timeout=5).json()['models']
                selected_model=next(m for m in tags if m['name']==LLM_MODEL)
                version=requests.get(base+'/api/version',timeout=5).json().get('version','unknown')
                metadata={'model':LLM_MODEL,'model_digest':selected_model['digest'],'ollama_version':version}
                dump(metadata_path,metadata)
            except (requests.RequestException,KeyError,ValueError,StopIteration):
                pass
            if metadata.get('model')==LLM_MODEL:
                client=mlflow.MlflowClient()
                for row in result:
                    client.log_param(row['run_id'],'ollama_model_digest',metadata['model_digest'])
                    client.log_param(row['run_id'],'ollama_version',metadata['ollama_version'])
                export_runs(c)
        return result
    if stage=='seleccion':
        return final(c)
    if stage=='verificar':
        return verify_registry(c)
    if stage=='all':
        for phase in ('eda','candidatos','ablacion','llm','tuning','seleccion','verificar'):
            print(f'FASE: {phase}',flush=True)
            execute(phase)
        return {'status':'Fases completadas; corridas exportadas y registro verificado. El documento técnico se redacta por separado.'}
    raise ValueError(stage)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['snapshot','eda','candidatos','tuning','ablacion','llm','seleccion','verificar','all','notebooks','captura'])
    args=parser.parse_args()
    result=execute(args.stage)
    print(json.dumps(result,indent=2,default=str,ensure_ascii=False))

if __name__=='__main__':
    main()
