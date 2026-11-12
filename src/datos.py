"""Datos, EDA, experimentación, tracking y documento técnico del PSet 3.

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
    ignored={'build_report','read_json','execute_notebooks','capture_mlflow','execute','main'}
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

# REPORTE
import json
from pathlib import Path
from html import escape
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image

NAME='PSet3_memo_Tierra_Rodriguez_Rodriguez_Cadena_Jarrin.pdf'

def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default

def build_report():
    """Ocho apartados del enunciado en seis páginas; solo evidencia efectivamente observada."""
    DOCS.mkdir(parents=True,exist_ok=True); EVIDENCE.mkdir(parents=True,exist_ok=True)
    snapshot=read_json(DATA/'snapshot.json',{})
    finding=read_json(EVIDENCE/'eda/findings.json',{})
    default=[r for r in read_json(DATA/'candidatos.json',{}).get('results',[]) if r.get('horizon')==3]
    tuned=[r for r in read_json(DATA/'tuning.json',{}).get('results',[]) if r.get('horizon')==3]
    selected=[r for r in read_json(DATA/'seleccion.json',{}).get('results',[]) if r.get('horizon')==3]
    ablated=[r for r in read_json(DATA/'ablacion.json',{}).get('results',[]) if r.get('horizon')==3]
    llm=read_json(EVIDENCE/'llm_status.json',{})
    split=read_json(DATA/'split.json',{})
    verified=read_json(EVIDENCE/'verificacion_registry.json',[])
    runs=pd.read_csv(DOCS/'corridas.csv') if (DOCS/'corridas.csv').exists() else pd.DataFrame()
    if 'tags.horizonte' in runs:
        runs=runs[pd.to_numeric(runs['tags.horizonte'],errors='coerce').eq(3)]
    else:
        runs=runs.iloc[0:0]
    finished=runs[runs.status.eq('FINISHED')] if 'status' in runs else runs
    counts=finished.get('tags.fase',pd.Series(dtype=str)).value_counts().to_dict()
    real=bool(snapshot) and not snapshot.get('synthetic',True)
    complete=(real and len(selected)==2 and len(tuned)==3 and isinstance(llm,list) and len(llm)==1
              and len(verified)==2 and counts.get('baseline',0)>=2 and counts.get('candidato',0)>=5
              and counts.get('tuning',0)>=90 and counts.get('final',0)>=2)
    styles=getSampleStyleSheet()
    body=ParagraphStyle('body',fontName='Helvetica',fontSize=8.4,leading=10.6,spaceAfter=6)
    small=ParagraphStyle('small',fontName='Helvetica',fontSize=6.9,leading=8.1,spaceAfter=4)
    heading=ParagraphStyle('heading',fontName='Helvetica-Bold',fontSize=13,leading=16,spaceAfter=7,spaceBefore=5)
    story=[]; md=[]
    def p(value,tiny=False):
        story.append(Paragraph(value,small if tiny else body)); md.append(value)
    def title(value):
        story.append(Paragraph(value,heading)); md.append('\n# '+value)
    def table(headers,rows,widths):
        data=[[Paragraph(escape(str(v)),small) for v in row] for row in [headers]+rows]
        tab=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
        tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dcecf1')),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),3),
            ('BOTTOMPADDING',(0,0),(-1,-1),3),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#167d9a'))]))
        story.extend([tab,Spacer(1,6)])
        md.append(' | '.join(headers)); md.extend(' | '.join(map(str,r)) for r in rows)
    def picture(path,height=150):
        if path.exists():
            from PIL import Image as PI
            with PI.open(path) as im:w,h=im.size
            scale=min(490/w,height/h)
            story.append(Image(str(path),width=w*scale,height=h*scale))
    def page():story.append(PageBreak())
    def f(value,n=3):return 'NA' if value is None else f'{value:.{n}f}'
    def pct(value):return 'indefinida' if value is None else f'{value:.1%}'
    names={'dummy':'Dummy','estacional':'Estacional','logistica':'Logística',
           'random_forest':'Random forest','hist_boosting':'Hist boosting','mlp':'MLP','gaussian_nb':'GaussianNB'}
    title('PSet 3 · Experimentación y selección del modelo')
    p('Tierra · Rodríguez · Rodríguez · Cadena · Jarrín · ECU 911',True)
    p('<b>'+('RESULTADOS REALES VERIFICADOS' if complete else 'RESULTADOS PARCIALES: ejecución t+3 pendiente de completar')+'</b>',True)
    p('Objetivo del PSet 1: anticipar alta demanda cantonal a t+3 para revisión humana; '
      'reducir al menos 10% los días de alta demanda no anticipados frente a la regla estacional. '
      'Se conserva y=1 cuando el conteo objetivo supera estrictamente P90 del cantón y weekday.')
    if snapshot:
        p(f'Fuente: {snapshot["source_table"]}. Snapshot {snapshot["date_min"]} a {snapshot["date_max"]}; '
          f'{snapshot["rows"]:,} filas × {snapshot["columns"]} columnas, {snapshot["cantones"]} unidades. '
          'La OBT actual amplía el histórico del PSet 2 hasta septiembre de 2026.',True)
        p('SHA256: '+snapshot['sha256'],True)
    title('1. EDA')
    p('La EDA que guía decisiones usa train; el hold-out permanece reservado. Se revisaron '
      'esquema/nulos, descriptivos, prevalencia, IQR y |Z|>=3. Se usa Spearman por asimetría '
      'y asociaciones monótonas; no se eliminan picos automáticamente porque son el fenómeno objetivo.')
    if finding:
        table(['Hallazgo y evidencia en train','Consecuencia experimental'],[
            [r['hallazgo']+': '+r['evidencia'],r['consecuencia']] for r in finding['findings']], [285,205])
        picture(EVIDENCE/'eda/distribucion_boxplot.png',140)
        p(f'Train: {finding["train_rows"]:,} filas. El año inicial de calibración tiene '
          f'{finding["calibration_rows"]:,} filas y se reserva para P90. Esquema, descriptivos, '
          'nulos y outliers por columna se muestran completos en el notebook 01.',True)
    else:p('EDA pendiente: no se inventan hallazgos.')
    page()
    title('2. Features')
    table(['Derivada / codificación','Justificación y disponibilidad'],[
        ['tendencia_7_28=(media7+1)/(media28+1)','Escalas cantonales muy distintas en EDA; cambio relativo de nivel. Ambas medias usan pasado anterior a t.'],
        ['cambio_semanal=(n_t-lag7)/(lag7+1)','Contraste con la semana previa; n_t conocido al cierre t bajo feed diario.'],
        ['estacional_relativa=(media4_weekday+1)/(media28+1)','EDA muestra patrón semanal. Para t+3 usa exclusivamente t-4,-11,-18,-25.'],
        ['sin/cos del weekday y mes objetivo','Calendario conocido de antemano; representa ciclos sin saltos artificiales.'],
        ['Cantón/provincia/región/weekday','One-hot ajustado al train de cada fold; categorías nuevas ignoradas.']],[240,250])
    p('Prueba de disponibilidad: las derivaciones no usan desplazamientos negativos ni obj_*. '
      'Se excluyen columnas fuera de la allowlist y la marca retrospectiva de anomalía. '
      'La disponibilidad diaria es un supuesto de evaluación retrospectiva: el portal público publica por mes.')
    if ablated:
        table(['Horizonte','AP sin ratios','AP con ratios','Delta / > SD'],[
            [f't+{r["horizon"]}',f(r['without_ap']),f(r['with_ap']),f'{f(r["delta"])} / {r["exceeds_std"]}'] for r in ablated], [90,120,120,160])
        p('Ablación con logística, mismos folds: retirar conjuntamente los tres ratios. Una mejora '
          'menor a la variación entre pliegues no demuestra aporte; no se oculta un resultado negativo.',True)
    title('3. Preparación y validación')
    p(f'Calibración P90: primer año, hasta {split.get("calibration_end","pendiente")}; es parte del '
      'bloque de entrenamiento, separado del ajuste de clasificadores. P90 se congela antes de los folds '
      f'para mantener la misma etiqueta. Hold-out: 20% final de fechas, desde {split.get("test_start","pendiente")}; '
      'se retiran siete días previos al corte y los objetivos finales no observados.')
    p('Tres folds expansivos con TimeSeriesSplit sobre fechas únicas y gap de siete días. Todos los '
      'cantones del día permanecen juntos; max(fecha_train)+7 &lt; min(fecha_validación). Mediana, '
      'StandardScaler y one-hot se ajustan exclusivamente con train en Pipeline/ColumnTransformer de src/pipeline.py.')
    p('Métrica técnica común: Average Precision (AP, PR-AUC no interpolada). Métrica de negocio: '
      '(FN_estacional - FN_modelo)/FN_estacional; indefinida si FN_estacional=0. El umbral se fija '
      f'en OOF minimizando {COST_FN:g}FN+{COST_FP:g}FP con precision>={MIN_PRECISION:.2f}; son supuestos '
      'relativos de sensibilidad, no dólares ni costos validados con ECU 911.')
    picture(EVIDENCE/'eda/correlacion.png',145)
    page()
    title('4. Diseño experimental')
    p('Primero se entrenaron las dos bases para t+3: Dummy de clase más frecuente y '
      'regla del PSet 1 (promedio de cuatro weekdays previos > P90). Su score media/P90 sirve '
      'para ordenar, no es probabilidad calibrada. Cinco familias con defaults salvo semilla: '
      'logística, random forest, hist gradient boosting, MLP y GaussianNB. Igual preparación, fechas y AP.')
    table(['Fase','Plan t+3','Corridas terminadas'],[
        ['Baselines','2',counts.get('baseline',0)],['Candidatos','5',counts.get('candidato',0)],
        ['Tuning','90 hijas + 3 padres',str(counts.get('tuning',0))+' + '+str(counts.get('tuning_padre',0))],
        ['Final','2 finalistas',counts.get('final',0)],['Ablación / LLM','1 / 1',str(counts.get('ablacion',0))+' / '+str(counts.get('llm',0))]], [110,230,150])
    p('MLflow en Docker: backend SQLite y artifact store en volúmenes persistentes, servidos por proxy. '
      'Cada configuración guarda hiperparámetros/semilla, tabla/corte/hash, train AP, CV media y SD, '
      'tiempos, folds, advertencias, fase/familia y commit/hash de código. Se registra el pipeline '
      'completo con firma y ejemplo. Tuning usa corridas hijas; docs/corridas.csv se exporta con mlflow.search_runs.')
    if isinstance(llm,list) and llm:
        table(['Prueba zero-shot Qwen 2.5 1.5B local','AP','Precision / recall'],[
            [f't+{r["horizon"]}: n={r["rows"]}; run {r["run_id"]}',f(r['metrics']['ap']),
             f'{f(r["metrics"]["precision"])} / {f(r["metrics"]["recall"])}'] for r in llm],[300,70,120])
        p('Ollama sin API key: JSON con probability, temperatura 0, prompt y respuestas guardados. '
          'Muestra aleatoria del último fold, sin etiquetas en el prompt y sin test. Se comparó contra '
          'OOF de las mismas filas. Umbral exploratorio .70 del memo; la muestra no participa en el '
          'ranking de CV completa ni prueba calibración.',True)
        r=next(r for r in llm if r['horizon']==3)
        comparison=[[names[a['name']],f(a['ap'])] for a in r['comparison_same_rows']]
        comparison.append(['Qwen 2.5 1.5B',f(r['metrics']['ap'])])
        table(['AP en las MISMAS 100 filas de validación t+3','AP'],comparison,[390,100])
        p('El experimento con LLM es exploratorio y se limita a t+3; la tabla muestra '
          'el resultado observado sin extrapolarlo a otros horizontes.',True)
    else:p('Evaluación LLM pendiente. Ollama local qwen2.5:1.5b está configurado; no se atribuyen resultados sin ejecutar.')
    picture(EVIDENCE/'mlflow_comparacion.png',230)
    if not (EVIDENCE/'mlflow_comparacion.png').exists():p('Captura real de comparación MLflow pendiente.',True)
    page()
    title('5. Resultados')
    if default:
        table(['h / modelo','Train AP','CV AP ± SD','Brecha','LR','Ajuste s'],[
            [f'{r["horizon"]} / {names[r["name"]]}',f(r['train_ap']),
             f'{f(r["cv_ap_mean"])} ± {f(r["cv_ap_std"])}',f(r['gap']),r['learning_rate'],f(r['fit_seconds'],1)]
             for r in default],[110,65,110,60,60,85])
        fig,ax=plt.subplots(figsize=(8,2.9))
        for h,offset in [(3,0)]:
            rows=[r for r in default if r['horizon']==h]
            ax.errorbar(np.arange(len(rows))+offset,[r['cv_ap_mean'] for r in rows],
                yerr=[r['cv_ap_std'] for r in rows],fmt='o',capsize=3,label=f't+{h}')
        ax.set_xticks(range(len(rows)),[names[r['name']] for r in rows],rotation=15,ha='right',fontsize=8)
        ax.set(ylabel='AP de CV ± SD');ax.legend();fig.tight_layout()
        fig.savefig(EVIDENCE/'comparacion_modelos.png',dpi=150);plt.close(fig)
        picture(EVIDENCE/'comparacion_modelos.png',160)
        failure=[f't+{r["horizon"]} {names[r["name"]]}' for r in default if r.get('beats_both_baselines') is False]
        p('No superan ambas bases por AP de CV: '+(', '.join(failure) if failure else 'ninguno')+'.')
        largest=max([r for r in default if r['phase']=='candidato'],key=lambda r:r['gap'])
        p(f'Mayor brecha observada: t+{largest["horizon"]} {names[largest["name"]]}, '
          f'train {f(largest["train_ap"])} frente a CV {f(largest["cv_ap_mean"])} '
          f'(brecha {f(largest["gap"])}), compatible con sobreajuste. La validación temporal '
          'y la dispersión entre folds determinan selección, no el ajuste aparente de train.')
        p('Ajuste s suma entrenamiento en folds y refit final; se registra también tiempo transcurrido '
          'con scoring. LR=NA si no aplica. Advertencias de convergencia se conservan en los artefactos; '
          'un default que no converge sigue siendo evidencia del experimento por defecto.',True)
    else:p('Fase de candidatos aún no completada.')
    page()
    title('6. Tuning')
    p('Se seleccionan los tres mejores defaults de t+3 por AP media de CV. Búsqueda '
      'aleatoria reproducible (ParameterSampler, semilla 42), 30 configuraciones por familia y los '
      'mismos folds. Permite explorar órdenes de magnitud con presupuesto fijo y registrar cada '
      'configuración como nested run con su modelo completo.')
    search={
        'logistica':'C log 1e-4..1e3; class_weight None/balanced; max_iter 500/1000.',
        'random_forest':'80..400 árboles; profundidad None/6/12/20; min hoja 1..30; max_features sqrt/.3/.7; ponderación None/balanced.',
        'hist_boosting':'LR log .01.. .3; 80..300 iter.; 7..63 hojas; L2 log 1e-4..10; min hoja 10..60.',
        'mlp':'(32),(64),(100),(64,32); alpha log 1e-6.. .1; LR log 1e-4..1e-2; 200/400 iter.; early stopping sí/no.',
        'gaussian_nb':'var_smoothing log 1e-12..1e-3.'}
    active=sorted({r['name'] for r in tuned})
    if active:table(['Familia ajustada','Espacio evaluado'],[[names[n],search[n]] for n in active],[120,370])
    p('LR y regularización usan escala logarítmica; complejidad y número de iteraciones tienen '
      'rangos acotados para controlar costo. Si delta AP no supera la mayor SD de default y '
      'ajustado, no se presenta como mejora demostrada y se conserva default para la selección.')
    if tuned:
        table(['h / familia','Configuraciones','AP default','AP ajustada ± SD','Delta / >SD'],[
            [f'{r["horizon"]} / {names[r["name"]]}','30',f(r['cv_ap_mean']-r['improvement']),
             f'{f(r["cv_ap_mean"])} ± {f(r["cv_ap_std"])}',f'{f(r["improvement"])} / {r["improvement_exceeds_std"]}'] for r in tuned],[115,70,85,120,100])
    trials_path=EVIDENCE/'tuning_trials.csv'
    if trials_path.exists():
        trials=pd.read_csv(trials_path)
        fig,ax=plt.subplots(figsize=(8,2.8))
        for (h,n),g in trials.groupby(['horizon','name']):
            ax.plot(range(1,len(g)+1),g.cv_ap_mean.cummax(),label=f't+{h} {names[n]}')
        ax.set(xlabel='Configuraciones',ylabel='Mejor AP acumulada');ax.legend(fontsize=6,ncol=2)
        fig.tight_layout();fig.savefig(EVIDENCE/'tuning.png',dpi=150);plt.close(fig)
        picture(EVIDENCE/'tuning.png',180)
    if not tuned:p('Tuning aún no completado; no se declaran mejoras sin medir.')
    p('Se compara ganador y segundo de familias distintas. Diferencia mayor a la SD es una '
      'regla descriptiva, no prueba formal de significancia. El uso de los mismos folds para tuning '
      'y selección puede ser optimista; test permanece independiente.',True)
    page()
    title('7. Decisión')
    if selected:
        table(['h / alias','Familia','AP CV ± SD','AP test','Recall','Red. FN'],[
            [f'{r["horizon"]}/{r["alias"]}',names[r['name']],f'{f(r["cv_ap_mean"])} ± {f(r["cv_ap_std"])}',
             f(r['ap']),f(r['recall']),pct(r['fn_reduction'])] for r in selected],[90,85,105,65,60,85])
        for h in HORIZONS:
            winner=next(r for r in selected if r['horizon']==h and r['alias']=='champion')
            runner=next(r for r in selected if r['horizon']==h and r['alias']=='challenger')
            p(f't+{h}: {names[winner["name"]]} se eligió por AP media de CV; diferencia '
              f'{f(winner["cv_difference"])} frente a SD máxima {f(max(winner["cv_ap_std"],runner["cv_ap_std"]))}. '
              f'¿Supera variación entre folds? {winner["cv_difference_exceeds_sd"]}. '
              f'Reducción FN={pct(winner["fn_reduction"])}; '
              +('cumple meta del 10%.' if winner['fn_reduction'] is not None and winner['fn_reduction']>=.1 else 'no demuestra cumplir la meta del 10%.'))
        table(['h / alias','Run final / versión Registry','Umbral / precisión / costo* / ms-fila'],[
            [f'{r["horizon"]}/{r["alias"]}',f'{r["run_id"]} / v{r["version"]}',
             f'{f(r["threshold"])} / {f(r["precision"])} / {f(r["cost_per_1000"],1)} / {f(r["inference_ms_per_row"],4)}'] for r in selected],[90,240,160])
        p('*Costo relativo por 1.000 cantón-días; supuestos 5FN+FP. Finalistas y umbrales '
          'se congelaron en CV/OOF antes de abrir test. Test no modifica ranking, hiperparámetros '
          'ni umbral. Se mide inferencia en el mismo entorno; no son costos monetarios.',True)
        for r in selected:
            if r['alias']=='champion':
                p(f't+{r["horizon"]}, importancia por permutación en train: '+', '.join(r.get('important_features_train',[])[:3])+'. '
                  'Es descriptiva y no causal; logística añade coeficientes, modelos complejos requieren inspección.',True)
        p('Registry: ecu911-alta-demanda-t3-modelo, aliases '
          'champion/challenger. Notebook 05 carga por alias en kernel nuevo y exige predicciones '
          'idénticas y AP igual al run final. '+f'Reproducciones exactas verificadas: {len(verified)}/2.',True)
    else:p('Selección y evaluación final aún pendientes.')
    if selected:
        fig,ax=plt.subplots(figsize=(6,2.1))
        group=[r for r in selected if r['horizon']==3]
        baseline=group[0]['business_baseline']['fn']
        ax.bar(['Estacional','Champion','Challenger'],[baseline]+[r['fn'] for r in group],color=['#6b8294','#167d9a','#ad7a43'])
        ax.set(title='t+3',ylabel='Días altos no anticipados')
        ax.tick_params(axis='x',labelsize=7)
        fig.tight_layout();fig.savefig(EVIDENCE/'decision_fn.png',dpi=150);plt.close(fig)
        picture(EVIDENCE/'decision_fn.png',110)
    title('8. Limitaciones')
    p('La fuente pública mensual no satisface la disponibilidad diaria asumida; esta es evaluación '
      'retrospectiva condicionada a datos oportunos. Enero 2024 se conserva con anomalía sin causa '
      'confirmada; no se usa su marca como predictor. P90 congelado puede perder vigencia y la '
      'prevalencia cambia. Geografía del PSet 2 contiene correcciones retrospectivas. Sin población '
      'INEC, capacidad operativa ni ID de incidente no se concluye saturación, exactitud factual '
      'o reducción de tiempos de respuesta. Costos y precisión mínima son supuestos; requieren '
      'validación institucional. Tuning/selección comparten folds; Qwen se evalúa en una muestra '
      'pequeña y temperatura cero no garantiza estabilidad entre versiones.')
    p('Fuentes: CD_PSet_3.pdf; memo PSet 1; github.com/junilai/Pset_2 '
      '(cc4df2dc56d9f17691956ba9c325245726eeba9b); documentación oficial de sklearn y MLflow.',True)
    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor('#167d9a'));canvas.line(42,805,553,805)
        canvas.setFont('Helvetica',7);canvas.drawString(42,815,'ECU 911 · PSet 3 · Experimentación con evidencia')
        canvas.drawRightString(553,24,str(doc.page))
    SimpleDocTemplate(str(DOCS/NAME),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=48,bottomMargin=40).build(story,onFirstPage=footer,onLaterPages=footer)
    (EVIDENCE/'memo.md').write_text('\n\n'.join(md))
    state={'real_snapshot':real,'real_results':len(selected)==2,'baseline_runs':sum(r['phase']=='baseline' for r in default),'candidate_runs':sum(r['phase']=='candidato' for r in default),
           'tuned_families':len(tuned),'final_runs':len(selected),'registry_verified':len(verified)==2,
           'mlflow_screenshot':(EVIDENCE/'mlflow_comparacion.png').exists(),'llm_executed':isinstance(llm,list),
           'ready_to_submit':complete and (EVIDENCE/'mlflow_comparacion.png').exists(),'memo':NAME}
    dump(EVIDENCE/'estado_entrega.json',state)
    return state


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
    if stage=='report':
        return build_report()
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
        for phase in ('eda','candidatos','ablacion','llm','tuning','seleccion','verificar','report'):
            print(f'FASE: {phase}',flush=True)
            execute(phase)
        return {'status':'Completado; revisar estado_entrega.json en la caché local y limitaciones LLM/captura.'}
    raise ValueError(stage)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['snapshot','eda','candidatos','tuning','ablacion','llm','seleccion','verificar','report','all','notebooks','captura'])
    args=parser.parse_args()
    result=execute(args.stage)
    print(json.dumps(result,indent=2,default=str,ensure_ascii=False))

if __name__=='__main__':
    main()
