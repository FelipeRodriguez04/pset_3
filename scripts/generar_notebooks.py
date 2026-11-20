from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
NOTEBOOKS.mkdir(parents=True, exist_ok=True)


SETUP = """from pathlib import Path
import os, sys
ROOT = Path.cwd()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from IPython.display import Image, display
from src.experimentos import *
df, data, info = cargar_contexto()
print(f'Fuente: {info.source}')
print(f'OBT: {info.rows:,} filas x {info.columns} columnas; corte {info.snapshot_end}')
print(f'Train: {len(data.train):,}; holdout: {len(data.test):,}; target positivo: {data.train[data.target_column].mean():.2%}')
print(f'Validación: 4 semestres completos; purga: {data.purge_days} días; features: {len(data.feature_columns)}')
"""


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def write(name: str, title: str, cells):
    notebook = nbf.v4.new_notebook(
        cells=[markdown(f"# Juan Diego Cadena, 329220\n\n## {title}"), *cells],
        metadata={
            "kernelspec": {"display_name": "Python 3 (PSet 3)", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
    )
    nbf.write(notebook, NOTEBOOKS / name)


write(
    "01_eda.ipynb",
    "01 - Exploración de la OBT",
    [
        markdown(
            "La exploración se hace antes de ajustar modelos. Uso **Spearman** porque los conteos y lags "
            "tienen cola larga y la relación esperada es monotónica, no necesariamente lineal. Los picos "
            "no se eliminan: son precisamente los días que el proyecto quiere anticipar."
        ),
        code(SETUP),
        code("eda = ejecutar_eda(df, data, info)\nimport pandas as pd\npd.DataFrame(eda['findings'])"),
        code(
            "for plot in ['01_target_distribucion.png','02_boxplot_outliers.png','03_correlacion_spearman.png','04_volumen_mensual.png','08_eda_feriados.png']:\n"
            "    display(Image(filename=str(ROOT/'docs'/'plots'/plot), width=850))"
        ),
        code("pd.read_csv(ROOT/'docs'/'eda_columnas.csv').head(40)"),
    ],
)

write(
    "02_features.ipynb",
    "02 - Features y control de fuga",
    [
        markdown(
            "Las features nuevas incluyen rezagos de 364/365/371 días, calendario de la fecha objetivo, ciclos "
            "anuales, ventanas de feriado, el promedio de las cuatro semanas equivalentes y tendencia 7/28. Todas existen al cierre "
            "del día de ejecución. `obj_n_emergencias_t3/t7` solo construye la etiqueta y nunca entra a X."
        ),
        code(SETUP),
        code(
            "import pandas as pd\n"
            "pd.DataFrame({\n"
            "'feature':['media_mismo_dia_4','media_mismo_periodo_anio','ratio_estacional_p90','temporada_objetivo','dias_al_feriado_mas_cercano'],\n"
            "'evidencia':['baseline estacional','mismo periodo del año anterior','niveles distintos entre cantones','Carnaval/fin de año/feriados','calendario oficial'],\n"
            "'disponible_en_inferencia':['sí: lags 7/14/21/28','sí: lags 364/365/371','sí: P90 solo train','sí: fecha conocida','sí: calendario conocido'],\n"
            "'codificacion':['numérica','numérica','numérica','categórica','numérica']})"
        ),
        code(
            "forbidden={'obj_n_emergencias_t3','obj_n_emergencias_t7','fecha_objetivo'}\n"
            "assert not forbidden.intersection(data.feature_columns)\n"
            "assert data.train.fecha_objetivo.max() < data.test.fecha.min()\n"
            "print('Sin columnas futuras, con purga t+3 y sin solapamiento temporal.')\n"
            "data.train[['fecha','fecha_objetivo','cod_canton','lag_364d','temporada_objetivo','es_ventana_feriado_objetivo',data.target_column]].tail(12)"
        ),
        markdown(
            "El percentil 90 se calcula exclusivamente con el conjunto de entrenamiento y se aplica después "
            "al holdout. La codificación ordinal dentro del `ColumnTransformer` evita una matriz one-hot muy "
            "grande y mantiene el experimento dentro de la memoria de la laptop."
        ),
    ],
)

write(
    "03_baseline_candidatos.ipynb",
    "03 - Líneas base y modelos candidatos",
    [
        markdown(
            "Se comparan la prevalencia (`DummyClassifier`), la regla de las últimas cuatro semanas y cinco "
            "familias. Todos usan cuatro semestres completos, una purga t+3 y PR-AUC ponderado para dar más importancia a feriados. Un LLM por prompting no es "
            "adecuado: no recibe 429 mil filas tabulares de forma reproducible ni entrega probabilidades "
            "calibradas; además implicaría enviar datos a un servicio externo."
        ),
        code(SETUP),
        code(
            "resultados = ejecutar_baselines_y_candidatos(data, info, force=False)\n"
            "resultados[['modelo','familia','cv_pr_auc_mean','cv_pr_auc_ponderado_mean','cv_pr_auc_feriados_mean','cv_pr_auc_fin_anio_mean','generalization_gap']]"
        ),
        code(
            "import matplotlib.pyplot as plt\n"
            "ax=resultados.sort_values('cv_pr_auc_ponderado_mean').plot.barh(x='modelo',y='cv_pr_auc_ponderado_mean',xerr='cv_pr_auc_ponderado_std',legend=False,figsize=(9,5))\n"
            "ax.set_xlabel('PR-AUC ponderado'); ax.set_title('Cuatro backtests semestrales'); plt.tight_layout(); plt.show()"
        ),
    ],
)

write(
    "04_tuning.ipynb",
    "04 - Ajuste de hiperparámetros",
    [
        markdown(
            "Se usa `ParameterSampler` porque 30 configuraciones por familia permiten cubrir escalas "
            "logarítmicas sin recorrer un producto cartesiano muy grande. Cada configuración queda como "
            "corrida hija de MLflow y usa los mismos cuatro backtests semestrales purgados."
        ),
        code(SETUP),
        code(
            "tuning = ejecutar_tuning(data, info, force=False)\n"
            "tuning[['modelo','configuraciones','base_cv_pr_auc_ponderado','best_cv_pr_auc_mean','best_cv_pr_auc_ponderado_mean','best_cv_pr_auc_feriados_mean','improvement_vs_default']]"
        ),
        code(
            "import pandas as pd, matplotlib.pyplot as plt\n"
            "fig,axes=plt.subplots(1,len(tuning),figsize=(5*len(tuning),4),squeeze=False)\n"
            "for ax,model in zip(axes[0],tuning.modelo):\n"
            "    d=pd.read_csv(ROOT/'docs'/f'tuning_{model}.csv')\n"
            "    ax.scatter(range(1,len(d)+1),d.cv_pr_auc_mean,s=20); ax.set_title(model); ax.set_xlabel('configuración'); ax.set_ylabel('PR-AUC CV')\n"
            "plt.tight_layout(); plt.show()"
        ),
    ],
)

write(
    "05_seleccion.ipynb",
    "05 - Selección y Model Registry",
    [
        markdown(
            "El holdout de abril-septiembre de 2026 se abre una sola vez para el ganador y el segundo. El "
            "umbral intenta reducir por lo menos 10% los picos no anticipados frente a la regla estacional, "
            "manteniendo una precisión mínima; como respaldo minimiza `15*FN + FP`."
        ),
        code(SETUP),
        code("seleccion = ejecutar_seleccion(data, info, df, force=False)\nseleccion"),
        code(
            "import mlflow, mlflow.sklearn\n"
            "from sklearn.metrics import average_precision_score\n"
            "configurar_mlflow()\n"
            "modelo=mlflow.sklearn.load_model(f\"models:/{seleccion['model_registry_name']}@champion\")\n"
            "score=modelo.predict_proba(data.test[data.feature_columns])[:,1]\n"
            "reproducida=average_precision_score(data.test[data.target_column],score)\n"
            "assert abs(reproducida-seleccion['champion_test_pr_auc'])<1e-12\n"
            "print(f\"PR-AUC reproducida desde @champion: {reproducida:.6f}\")"
        ),
        code("import pandas as pd\npd.read_csv(ROOT/'docs'/'finalistas_test.csv')"),
        code("pd.read_csv(ROOT/'docs'/'pruebas_estacionales.csv').query(\"alcance == 'backtests_oof'\")"),
        markdown(
            "También se entrena una sensibilidad t+7 con la familia ganadora. La selección principal queda "
            "en t+3 porque es el horizonte operativo prioritario del PSet 1; t+7 se registra por separado "
            "para medir la degradación."
        ),
    ],
)

print(f"Notebooks creados en {NOTEBOOKS}")
