"""Crea el memo técnico de seis páginas para el PSet 3."""

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PLOTS = DOCS / "plots"
OUTPUT = DOCS / "PSet3_memo_Tierra_Rodriguez_Rodriguez_Cadena_Jarrin.pdf"

NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#2A6F97")
TEAL = colors.HexColor("#2A9D8F")
LIGHT = colors.HexColor("#EAF2F8")
GRAY = colors.HexColor("#5D6D7E")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=24, leading=29, textColor=NAVY, alignment=TA_CENTER, spaceAfter=14))
styles.add(ParagraphStyle(name="CoverSub", parent=styles["Normal"], fontSize=12, leading=17, textColor=GRAY, alignment=TA_CENTER))
styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=16, leading=19, textColor=NAVY, spaceAfter=8))
styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, leading=14, textColor=BLUE, spaceBefore=5, spaceAfter=4))
styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.15, leading=12.2, alignment=TA_JUSTIFY, spaceAfter=5))
styles.add(ParagraphStyle(name="BodyLeftx", parent=styles["Bodyx"], alignment=0))
styles.add(ParagraphStyle(name="Smallx", parent=styles["BodyText"], fontSize=7.8, leading=9.5, textColor=GRAY))
styles.add(ParagraphStyle(name="Callout", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=10.2, leading=13.2, textColor=NAVY, leftIndent=8, rightIndent=8, spaceBefore=6, spaceAfter=6))


def p(text: str, style: str = "Bodyx") -> Paragraph:
    return Paragraph(text, styles[style])


def tabla(data, widths, header=True, font_size=8.2) -> Table:
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    comandos = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B8C6D1")),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, colors.HexColor("#F5F8FA")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        comandos += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    t.setStyle(TableStyle(comandos))
    return t


def imagen(nombre: str, ancho: float, alto: float) -> Image:
    return Image(str(PLOTS / nombre), width=ancho, height=alto)


def encabezado_pie(canvas, doc) -> None:
    canvas.saveState()
    if doc.page > 1:
        canvas.setStrokeColor(colors.HexColor("#D5DEE5"))
        canvas.line(1.7 * cm, 1.35 * cm, 19.3 * cm, 1.35 * cm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(GRAY)
        canvas.drawString(1.7 * cm, 0.95 * cm, "PSet 3 · Predicción de alta demanda ECU 911")
        canvas.drawRightString(19.3 * cm, 0.95 * cm, f"Página {doc.page} de 6")
    canvas.restoreState()


def crear() -> None:
    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4,
        rightMargin=1.7 * cm, leftMargin=1.7 * cm,
        topMargin=1.45 * cm, bottomMargin=1.55 * cm,
        title="PSet 3 - Experimentación y selección de modelo ECU 911",
        author="Tierra, Rodríguez, Rodríguez, Cadena y Jarrín",
    )
    story = []
    candidatos = pd.read_csv(DOCS / "candidatos.csv").set_index("modelo")
    nombres_modelos = {
        "dummy": "Dummy (prevalencia)",
        "regla_estacional": "Regla estacional",
        "logistica": "Regresión logística",
        "random_forest": "Random Forest",
        "hist_gradient_boosting": "HistGradientBoosting",
        "mlp": "MLP",
        "catboost": "CatBoost",
    }
    learning_rates = {
        "dummy": "N/A",
        "regla_estacional": "N/A",
        "logistica": "N/A",
        "random_forest": "N/A",
        "hist_gradient_boosting": "0,08",
        "mlp": "0,001",
        "catboost": "0,08",
    }
    orden_modelos = [
        "dummy",
        "regla_estacional",
        "logistica",
        "random_forest",
        "hist_gradient_boosting",
        "mlp",
        "catboost",
    ]

    def coma(valor: float, decimales: int = 3) -> str:
        return f"{valor:.{decimales}f}".replace(".", ",")

    tabla_candidatos = [["Modelo", "Train", "CV ± desv.", "Brecha", "LR", "Fit / CV (s)"]]
    for modelo in orden_modelos:
        fila = candidatos.loc[modelo]
        tabla_candidatos.append([
            nombres_modelos[modelo],
            coma(fila["train_pr_auc"]),
            f"{coma(fila['cv_pr_auc_mean'])} ± {coma(fila['cv_pr_auc_std'])}",
            coma(fila["generalization_gap"]),
            learning_rates[modelo],
            f"{coma(fila['train_fit_seconds'], 1)} / {coma(fila['cv_fit_seconds'], 1)}",
        ])

    # 1. Portada y resumen ejecutivo
    story += [Spacer(1, 1.7 * cm), p("PSet 3", "CoverTitle"), p("Experimentación y selección de un modelo para anticipar alta demanda de emergencias", "CoverSub"), Spacer(1, 1.0 * cm)]
    story.append(tabla([
        [p("Equipo", "Smallx"), p("Jarod Tierra · Felipe Rodríguez · Santiago Rodríguez · Juan Diego Cadena · Miguel Jarrín", "Smallx")],
        [p("Materia", "Smallx"), p("Pipelines de Ciencia de Datos", "Smallx")],
        [p("Fecha", "Smallx"), p("16 de noviembre de 2026", "Smallx")],
        [p("Caso", "Smallx"), p("ECU 911: alerta de demanda inusualmente alta por cantón", "Smallx")],
    ], [3.2 * cm, 12.6 * cm], header=False, font_size=8.3))
    story += [Spacer(1, 0.7 * cm), p("Resumen ejecutivo", "H1x")]
    story.append(p("Construimos un experimento reproducible para predecir, con tres días de anticipación, si el número de emergencias de un cantón superará su percentil 90 histórico para ese día de la semana. Trabajamos con la OBT local completa: 429.632 observaciones, 224 cantones y datos desde julio de 2021 hasta septiembre de 2026. Dejamos la lectura desde Snowflake preparada, pero no fue necesaria para estas pruebas."))
    story.append(p("La comparación incluyó dos baselines y cinco familias de modelos. Los tres candidatos estables recibieron 30 configuraciones cada uno. HistGradientBoosting quedó como <i>champion</i> con PR-AUC 0,2693 en holdout y redujo 36,3% los picos omitidos respecto a la regla estacional. La validación usa cuatro semestres completos y pruebas específicas de feriados, diciembre y fin de año."))
    story.append(Table([[p("Resultado principal: 36,3% menos picos omitidos y PR-AUC medio 0,5405 en feriados de los cuatro backtests.", "Callout")]], colWidths=[15.8 * cm], style=TableStyle([("BACKGROUND", (0,0), (-1,-1), LIGHT), ("BOX", (0,0), (-1,-1), 0.8, BLUE), ("VALIGN", (0,0), (-1,-1), "MIDDLE")])))
    story.append(PageBreak())

    # 2. Problema y EDA
    story += [p("1. Problema y exploración de datos", "H1x"), p("Objetivo y unidad de análisis", "H2x")]
    story.append(p("La unidad de análisis es cantón-día. En el día <i>t</i>, el sistema estima si en <i>t+3</i> habrá demanda alta. El target vale 1 cuando el conteo futuro supera el P90 del cantón y día de semana, calculado solo con el periodo de entrenamiento. Así se respeta la estacionalidad local y se evita que información del holdout cambie la definición del problema."))
    story.append(tabla([
        ["Periodo", "Filas", "Uso"],
        ["2021-07-01 a 2026-03-28", "387.968", "Entrenamiento purgado + backtests"],
        ["2026-04-01 a 2026-09-27", "40.320", "Holdout final (una sola vez)"],
        ["Total OBT", "429.632", "224 cantones, 37 columnas"],
    ], [5.1 * cm, 3.4 * cm, 7.3 * cm]))
    story += [p("Hallazgos del EDA", "H2x")]
    story.append(p("Usamos correlación de Spearman porque el conteo tiene cola larga y valores atípicos; entre las variables disponibles, <font name='Courier'>ratio_media_7_p90</font> mostró la asociación monotónica más alta con la clase (0,106)."))
    story.append(tabla([
        ["Hallazgo", "Evidencia", "Consecuencia"],
        ["Desbalance", "8,17% de positivos", "PR-AUC y costo 15×FN + FP; no accuracy sola"],
        ["Picos reales", "12,66% fuera de IQR; 0,93% con |z|≥3", "No eliminarlos; probar árboles y métricas robustas"],
        ["Estacionalidad", p("Carnaval 40,4%; fin de año 12,6%; regular 7,6%", "Smallx"), p("Calendario, backtests semestrales y ponderación", "Smallx")],
    ], [3.0 * cm, 5.5 * cm, 7.3 * cm], font_size=6.6))
    story.append(KeepTogether([imagen("01_target_distribucion.png", 7.65 * cm, 3.55 * cm), Spacer(1, 0.05 * cm)]))
    story.append(KeepTogether([imagen("08_eda_feriados.png", 15.3 * cm, 4.7 * cm)]))
    story.append(p("Figura 1. Desbalance del target y prevalencia por temporada y feriado. Los gráficos completos se guardan en <font name='Courier'>docs/plots</font>.", "Smallx"))
    story.append(PageBreak())

    # 3. Features y validación
    story += [p("2. Features, pipeline y validación", "H1x"), p("Variables construidas", "H2x")]
    story.append(p("Las variables se calculan antes del día de predicción y nunca usan información desconocida. Además de historia reciente, agregamos rezagos 364/365/371, ciclos seno/coseno, mes y día de la fecha objetivo, nombre y distancia al feriado, ventana ±2 días, fin de año, Carnaval, Semana Santa e interacción con fiestas locales. El calendario es conocido al momento de inferencia."))
    story.append(tabla([
        ["Grupo", "Ejemplos", "Razón"],
        ["Calendario", "día de semana, mes, feriado", "Captura patrones regulares"],
        ["Historia", "lag 1/7/14/21/28 y 364/365/371", "Compara corto plazo y año anterior"],
        ["Tendencia", "media 7 y 28, cambio 1 y 7", "Detecta aumentos recientes"],
        ["Normalización", "ratio lag/P90 y media/P90", "Compara cantones en su propia escala"],
        ["Calendario objetivo", "feriado, ventana, temporada, ciclos", "Representa eventos conocidos"],
    ], [3.0 * cm, 5.6 * cm, 7.2 * cm], font_size=7.8))
    story += [p("Pipeline reproducible", "H2x")]
    story.append(p("Todas las familias comparten un <i>ColumnTransformer</i>: las variables numéricas se imputan y escalan; las categóricas se imputan y codifican dentro del pipeline. El estimador es el último paso. Esto evita que el preprocesamiento aprenda de validación o holdout. Las columnas <font name='Courier'>obj_*</font>, targets desplazados y cualquier variable futura quedan excluidas por contrato y por pruebas unitarias."))
    story += [p("Esquema de validación", "H2x")]
    story.append(p("Se fijó un holdout de seis meses antes de comparar modelos. Dentro del entrenamiento se usaron cuatro backtests expansivos: enero-junio y julio-diciembre de 2024 y 2025. Antes de cada bloque se purgan tres días por el horizonte t+3. La selección usa PR-AUC ponderado: peso 2 para ventanas de feriado y 2,5 para fin de año, sin duplicar filas."))
    story.append(tabla([
        ["Decisión", "Aplicación"],
        ["Métrica técnica", "PR-AUC general, ponderado, peor semestre y desviación"],
        ["Costo operativo", "15 × falsos negativos + falsos positivos"],
        ["Meta de negocio", "≥10% menos picos omitidos que la regla estacional"],
        ["Umbral", "Elegido con predicciones OOF; nunca con el holdout"],
        ["Pruebas especiales", "Feriados, diciembre, fin de año, Carnaval y días regulares"],
    ], [4.5 * cm, 11.3 * cm]))
    story.append(p("Se añadieron siete pruebas automatizadas: regla de negocio, calendario conocido, disponibilidad temporal, exclusión de columnas futuras, purga del holdout, ajuste del pipeline y ausencia de solapamiento."))
    story += [p("Baselines y comparación inicial", "H2x")]
    story.append(p("Comparamos dos baselines y cinco aproximaciones con el mismo pipeline y los mismos cuatro backtests. La tabla usa PR-AUC general para hacer visible el sobreajuste; LR es la tasa de aprendizaje por defecto y no aplica a reglas, modelos lineales ni bagging."))
    story.append(tabla(
        tabla_candidatos,
        [3.6 * cm, 1.55 * cm, 3.0 * cm, 1.65 * cm, 1.45 * cm, 2.55 * cm],
        font_size=6.5,
    ))
    story.append(PageBreak())

    # 4. Experimentos
    story += [p("3. Diseño experimental y tuning", "H1x")]
    story.append(p("Evaluamos regresión logística, Random Forest, HistGradientBoosting, MLP y CatBoost. Random Forest no pasó al tuning por una brecha de generalización de 0,268. No usamos un LLM por prompting porque el caso exige probabilidades tabulares reproducibles y no era válido enviar la OBT a un servicio externo."))
    story.append(imagen("05_comparacion_candidatos.png", 15.3 * cm, 5.0 * cm))
    story.append(p("Figura 2. PR-AUC ponderado medio y desviación temporal. HistGradientBoosting lideró; Random Forest mostró una brecha de generalización de 0,268.", "Smallx"))
    story += [p("Tuning controlado", "H2x")]
    story.append(p("ParameterSampler evaluó 30 configuraciones por familia con semilla 42. Cada configuración fue una corrida hija en MLflow, usó los mismos cuatro semestres y se entrenó secuencialmente con dos hilos."))
    story.append(imagen("06_tuning_90_configuraciones.png", 15.3 * cm, 4.7 * cm))
    story.append(p("Figura 3. Las 90 configuraciones se compararon con PR-AUC temporal; la selección final usó el PR-AUC ponderado por temporadas críticas.", "Smallx"))
    story += [p("Espacios de búsqueda", "H2x")]
    story.append(tabla([
        ["Familia", "Parámetros y rangos", "Escala"],
        ["HistGradientBoosting", "LR 0,01-0,224; iter. 60-160; hojas 15-63; min. hoja 15-100; L2 0,001-10", "LR y L2 log"],
        ["CatBoost", "LR 0,01-0,316; iter. 60-160; profundidad 4-8; L2 0,1-19,95; fuerza 0-2", "LR y L2 log"],
        ["MLP", "LR 0,0001-0,01; alpha 1e-5-0,1; capas 16 a 48/24; batch 256-1024", "LR y alpha log"],
    ], [3.0 * cm, 10.1 * cm, 2.7 * cm], font_size=6.1))
    story.append(tabla([
        ["Familia", "Base", "Mejor", "Mejora", "Supera desv."],
        ["HistGradientBoosting", "0,2267", "0,2316 ± 0,0603", "+0,0048", "No"],
        ["CatBoost", "0,2247", "0,2276 ± 0,0578", "+0,0029", "No"],
        ["MLP", "0,2065", "0,2113 ± 0,0417", "+0,0048", "No"],
    ], [4.2 * cm, 2.1 * cm, 3.8 * cm, 2.3 * cm, 3.4 * cm], font_size=6.4))
    story.append(p("Ninguna mejora superó una desviación entre pliegues; se conservó esa incertidumbre en la decisión.", "Smallx"))
    story.append(PageBreak())

    # 5. Resultado y MLflow
    story += [p("4. Resultado final y trazabilidad", "H1x")]
    story.append(tabla([
        ["Rol", "Modelo", "Umbral", "PR-AUC", "Precisión", "Recall", "FN", "Reducción FN"],
        ["Champion", "HistGradientBoosting", "0,35", "0,2693", "0,2249", "0,4838", "1.645", "36,3%"],
        ["Challenger", "CatBoost", "0,52", "0,2394", "0,2323", "0,4449", "1.769", "31,5%"],
        ["Baseline", "Regla estacional", "—", "0,1209", "0,1381", "0,1901", "2.581", "—"],
    ], [2.2 * cm, 3.5 * cm, 1.7 * cm, 1.9 * cm, 2.0 * cm, 1.7 * cm, 1.4 * cm, 2.3 * cm], font_size=7.3))
    story.append(p("La ventaja ponderada de HistGradientBoosting sobre CatBoost fue 0,0039, menor que la desviación de ambos; por eso también decidimos con el holdout y el costo. Al unir las predicciones OOF obtuvo PR-AUC 0,5499 en feriados, 0,5285 en fin de año y 0,4682 en Carnaval/Semana Santa. En 40.320 filas, ajuste/inferencia fueron 21,30/0,576 s frente a 14,78/0,572 s: operativamente quedaron casi empatados."))
    story.append(imagen("07_resultado_holdout.png", 15.5 * cm, 7.3 * cm))
    story.append(p("Figura 4. Resultado técnico y operativo en el holdout reservado.", "Smallx"))
    story += [p("Seguimiento con MLflow", "H2x")]
    story.append(p("El experimento contiene 105 corridas y 90 configuraciones válidas. El champion es la versión 3 (run_id <font name='Courier'>b94b74256a114988b926217055c66134</font>) y el challenger la versión 4. Al recargar el alias <font name='Courier'>champion</font> se reprodujo PR-AUC 0,2692505238 exactamente.", "BodyLeftx"))
    story.append(Image(str(DOCS / "evidencia" / "mlflow_comparacion.png"), width=15.5 * cm, height=9.0 * cm))
    story.append(p("Figura 5. Evidencia de corridas y jerarquía de tuning en MLflow 2.17.2.", "Smallx"))
    story.append(PageBreak())

    # 6. Decisión, limitaciones y reproducción
    story += [p("5. Decisión, sensibilidad y limitaciones", "H1x"), p("Decisión recomendada", "H2x")]
    story.append(p("Recomendamos HistGradientBoosting con horizonte t+3 y umbral 0,35. Combina estabilidad, desempeño en temporadas críticas, costo de errores y 36,3% menos picos omitidos. CatBoost se conserva como challenger. La sensibilidad t+7 obtuvo PR-AUC 0,2634, precisión 0,3083 y recall 0,3249; es viable, pero t+3 ofrece una reacción más cercana a la operación."))
    story += [p("Limitaciones", "H2x")]
    story.append(p("Primero, la etiqueta depende de la calidad del conteo histórico y no mide severidad ni capacidad disponible. Segundo, cuatro backtests mejoran la evidencia, pero una puesta en producción requiere monitoreo por cantón y nuevos periodos. Tercero, la OBT identifica feriados nacionales y solo algunas fiestas locales; conviene ampliar ese calendario. Cuarto, el costo 15:1 debe validarse con personal operativo. Quinto, ambos finalistas necesitan explicaciones posprocesadas (permutación o SHAP); su salida no debe interpretarse como causal."))
    story.append(p("La señal es mucho mayor en feriados que en días regulares, pero el holdout disponible termina en septiembre y no contiene Navidad. Por eso los backtests de 2024-2025 son evidencia estacional y el próximo diciembre debe tratarse como una prueba futura adicional."))
    story += [p("Reproducibilidad y conexión futura", "H2x")]
    story.append(tabla([
        ["Elemento", "Evidencia"],
        ["Datos", "SHA-256: 4f2c2c024d1e…f323e8137a"],
        ["Código", "5 notebooks, src/datos.py, src/pipeline.py y src/experimentos.py"],
        ["Resultados", "docs/corridas.csv y artefactos de MLflow"],
        ["Registry", "ecu911-alta-demanda-modelo@champion / @challenger"],
        ["Snowflake", "DATA_SOURCE=snowflake; mismo contrato de columnas"],
    ], [3.7 * cm, 12.1 * cm], font_size=7.9))
    story.append(p("Para pasar a Snowflake no se deben reescribir los notebooks. Se carga la misma OBT, se configuran credenciales en un archivo <font name='Courier'>.env</font> ignorado por Git y se cambia la fuente. El loader normaliza nombres y valida el contrato antes de entrenar. Así, las pruebas locales sirven como ensayo completo y la cuenta nueva solo reemplaza el origen de datos."))
    story += [p("Conclusión", "H2x")]
    story.append(p("El PSet deja un flujo completo: EDA estacional, features sin fuga, cuatro backtests purgados, 90 configuraciones, pruebas especiales, evaluación final, selección de umbral, registro y reproducción. El resultado mejora la evidencia técnica y cumple la meta de negocio sin depender de Snowflake para la demostración local."))
    story += [Spacer(1, 0.25 * cm), p("Referencias: documentación oficial de scikit-learn (Pipeline y métricas), MLflow Model Registry y CatBoost. Consultadas para la implementación reproducible del ejercicio.", "Smallx")]

    doc.build(story, onFirstPage=encabezado_pie, onLaterPages=encabezado_pie)
    print(f"Memo creado en {OUTPUT}")


if __name__ == "__main__":
    crear()
