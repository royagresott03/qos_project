"""
Módulo de exportación de resultados QoS.
Genera archivos Excel con predicciones, reportes de métricas y empaqueta
todos los artefactos del análisis.
"""

import os
import json
import pickle
import zipfile
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exportación de predicciones a Excel
# ---------------------------------------------------------------------------

def export_predictions_to_excel(
    original_df: pd.DataFrame,
    y_pred_labels: np.ndarray,
    y_true_labels: Optional[np.ndarray],
    proba_matrix: Optional[np.ndarray],
    class_names: List[str],
    output_path: str = "reports/predicciones_qos.xlsx",
) -> str:
    """
    Exporta el dataset original enriquecido con predicciones del modelo.

    Columnas añadidas:
        - prediccion_calidad: Clase predicha en texto
        - correcto: Si la predicción coincide con el valor real (si disponible)
        - prob_<clase>: Probabilidad de cada clase (si el modelo las provee)

    Args:
        original_df: DataFrame original (test set con features)
        y_pred_labels: Predicciones del modelo en texto
        y_true_labels: Etiquetas reales en texto (puede ser None)
        proba_matrix: Matriz de probabilidades (puede ser None)
        class_names: Lista de nombres de clases
        output_path: Ruta del archivo Excel de salida

    Returns:
        Ruta del archivo generado
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df_out = original_df.copy().reset_index(drop=True)

    # Columna de predicción
    df_out["prediccion_calidad"] = y_pred_labels

    # Columna de corrección (si hay etiquetas reales)
    if y_true_labels is not None:
        df_out["calidad_real"] = y_true_labels
        df_out["correcto"] = (
            pd.Series(y_pred_labels).values == pd.Series(y_true_labels).values
        )

    # Probabilidades por clase
    if proba_matrix is not None:
        for i, cls in enumerate(class_names):
            df_out[f"prob_{cls}"] = np.round(proba_matrix[:, i], 4)

    # Columna de timestamp
    df_out["timestamp_prediccion"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_out.to_excel(writer, index=False, sheet_name="Predicciones")

        # Hoja de resumen
        summary = pd.DataFrame({
            "Métrica": ["Total muestras", "Clases disponibles", "Fecha"],
            "Valor": [
                len(df_out),
                ", ".join(class_names),
                datetime.now().strftime("%Y-%m-%d %H:%M"),
            ],
        })
        summary.to_excel(writer, index=False, sheet_name="Resumen")

    logger.info("Predicciones exportadas: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Exportación del reporte de métricas
# ---------------------------------------------------------------------------

def export_metrics_report(
    model_name: str,
    metrics: Dict[str, Any],
    comparison_df: pd.DataFrame,
    class_names: List[str],
    preprocessing_summary: Dict[str, Any],
    output_path: str = "reports/reporte_metricas.xlsx",
) -> str:
    """
    Genera un reporte en Excel con métricas del modelo y comparación.

    Hojas generadas:
        1. Métricas_Mejor_Modelo  – accuracy, precision, recall, f1
        2. Comparacion_Modelos    – tabla comparativa de todos los modelos
        3. Configuracion          – parámetros de preprocesamiento

    Args:
        model_name: Nombre del mejor modelo
        metrics: Diccionario con métricas del mejor modelo
        comparison_df: DataFrame comparativo de todos los modelos
        class_names: Nombres de las clases
        preprocessing_summary: Parámetros del preprocesador
        output_path: Ruta del archivo de salida

    Returns:
        Ruta del archivo generado
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:

        # Hoja 1: Métricas del mejor modelo
        metrics_data = {
            "Métrica": ["Modelo", "Accuracy", "Precision", "Recall", "F1-Score",
                        "Tiempo Entrenamiento (s)", "Clases"],
            "Valor": [
                model_name,
                metrics.get("accuracy", "N/A"),
                metrics.get("precision", "N/A"),
                metrics.get("recall", "N/A"),
                metrics.get("f1_score", "N/A"),
                metrics.get("train_time_sec", "N/A"),
                ", ".join(class_names),
            ],
        }
        pd.DataFrame(metrics_data).to_excel(
            writer, index=False, sheet_name="Metricas_Mejor_Modelo"
        )

        # Hoja 2: Comparación de modelos
        comparison_df.to_excel(
            writer, index=False, sheet_name="Comparacion_Modelos"
        )

        # Hoja 3: Configuración de preprocesamiento
        config_rows = [
            {"Parámetro": k, "Valor": str(v)}
            for k, v in preprocessing_summary.items()
        ]
        pd.DataFrame(config_rows).to_excel(
            writer, index=False, sheet_name="Configuracion"
        )

    logger.info("Reporte de métricas exportado: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Exportar reporte en texto / JSON
# ---------------------------------------------------------------------------

def export_classification_report_txt(
    classification_report_str: str,
    model_name: str,
    output_path: str = "reports/classification_report.txt",
) -> str:
    """
    Guarda el reporte de clasificación completo en un archivo de texto.

    Args:
        classification_report_str: Salida de sklearn.metrics.classification_report
        model_name: Nombre del modelo
        output_path: Ruta del archivo de salida

    Returns:
        Ruta del archivo generado
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    header = (
        f"REPORTE DE CLASIFICACIÓN QoS — {model_name}\n"
        f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{'=' * 60}\n\n"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header + classification_report_str)

    logger.info("Reporte de texto guardado: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Empaquetar todos los artefactos en un ZIP
# ---------------------------------------------------------------------------

def package_results_zip(
    files_to_include: List[str],
    output_zip: str = "reports/resultados_qos.zip",
) -> str:
    """
    Empaqueta todos los artefactos del análisis en un único archivo ZIP.

    Args:
        files_to_include: Lista de rutas de archivos a incluir
        output_zip: Ruta del archivo ZIP de salida

    Returns:
        Ruta del ZIP generado
    """
    os.makedirs(os.path.dirname(output_zip) or ".", exist_ok=True)

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for filepath in files_to_include:
            if os.path.isfile(filepath):
                arcname = os.path.basename(filepath)
                zf.write(filepath, arcname)
                logger.info("  + %s", arcname)
            else:
                logger.warning("Archivo no encontrado, omitido: %s", filepath)

    logger.info("ZIP generado: %s (%d archivos)", output_zip, len(zf.namelist()))
    return output_zip
