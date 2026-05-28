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


# Exportación de predicciones a Excel

def export_predictions_to_excel(
    original_df: pd.DataFrame,
    y_pred_labels: np.ndarray,
    y_true_labels: Optional[np.ndarray],
    proba_matrix: Optional[np.ndarray],
    class_names: List[str],
    output_path: str = "reports/predicciones_qos.xlsx",
) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df_out = original_df.copy().reset_index(drop=True)


    df_out["prediccion_calidad"] = y_pred_labels


    if y_true_labels is not None:
        df_out["calidad_real"] = y_true_labels
        df_out["correcto"] = (
            pd.Series(y_pred_labels).values == pd.Series(y_true_labels).values
        )


    if proba_matrix is not None:
        for i, cls in enumerate(class_names):
            df_out[f"prob_{cls}"] = np.round(proba_matrix[:, i], 4)

    df_out["timestamp_prediccion"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_out.to_excel(writer, index=False, sheet_name="Predicciones")


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



def export_metrics_report(
    model_name: str,
    metrics: Dict[str, Any],
    comparison_df: pd.DataFrame,
    class_names: List[str],
    preprocessing_summary: Dict[str, Any],
    output_path: str = "reports/reporte_metricas.xlsx",
) -> str:

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:


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


        comparison_df.to_excel(
            writer, index=False, sheet_name="Comparacion_Modelos"
        )


        config_rows = [
            {"Parámetro": k, "Valor": str(v)}
            for k, v in preprocessing_summary.items()
        ]
        pd.DataFrame(config_rows).to_excel(
            writer, index=False, sheet_name="Configuracion"
        )

    logger.info("Reporte de métricas exportado: %s", output_path)
    return output_path



def export_classification_report_txt(
    classification_report_str: str,
    model_name: str,
    output_path: str = "reports/classification_report.txt",
) -> str:

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



def package_results_zip(
    files_to_include: List[str],
    output_zip: str = "reports/resultados_qos.zip",
) -> str:

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
