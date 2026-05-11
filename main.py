"""
main.py — Punto de entrada para ejecución en modo consola (sin interfaz).
Útil para entornos de servidor, scripts batch o CI/CD.

Uso:
    python main.py --file data/qos_datos.xlsx --target calidad_red
    python main.py --generate-sample
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuración de logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Importaciones del proyecto
# ---------------------------------------------------------------------------
from app.data_loader import load_file, validate_dataframe, detect_target_column
from app.preprocessor import QoSPreprocessor
from app.models import QoSModelTrainer
from app.visualizations import generate_all_plots
from app.exporter import (
    export_predictions_to_excel,
    export_metrics_report,
    export_classification_report_txt,
    package_results_zip,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QoS ML Analyzer — Predicción de calidad en redes móviles"
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Ruta al archivo Excel (.xlsx) o CSV con datos QoS."
    )
    parser.add_argument(
        "--target", type=str, default=None,
        help="Nombre de la columna objetivo. Si se omite, se detecta automáticamente."
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fracción del dataset para test (default: 0.2)."
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="Semilla aleatoria (default: 42)."
    )
    parser.add_argument(
        "--k-features", type=int, default=10,
        help="Número de características a seleccionar con SelectKBest (0=todas)."
    )
    parser.add_argument(
        "--generate-sample", action="store_true",
        help="Genera un dataset de ejemplo y lo guarda en data/."
    )
    return parser.parse_args()


def run_pipeline(
    filepath: str,
    target_col: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    k_features: int = 10,
) -> None:
    """
    Ejecuta el pipeline completo de ML en modo consola.

    Args:
        filepath: Ruta al dataset
        target_col: Columna objetivo (None = auto-detect)
        test_size: Fracción de test
        random_state: Semilla aleatoria
        k_features: Número de features para SelectKBest
    """
    # ── 1. Carga y validación ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PASO 1: Carga de datos")
    df = load_file(filepath)

    validation = validate_dataframe(df)
    logger.info("Filas: %d | Columnas: %d | Duplicados: %d",
                validation["total_rows"], validation["total_columns"],
                validation["duplicate_rows"])

    if not validation["is_valid"]:
        for e in validation["errors"]:
            logger.error("VALIDACIÓN: %s", e)
        sys.exit(1)

    for w in validation["warnings"]:
        logger.warning("VALIDACIÓN: %s", w)

    # ── 2. Detección de columna objetivo ───────────────────────────────────
    if target_col is None:
        target_col = detect_target_column(df)
        if target_col is None:
            logger.error("No se pudo detectar columna objetivo. Usa --target <nombre>.")
            sys.exit(1)
    logger.info("Columna objetivo: '%s'", target_col)

    # ── 3. Preprocesamiento ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PASO 2: Preprocesamiento")
    preprocessor = QoSPreprocessor(
        test_size=test_size,
        random_state=random_state,
        k_best_features=k_features,
    )
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df, target_col)
    logger.info("Clases: %s", preprocessor.class_names)
    logger.info("Features seleccionadas: %s", preprocessor.selected_features)

    # ── 4. Entrenamiento ───────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PASO 3: Entrenamiento de modelos")
    trainer = QoSModelTrainer(models_dir="models", random_state=random_state)
    results = trainer.train_all(
        X_train, X_test, y_train, y_test,
        class_names=preprocessor.class_names,
    )

    # Tabla comparativa
    comp_df = trainer.get_comparison_dataframe()
    logger.info("\n%s", comp_df.to_string(index=False))
    logger.info("Mejor modelo: %s", trainer.best_model_name)

    # ── 5. Visualizaciones ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PASO 4: Generando visualizaciones")
    best_metrics = results[trainer.best_model_name]
    importances_df = trainer.get_feature_importances(preprocessor.selected_features)

    graph_paths = generate_all_plots(
        df=df,
        target_col=target_col,
        y_test=y_test,
        class_names=preprocessor.class_names,
        confusion_mat=best_metrics["confusion_matrix"],
        best_model_name=trainer.best_model_name,
        importances_df=importances_df,
        comparison_df=comp_df,
        graphs_dir="graphs",
    )

    # ── 6. Exportación ─────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PASO 5: Exportando resultados")
    os.makedirs("reports", exist_ok=True)

    y_pred = trainer.predict(X_test)
    y_pred_labels = preprocessor.decode_labels(y_pred)
    y_true_labels = preprocessor.decode_labels(y_test)
    proba = trainer.predict_proba(X_test)

    feat_cols = preprocessor.selected_features
    X_test_df = df.drop(columns=[target_col]).iloc[: len(X_test)][feat_cols]

    pred_path = export_predictions_to_excel(
        original_df=X_test_df.reset_index(drop=True),
        y_pred_labels=y_pred_labels,
        y_true_labels=y_true_labels,
        proba_matrix=proba,
        class_names=preprocessor.class_names,
        output_path="reports/predicciones_qos.xlsx",
    )

    metrics_path = export_metrics_report(
        model_name=trainer.best_model_name,
        metrics=best_metrics,
        comparison_df=comp_df,
        class_names=preprocessor.class_names,
        preprocessing_summary=preprocessor.get_preprocessing_summary(),
        output_path="reports/reporte_metricas.xlsx",
    )

    txt_path = export_classification_report_txt(
        classification_report_str=best_metrics["classification_report"],
        model_name=trainer.best_model_name,
        output_path="reports/classification_report.txt",
    )

    model_pkl = trainer.save_best_model("best_model.pkl")

    zip_path = package_results_zip(
        files_to_include=list(graph_paths.values()) + [pred_path, metrics_path, txt_path, model_pkl],
        output_zip="reports/resultados_qos.zip",
    )

    logger.info("=" * 60)
    logger.info("✅ Pipeline completado con éxito.")
    logger.info("  Predicciones  : %s", pred_path)
    logger.info("  Métricas      : %s", metrics_path)
    logger.info("  Modelo        : %s", model_pkl)
    logger.info("  ZIP           : %s", zip_path)


def main() -> None:
    args = parse_args()

    if args.generate_sample:
        from data.sample_qos_dataset import generate_qos_dataset
        os.makedirs("data", exist_ok=True)
        df = generate_qos_dataset(1000)
        out = "data/qos_datos_ejemplo.xlsx"
        df.to_excel(out, index=False)
        logger.info("Dataset de ejemplo generado: %s", out)
        return

    if args.file is None:
        logger.error("Debes indicar un archivo con --file o usar --generate-sample.")
        sys.exit(1)

    run_pipeline(
        filepath=args.file,
        target_col=args.target,
        test_size=args.test_size,
        random_state=args.random_state,
        k_features=args.k_features,
    )


if __name__ == "__main__":
    main()
