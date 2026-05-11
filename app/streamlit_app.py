"""
Interfaz gráfica Streamlit para el sistema de predicción de calidad QoS
en redes móviles mediante Machine Learning.

Ejecutar con:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import io
import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Asegurar que el paquete raíz esté en el path
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.data_loader import (
    load_file, validate_dataframe, detect_target_column,
    get_feature_columns, preview_dataframe,
)
from app.preprocessor import QoSPreprocessor
from app.models import QoSModelTrainer
from app.visualizations import (
    generate_all_plots,
    plot_correlation_heatmap,
    plot_class_distribution,
    plot_feature_histograms,
    plot_feature_importances,
    plot_confusion_matrix,
    plot_model_comparison,
)
from app.exporter import (
    export_predictions_to_excel,
    export_metrics_report,
    export_classification_report_txt,
    package_results_zip,
)

# ---------------------------------------------------------------------------
# Configuración de logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("streamlit_app")

# ---------------------------------------------------------------------------
# Directorios de trabajo
# ---------------------------------------------------------------------------
GRAPHS_DIR = str(ROOT / "graphs")
MODELS_DIR = str(ROOT / "models")
REPORTS_DIR = str(ROOT / "reports")
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Configuración de la página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="QoS ML Analyzer",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS personalizado
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Fondo y fuentes */
    .stApp { background-color: #F0F4F8; }
    .block-container { padding-top: 1.5rem; }

    /* Tarjetas de métricas */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 0.75rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E3A5F;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Banner superior */
    .hero-banner {
        background: linear-gradient(135deg, #1E3A5F 0%, #2E6DA4 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .hero-banner h1 { margin: 0; font-size: 1.8rem; }
    .hero-banner p  { margin: 0.3rem 0 0; opacity: 0.85; }

    /* Botón principal */
    div.stButton > button:first-child {
        background: #2E6DA4;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
    }
    div.stButton > button:first-child:hover {
        background: #1E3A5F;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }

    /* Separadores de sección */
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1E3A5F;
        border-left: 4px solid #2E6DA4;
        padding-left: 0.75rem;
        margin: 1.2rem 0 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers de UI
# ---------------------------------------------------------------------------

def metric_card(label: str, value: str, delta: str = "") -> None:
    delta_html = f"<div style='color:#10B981;font-size:.85rem;'>{delta}</div>" if delta else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title: str) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def show_image(path: str, caption: str = "") -> None:
    """Muestra una imagen si el archivo existe."""
    if path and os.path.isfile(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.info(f"Gráfica no disponible: {caption}")


# ---------------------------------------------------------------------------
# Banner principal
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero-banner">
        <h1>📡 QoS ML Analyzer</h1>
        <p>Sistema de predicción de calidad de servicio en redes móviles
        mediante Inteligencia Artificial</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — configuración del análisis
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Antu_network-wireless.svg/240px-Antu_network-wireless.svg.png",
        width=60,
    )
    st.markdown("## ⚙️ Configuración")

    test_size = st.slider(
        "Tamaño del conjunto de test",
        min_value=0.10, max_value=0.40, value=0.20, step=0.05,
        help="Fracción del dataset reservada para evaluación.",
    )
    random_state = st.number_input(
        "Semilla aleatoria", min_value=0, max_value=9999,
        value=42, step=1,
        help="Garantiza reproducibilidad de los resultados.",
    )
    remove_outliers = st.checkbox("Eliminar outliers extremos (z>3)", value=True)

    st.markdown("---")
    st.markdown("### 📊 Selección de características")
    k_features = st.slider(
        "Número de features (0 = todas)",
        min_value=0, max_value=20, value=10,
        help="SelectKBest con prueba F de ANOVA.",
    )

    st.markdown("---")
    st.markdown(
        "**Versión:** 1.0.0  \n"
        "**Framework:** Streamlit + Sklearn  \n"
        "**Autor:** QoS ML System"
    )

# ---------------------------------------------------------------------------
# Inicializar estado de sesión
# ---------------------------------------------------------------------------
for key in ["df", "preprocessor", "trainer", "X_test_raw", "results_ready",
            "graph_paths", "target_col"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "results_ready" not in st.session_state:
    st.session_state["results_ready"] = False

# ---------------------------------------------------------------------------
# PASO 1 — Carga de datos
# ---------------------------------------------------------------------------
section_title("1. Cargar Dataset")

col_upload, col_sample = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Sube tu archivo Excel (.xlsx) o CSV",
        type=["xlsx", "xls", "csv"],
        help="El archivo debe contener columnas numéricas de métricas QoS "
             "y una columna objetivo (calidad_red, quality, label, etc.)",
    )

with col_sample:
    st.markdown("**¿No tienes dataset?**")
    if st.button("🎲 Generar dataset de ejemplo", use_container_width=True):
        from data.sample_qos_dataset import generate_qos_dataset
        sample_df = generate_qos_dataset(1000)
        tmp_path = os.path.join(tempfile.gettempdir(), "qos_sample.xlsx")
        sample_df.to_excel(tmp_path, index=False)
        with open(tmp_path, "rb") as f:
            st.download_button(
                "⬇️ Descargar dataset ejemplo",
                data=f.read(),
                file_name="qos_datos_ejemplo.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        st.session_state["df"] = sample_df
        st.success("Dataset de ejemplo cargado en memoria.")

# Cargar desde archivo subido
if uploaded_file:
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=Path(uploaded_file.name).suffix
    ) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Cargando archivo..."):
        try:
            df = load_file(tmp_path)
            st.session_state["df"] = df
            st.success(f"✅ Archivo cargado: **{uploaded_file.name}** — {df.shape[0]} filas × {df.shape[1]} columnas")
        except Exception as e:
            st.error(f"Error al cargar el archivo: {e}")

# ---------------------------------------------------------------------------
# Mostrar previsualización si hay datos
# ---------------------------------------------------------------------------
if st.session_state["df"] is not None:
    df = st.session_state["df"]

    with st.expander("📋 Vista previa del dataset", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(df.head(10), use_container_width=True)
        with col2:
            # Validación
            validation = validate_dataframe(df)
            st.markdown("**Resumen de validación:**")
            v_cols = st.columns(3)
            v_cols[0].metric("Filas", validation["total_rows"])
            v_cols[1].metric("Columnas", validation["total_columns"])
            v_cols[2].metric("Duplicados", validation["duplicate_rows"])

            if validation["warnings"]:
                for w in validation["warnings"]:
                    st.warning(w)
            if not validation["is_valid"]:
                for e in validation["errors"]:
                    st.error(e)

            # Nulos
            null_series = pd.Series(validation["null_percentage"])
            if null_series.max() > 0:
                st.markdown("**% Valores nulos por columna:**")
                st.bar_chart(null_series[null_series > 0])

    # Selección de columna objetivo
    section_title("2. Columna Objetivo")
    auto_target = detect_target_column(df)
    all_cols = df.columns.tolist()

    target_col = st.selectbox(
        "Selecciona la columna que representa la calidad de red (target)",
        options=all_cols,
        index=all_cols.index(auto_target) if auto_target in all_cols else 0,
        help="Esta es la variable que el modelo aprenderá a predecir.",
    )
    st.session_state["target_col"] = target_col
    st.info(f"Columna objetivo seleccionada: **{target_col}** | "
            f"Valores únicos: {df[target_col].nunique()} — {list(df[target_col].unique())[:6]}")

    # ---------------------------------------------------------------------------
    # PASO 2 — Entrenamiento
    # ---------------------------------------------------------------------------
    section_title("3. Entrenar Modelos")

    if st.button("🚀 Iniciar Entrenamiento", type="primary", use_container_width=True):
        progress = st.progress(0, text="Iniciando pipeline...")
        status = st.empty()

        try:
            # Preprocesamiento
            status.info("⚙️ Preprocesando datos...")
            progress.progress(15, "Preprocesando...")
            preprocessor = QoSPreprocessor(
                test_size=test_size,
                random_state=int(random_state),
                k_best_features=int(k_features),
            )
            X_train, X_test, y_train, y_test = preprocessor.fit_transform(df, target_col)
            st.session_state["preprocessor"] = preprocessor

            # Guardar X_test sin normalizar para exportación
            feat_cols = preprocessor.selected_features
            X_test_df = df.drop(columns=[target_col]).iloc[: len(X_test)][feat_cols]
            st.session_state["X_test_raw"] = X_test_df.reset_index(drop=True)

            progress.progress(30, "Entrenando modelos...")
            status.info("🤖 Entrenando y comparando modelos...")

            # Entrenamiento
            trainer = QoSModelTrainer(
                models_dir=MODELS_DIR,
                random_state=int(random_state),
            )
            results = trainer.train_all(
                X_train, X_test, y_train, y_test,
                class_names=preprocessor.class_names,
            )
            st.session_state["trainer"] = trainer

            progress.progress(65, "Generando visualizaciones...")
            status.info("📊 Generando gráficas...")

            # Gráficas
            best_metrics = results[trainer.best_model_name]
            importances_df = trainer.get_feature_importances(preprocessor.selected_features)
            comparison_df = trainer.get_comparison_dataframe()

            graph_paths = generate_all_plots(
                df=df,
                target_col=target_col,
                y_test=y_test,
                class_names=preprocessor.class_names,
                confusion_mat=best_metrics["confusion_matrix"],
                best_model_name=trainer.best_model_name,
                importances_df=importances_df,
                comparison_df=comparison_df,
                graphs_dir=GRAPHS_DIR,
            )
            st.session_state["graph_paths"] = graph_paths

            progress.progress(85, "Exportando resultados...")
            status.info("💾 Exportando archivos...")

            # Exportar predicciones
            y_pred = trainer.predict(X_test)
            y_pred_labels = preprocessor.decode_labels(y_pred)
            y_true_labels = preprocessor.decode_labels(y_test)
            proba = trainer.predict_proba(X_test)

            pred_path = export_predictions_to_excel(
                original_df=st.session_state["X_test_raw"],
                y_pred_labels=y_pred_labels,
                y_true_labels=y_true_labels,
                proba_matrix=proba,
                class_names=preprocessor.class_names,
                output_path=os.path.join(REPORTS_DIR, "predicciones_qos.xlsx"),
            )

            # Exportar métricas
            metrics_path = export_metrics_report(
                model_name=trainer.best_model_name,
                metrics=best_metrics,
                comparison_df=comparison_df,
                class_names=preprocessor.class_names,
                preprocessing_summary=preprocessor.get_preprocessing_summary(),
                output_path=os.path.join(REPORTS_DIR, "reporte_metricas.xlsx"),
            )

            # Exportar reporte texto
            txt_path = export_classification_report_txt(
                classification_report_str=best_metrics["classification_report"],
                model_name=trainer.best_model_name,
                output_path=os.path.join(REPORTS_DIR, "classification_report.txt"),
            )

            # Guardar modelo
            model_pkl_path = trainer.save_best_model("best_model.pkl")

            # ZIP con todo
            all_files = list(graph_paths.values()) + [
                pred_path, metrics_path, txt_path, model_pkl_path
            ]
            zip_path = package_results_zip(
                all_files,
                output_zip=os.path.join(REPORTS_DIR, "resultados_qos.zip"),
            )

            st.session_state["export_paths"] = {
                "predicciones": pred_path,
                "metricas": metrics_path,
                "reporte_txt": txt_path,
                "modelo_pkl": model_pkl_path,
                "zip": zip_path,
            }
            st.session_state["results_ready"] = True

            progress.progress(100, "¡Completado!")
            status.success(
                f"✅ Entrenamiento completado. Mejor modelo: "
                f"**{trainer.best_model_name}** "
                f"(Accuracy: {best_metrics['accuracy']:.4f})"
            )

        except Exception as e:
            st.error(f"❌ Error durante el entrenamiento: {e}")
            logger.exception("Error en entrenamiento")
            progress.empty()

# ---------------------------------------------------------------------------
# PASO 3 — Resultados
# ---------------------------------------------------------------------------
if st.session_state.get("results_ready") and st.session_state.get("trainer"):
    trainer: QoSModelTrainer = st.session_state["trainer"]
    preprocessor: QoSPreprocessor = st.session_state["preprocessor"]
    graph_paths: dict = st.session_state["graph_paths"]
    best_name = trainer.best_model_name
    best_metrics = trainer.results[best_name]

    # ─── Métricas ─────────────────────────────────────────────────────────
    section_title("4. Resultados del Mejor Modelo")

    st.markdown(
        f"<div style='padding:0.5rem 0;'>"
        f"🏆 Mejor modelo seleccionado: <strong style='color:#1E3A5F'>{best_name}</strong>"
        f"</div>",
        unsafe_allow_html=True,
    )

    m_cols = st.columns(4)
    kpi_map = [
        ("Accuracy",   f"{best_metrics['accuracy']:.4f}"),
        ("Precision",  f"{best_metrics['precision']:.4f}"),
        ("Recall",     f"{best_metrics['recall']:.4f}"),
        ("F1-Score",   f"{best_metrics['f1_score']:.4f}"),
    ]
    for col, (label, val) in zip(m_cols, kpi_map):
        with col:
            metric_card(label, val)

    # ─── Comparación de modelos ─────────────────────────────────────────
    section_title("5. Comparación de Modelos")
    comparison_df = trainer.get_comparison_dataframe()
    st.dataframe(
        comparison_df.style.background_gradient(cmap="Blues", subset=["Accuracy", "F1-Score"]),
        use_container_width=True,
    )

    # ─── Reporte de clasificación ────────────────────────────────────────
    with st.expander("📄 Classification Report completo"):
        st.code(best_metrics["classification_report"], language="text")

    # ─── Visualizaciones ─────────────────────────────────────────────────
    section_title("6. Visualizaciones")

    tabs = st.tabs([
        "🔥 Heatmap", "📊 Clases", "📈 Histogramas",
        "🎯 Confusión", "🌳 Importancia", "⚖️ Comparación"
    ])

    with tabs[0]:
        show_image(graph_paths.get("heatmap", ""), "Mapa de Correlación")
    with tabs[1]:
        show_image(graph_paths.get("clases", ""), "Distribución de Clases")
    with tabs[2]:
        show_image(graph_paths.get("histogramas", ""), "Histogramas de Variables")
    with tabs[3]:
        show_image(graph_paths.get("confusion", ""), "Matriz de Confusión")
    with tabs[4]:
        imp_path = graph_paths.get("importancia")
        if imp_path:
            show_image(imp_path, "Importancia de Variables")
        else:
            st.info("El modelo seleccionado no provee importancia de características.")
    with tabs[5]:
        show_image(graph_paths.get("comparacion", ""), "Comparación de Modelos")

    # ─── Predicción individual ──────────────────────────────────────────
    section_title("7. Predicción Manual")
    st.markdown("Introduce valores de métricas QoS para obtener una predicción en tiempo real:")

    feat_names = preprocessor.selected_features
    n_feats = len(feat_names)
    input_cols = st.columns(min(4, n_feats))
    input_values = {}

    df_ref = st.session_state["df"]
    for i, feat in enumerate(feat_names):
        col = input_cols[i % len(input_cols)]
        default_val = float(df_ref[feat].median()) if feat in df_ref.columns else 0.0
        input_values[feat] = col.number_input(
            feat.replace("_", " ").title(),
            value=default_val,
            format="%.2f",
            key=f"input_{feat}",
        )

    if st.button("🔍 Predecir calidad", use_container_width=True):
        try:
            input_df = pd.DataFrame([input_values])
            X_new = preprocessor.transform_new(input_df)
            pred_encoded = trainer.predict(X_new)
            pred_label = preprocessor.decode_labels(pred_encoded)[0]
            proba = trainer.predict_proba(X_new)

            color_map = {
                "Excelente": "#10B981", "Buena": "#3B82F6",
                "Regular": "#F59E0B", "Mala": "#EF4444",
            }
            color = color_map.get(pred_label, "#6B7280")

            st.markdown(
                f"""
                <div style='background:{color}22;border-left:5px solid {color};
                padding:1rem 1.5rem;border-radius:8px;margin:0.5rem 0;'>
                    <div style='font-size:1.5rem;font-weight:700;color:{color}'>
                        📡 Calidad predicha: {pred_label}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if proba is not None:
                prob_df = pd.DataFrame(
                    {"Clase": preprocessor.class_names,
                     "Probabilidad": np.round(proba[0] * 100, 2)}
                ).sort_values("Probabilidad", ascending=False)
                st.bar_chart(prob_df.set_index("Clase"))

        except Exception as e:
            st.error(f"Error en predicción: {e}")

    # ─── Descargas ───────────────────────────────────────────────────────
    section_title("8. Exportar Resultados")
    export_paths = st.session_state.get("export_paths", {})

    d_cols = st.columns(4)

    def _download_btn(col, label, path, mime, fname):
        if path and os.path.isfile(path):
            with open(path, "rb") as f:
                col.download_button(label, f.read(), file_name=fname, mime=mime,
                                    use_container_width=True)

    _download_btn(d_cols[0], "📥 Predicciones Excel",
                  export_paths.get("predicciones"),
                  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                  "predicciones_qos.xlsx")

    _download_btn(d_cols[1], "📊 Reporte Métricas",
                  export_paths.get("metricas"),
                  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                  "reporte_metricas.xlsx")

    _download_btn(d_cols[2], "🤖 Modelo (.pkl)",
                  export_paths.get("modelo_pkl"),
                  "application/octet-stream",
                  "best_model.pkl")

    _download_btn(d_cols[3], "📦 Todo (ZIP)",
                  export_paths.get("zip"),
                  "application/zip",
                  "resultados_qos.zip")
