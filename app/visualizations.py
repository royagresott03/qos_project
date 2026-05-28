import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Backend sin pantalla (compatible con Streamlit)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


PALETTE = "viridis"
FIG_DPI = 150
TITLE_FONTSIZE = 14
AXIS_FONTSIZE = 11
GRAPHS_DIR = "graphs"

sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": FIG_DPI,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

os.makedirs(GRAPHS_DIR, exist_ok=True)


def _save_and_close(fig: plt.Figure, filename: str, graphs_dir: str = GRAPHS_DIR) -> str:
    """Guarda la figura y cierra el objeto para liberar memoria."""
    path = os.path.join(graphs_dir, filename)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Gráfica guardada: %s", path)
    return path



def plot_correlation_heatmap(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    graphs_dir: str = GRAPHS_DIR,
    max_cols: int = 15,
) -> str:

    num_df = df.select_dtypes(include=[np.number])
    if target_col and target_col in num_df.columns:
        num_df = num_df.drop(columns=[target_col])


    if num_df.shape[1] > max_cols:
        num_df = num_df.iloc[:, :max_cols]

    corr = num_df.corr()
    n = len(corr)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.7), max(6, n * 0.6)))

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 8},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Mapa de Correlación entre Variables QoS", fontsize=TITLE_FONTSIZE, pad=15)
    fig.tight_layout()
    return _save_and_close(fig, "heatmap_correlacion.png", graphs_dir)




def plot_class_distribution(
    y_labels: np.ndarray,
    class_names: List[str],
    graphs_dir: str = GRAPHS_DIR,
) -> str:

    fig, ax = plt.subplots(figsize=(8, 5))

    # Calcular conteos
    unique, counts = np.unique(y_labels, return_counts=True)
    labels = [class_names[int(u)] if u < len(class_names) else str(u) for u in unique]
    total = counts.sum()

    colors = sns.color_palette(PALETTE, len(unique))
    bars = ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=1.5)


    for bar, count in zip(bars, counts):
        pct = count / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            f"{count}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=AXIS_FONTSIZE,
        )

    ax.set_title("Distribución de Clases de Calidad QoS", fontsize=TITLE_FONTSIZE, pad=12)
    ax.set_xlabel("Calidad de Red", fontsize=AXIS_FONTSIZE)
    ax.set_ylabel("Número de Muestras", fontsize=AXIS_FONTSIZE)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    return _save_and_close(fig, "distribucion_clases.png", graphs_dir)




def plot_feature_histograms(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    graphs_dir: str = GRAPHS_DIR,
    max_features: int = 12,
) -> str:

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)
    num_cols = num_cols[:max_features]

    n_cols = 3
    n_rows = (len(num_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, col in enumerate(num_cols):
        ax = axes_flat[i]
        if target_col and target_col in df.columns:
            for cat in df[target_col].unique():
                subset = df[df[target_col] == cat][col].dropna()
                ax.hist(subset, bins=25, alpha=0.6, label=str(cat), edgecolor="none")
            ax.legend(fontsize=8, framealpha=0.7)
        else:
            ax.hist(df[col].dropna(), bins=25, color="#4C72B0", edgecolor="none")

        ax.set_title(col.replace("_", " ").title(), fontsize=10)
        ax.set_xlabel("Valor", fontsize=8)
        ax.set_ylabel("Frecuencia", fontsize=8)

    # Ocultar ejes vacíos
    for j in range(len(num_cols), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Distribución de Variables QoS por Clase", fontsize=TITLE_FONTSIZE, y=1.01)
    fig.tight_layout()
    return _save_and_close(fig, "histogramas_variables.png", graphs_dir)




def plot_feature_importances(
    importances_df: pd.DataFrame,
    graphs_dir: str = GRAPHS_DIR,
    top_n: int = 15,
) -> str:

    df_top = importances_df.head(top_n).copy()
    df_top["feature"] = df_top["feature"].str.replace("_", " ").str.title()

    fig, ax = plt.subplots(figsize=(9, max(5, len(df_top) * 0.45)))
    colors = sns.color_palette(PALETTE, len(df_top))[::-1]

    ax.barh(df_top["feature"], df_top["importance"], color=colors, edgecolor="white")
    ax.set_xlabel("Importancia Relativa", fontsize=AXIS_FONTSIZE)
    ax.set_title(f"Top {len(df_top)} Variables más Importantes", fontsize=TITLE_FONTSIZE, pad=12)
    ax.invert_yaxis()

    # Anotar valores
    for patch in ax.patches:
        w = patch.get_width()
        ax.text(
            w + 0.001, patch.get_y() + patch.get_height() / 2,
            f"{w:.4f}", va="center", fontsize=9,
        )

    fig.tight_layout()
    return _save_and_close(fig, "importancia_variables.png", graphs_dir)


# 5. Matriz de confusión
def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    model_name: str = "Modelo",
    graphs_dir: str = GRAPHS_DIR,
) -> str:

    fig, ax = plt.subplots(figsize=(8, 6))


    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Proporción")

    n = len(class_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)


    thresh = 0.5
    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(
                j, i,
                f"{cm[i, j]}\n({cm_norm[i, j]:.1%})",
                ha="center", va="center",
                fontsize=9, color=color, fontweight="bold",
            )

    ax.set_xlabel("Predicción", fontsize=AXIS_FONTSIZE)
    ax.set_ylabel("Valor Real", fontsize=AXIS_FONTSIZE)
    ax.set_title(
        f"Matriz de Confusión — {model_name}",
        fontsize=TITLE_FONTSIZE, pad=14,
    )
    fig.tight_layout()
    return _save_and_close(fig, "matriz_confusion.png", graphs_dir)




def plot_model_comparison(
    comparison_df: pd.DataFrame,
    graphs_dir: str = GRAPHS_DIR,
) -> str:

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    df_plot = comparison_df.dropna(subset=["Accuracy"])[["Modelo"] + metrics].copy()

    x = np.arange(len(df_plot))
    width = 0.2
    fig, ax = plt.subplots(figsize=(max(10, len(df_plot) * 2.2), 6))
    colors = sns.color_palette(PALETTE, len(metrics))

    for idx, (metric, color) in enumerate(zip(metrics, colors)):
        offset = (idx - len(metrics) / 2 + 0.5) * width
        bars = ax.bar(x + offset, df_plot[metric], width, label=metric, color=color)
        for bar in bars:
            h = bar.get_height()
            if h:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.005,
                    f"{h:.3f}",
                    ha="center", va="bottom", fontsize=7, rotation=45,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(df_plot["Modelo"], rotation=15, ha="right", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Puntuación", fontsize=AXIS_FONTSIZE)
    ax.set_title("Comparación de Modelos de ML", fontsize=TITLE_FONTSIZE, pad=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    fig.tight_layout()
    return _save_and_close(fig, "comparacion_modelos.png", graphs_dir)




def generate_all_plots(
    df: pd.DataFrame,
    target_col: str,
    y_test: np.ndarray,
    class_names: List[str],
    confusion_mat: np.ndarray,
    best_model_name: str,
    importances_df: Optional[pd.DataFrame],
    comparison_df: pd.DataFrame,
    graphs_dir: str = GRAPHS_DIR,
) -> Dict[str, str]:

    paths: Dict[str, str] = {}

    paths["heatmap"] = plot_correlation_heatmap(df, target_col, graphs_dir)
    paths["clases"] = plot_class_distribution(y_test, class_names, graphs_dir)
    paths["histogramas"] = plot_feature_histograms(df, target_col, graphs_dir)
    paths["confusion"] = plot_confusion_matrix(
        confusion_mat, class_names, best_model_name, graphs_dir
    )
    paths["comparacion"] = plot_model_comparison(comparison_df, graphs_dir)

    if importances_df is not None and not importances_df.empty:
        paths["importancia"] = plot_feature_importances(importances_df, graphs_dir)

    logger.info("Todas las gráficas generadas: %s", list(paths.keys()))
    return paths
