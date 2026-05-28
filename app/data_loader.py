import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Constantes del módulo
SUPPORTED_EXTENSIONS = {".xlsx", ".xls", ".csv"}

EXPECTED_QOS_COLUMNS = [
    "latencia_ms", "jitter_ms", "rtt_ms", "perdida_paquetes",
    "throughput_mbps", "ancho_banda_mbps", "intensidad_senal",
    "velocidad_descarga", "velocidad_subida", "congestion_red",
]

TARGET_COLUMN_CANDIDATES = ["calidad_red", "quality", "label", "clase", "categoria"]

# Funciones públicas

def load_file(filepath: str) -> pd.DataFrame:

    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {filepath}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Formato '{ext}' no soportado. Use: {SUPPORTED_EXTENSIONS}"
        )

    logger.info("Cargando archivo: %s", filepath)

    if ext in {".xlsx", ".xls"}:
        df = pd.read_excel(filepath, engine="openpyxl")
    else:
        # Detectar separador automáticamente
        df = pd.read_csv(filepath, sep=None, engine="python")

    logger.info("Archivo cargado: %d filas × %d columnas", *df.shape)
    return df


def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_columns": df.select_dtypes(exclude=[np.number]).columns.tolist(),
        "null_counts": df.isnull().sum().to_dict(),
        "null_percentage": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "duplicate_rows": df.duplicated().sum(),
        "warnings": [],
        "errors": [],
        "is_valid": True,
    }

    # Validar tamaño mínimo
    if len(df) < 20:
        report["errors"].append(
            "El dataset tiene menos de 20 filas. Insuficiente para entrenar."
        )
        report["is_valid"] = False

    # Advertir columnas con muchos nulos
    for col, pct in report["null_percentage"].items():
        if pct > 50:
            report["warnings"].append(
                f"Columna '{col}' tiene {pct:.1f}% de valores nulos."
            )

    # Advertir duplicados
    if report["duplicate_rows"] > 0:
        report["warnings"].append(
            f"Se encontraron {report['duplicate_rows']} filas duplicadas."
        )

    return report


def detect_target_column(df: pd.DataFrame) -> Optional[str]:

    columns_lower = {c.lower(): c for c in df.columns}

   
    for candidate in TARGET_COLUMN_CANDIDATES:
        if candidate in columns_lower:
            return columns_lower[candidate]


    for col in df.select_dtypes(exclude=[np.number]).columns:
        n_unique = df[col].nunique()
        if 2 <= n_unique <= 6:
            logger.info("Columna objetivo detectada automáticamente: %s", col)
            return col

    return None


def get_feature_columns(df: pd.DataFrame, target_col: str) -> list:

    return [c for c in df.columns if c != target_col]


def preview_dataframe(df: pd.DataFrame, n: int = 5) -> Dict[str, Any]:

    return {
        "head": df.head(n),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "describe": df.describe(include="all"),
        "shape": df.shape,
    }
