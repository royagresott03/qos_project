"""
Módulo de preprocesamiento de datos QoS.
Realiza limpieza, codificación, normalización y división del dataset.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
import logging

logger = logging.getLogger(__name__)



# Clase principal de preprocesamiento


class QoSPreprocessor:
    """
    Pipeline de preprocesamiento completo para datos QoS.

    Atributos:
        scaler: StandardScaler ajustado en train
        label_encoder: LabelEncoder para la variable objetivo
        feature_names: Nombres de columnas de entrada seleccionadas
        selected_features: Características seleccionadas por SelectKBest
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42,
                 k_best_features: int = 10):
        """
        Inicializa el preprocesador.

        Args:
            test_size: Fracción del dataset para test (default 0.2)
            random_state: Semilla de aleatoriedad para reproducibilidad
            k_best_features: Número de características a seleccionar (0 = todas)
        """
        self.test_size = test_size
        self.random_state = random_state
        self.k_best_features = k_best_features

        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names: List[str] = []
        self.selected_features: List[str] = []
        self.class_names: List[str] = []
        self._fitted = False

    # Pasos individuales de preprocesamiento

    def _drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Elimina filas duplicadas exactas."""
        n_before = len(df)
        df = df.drop_duplicates()
        n_removed = n_before - len(df)
        if n_removed:
            logger.info("Eliminadas %d filas duplicadas.", n_removed)
        return df

    def _handle_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Maneja valores nulos:
        - Columnas numéricas → rellena con la mediana
        - Columnas categóricas → rellena con la moda
        """
        for col in df.columns:
            n_nulls = df[col].isnull().sum()
            if n_nulls == 0:
                continue

            if pd.api.types.is_numeric_dtype(df[col]):
                fill_value = df[col].median()
                strategy = "mediana"
            else:
                fill_value = df[col].mode()[0]
                strategy = "moda"

            df[col] = df[col].fillna(fill_value)
            logger.info(
                "Columna '%s': %d nulos rellenados con %s (%.4f).",
                col, n_nulls, strategy, fill_value
            )
        return df

    def _encode_categoricals(
        self, df: pd.DataFrame, target_col: str
    ) -> pd.DataFrame:
        """
        Codifica variables categóricas distintas al target.
        Usa codificación ordinal simple para variables binarias,
        y one-hot encoding para las demás (drop_first=True).
        """
        cat_cols = [
            c for c in df.select_dtypes(exclude=[np.number]).columns
            if c != target_col
        ]
        if not cat_cols:
            return df

        # One-hot encoding para categóricas no binarias
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        logger.info("Columnas categóricas codificadas: %s", cat_cols)
        return df

    def _remove_outliers_iqr(
        self, df: pd.DataFrame, target_col: str, threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Elimina outliers extremos usando z-score > threshold en columnas numéricas.
        No modifica la columna target

            df: DataFrame de entrada
            target_col: Columna a excluir de la detección de outliers
            threshold: Número de desviaciones estándar para considerar outlier

        Returns:
            DataFrame sin outliers extremos
        """
        num_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != target_col
        ]
        if not num_cols:
            return df

        from scipy import stats as scipy_stats
        z_scores = np.abs(scipy_stats.zscore(df[num_cols]))
        mask = (z_scores < threshold).all(axis=1)
        n_removed = (~mask).sum()
        if n_removed:
            logger.info(
                "Eliminados %d outliers extremos (z-score > %.1f).",
                n_removed, threshold
            )
        return df[mask].reset_index(drop=True)

    # Método principal fit_transform
        """
        Aplica el pipeline completo de preprocesamiento y retorna
        los conjuntos train/test listos para el ML.

        Pasos:
            Eliminar duplicados
            Manejar nulos
            Codificar categóricas
            Eliminar outliers extremos
            Codificar target
            Normalizar features
            Selección de características (opcional)
            Dividir en train/test

        Args:
            df: DataFrame completo (features + target)
            target_col: Nombre de la columna objetivo

        Returns:
            Tupla (X_train, X_test, y_train, y_test)
        """
    def fit_transform(
        self, df: pd.DataFrame, target_col: str
    ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:

        logger.info("Iniciando pipeline de preprocesamiento...")

        # 1. Duplicados
        df = self._drop_duplicates(df)

        # 2. Nulos
        df = self._handle_nulls(df)

        # 3. Codificar categóricas (excepto target)
        df = self._encode_categoricals(df, target_col)

        # 4. Outliers extremos
        try:
            df = self._remove_outliers_iqr(df, target_col)
        except Exception as e:
            logger.warning("Detección de outliers omitida: %s", e)

        # 5. Separar features y target
        y_raw = df[target_col].values
        X_df = df.drop(columns=[target_col])
        self.feature_names = X_df.columns.tolist()

        # 6. Codificar target con LabelEncoder
        y_encoded = self.label_encoder.fit_transform(y_raw)
        self.class_names = list(self.label_encoder.classes_)
        logger.info("Clases detectadas: %s", self.class_names)

        X = X_df.values.astype(float)

        # 7. Selección de características (SelectKBest)
        k = min(self.k_best_features, X.shape[1])
        if self.k_best_features > 0 and k < X.shape[1]:
            selector = SelectKBest(f_classif, k=k)
            X = selector.fit_transform(X, y_encoded)
            selected_mask = selector.get_support()
            self.selected_features = [
                self.feature_names[i]
                for i, sel in enumerate(selected_mask) if sel
            ]
            logger.info("Características seleccionadas: %s", self.selected_features)
        else:
            self.selected_features = self.feature_names

        # 8. Dividir en train/test estratificado
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_encoded,
        )

        # 9. Normalizar SOLO con estadísticas de train (evitar data leakage)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self._fitted = True
        logger.info(
            "Preprocesamiento completo. Train: %d | Test: %d",
            len(X_train), len(X_test)
        )
        return X_train, X_test, y_train, y_test

    def transform_new(self, df: pd.DataFrame, target_col: Optional[str] = None) -> np.ndarray:
        """
        Transforma nuevos datos usando el scaler ajustado en entrenamiento.

        Args:
            df: Nuevos datos (sin columna target)
            target_col: Si está presente, se descarta

        Returns:
            Array numpy normalizado y listo para la predicción
        """
        if not self._fitted:
            raise RuntimeError("Llama fit_transform() antes de transform_new().")

        if target_col and target_col in df.columns:
            df = df.drop(columns=[target_col])

        # Asegurar mismo orden de columnas
        missing = set(self.selected_features) - set(df.columns)
        if missing:
            raise ValueError(f"Columnas faltantes en nuevos datos: {missing}")

        X = df[self.selected_features].values.astype(float)
        return self.scaler.transform(X)

    def decode_labels(self, y_encoded: np.ndarray) -> np.ndarray:
        """
        esto convierte etiquetas numéricas de vuelta a texto.

        Args:
            y_encoded: Array de enteros codificados

        Returns:
            Array de strings con las clases originales
        """
        return self.label_encoder.inverse_transform(y_encoded)

    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Retorna un resumen del preprocesamiento aplicado."""
        return {
            "test_size": self.test_size,
            "random_state": self.random_state,
            "features_used": self.selected_features,
            "n_features": len(self.selected_features),
            "class_names": self.class_names,
            "scaler_mean": self.scaler.mean_.tolist() if self._fitted else None,
            "scaler_std": self.scaler.scale_.tolist() if self._fitted else None,
        }
