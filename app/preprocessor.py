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


class QoSPreprocessor:
    """
    Pipeline de preprocesamiento completo para datos QoS.
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42,
                 k_best_features: int = 10):
        self.test_size = test_size
        self.random_state = random_state
        self.k_best_features = k_best_features

        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names: List[str] = []
        self.selected_features: List[str] = []
        self.class_names: List[str] = []
        self._fitted = False

    def _drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        n_before = len(df)
        df = df.drop_duplicates()
        n_removed = n_before - len(df)
        if n_removed:
            logger.info("Eliminadas %d filas duplicadas.", n_removed)
        return df

    def _handle_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
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
            logger.info("Columna '%s': %d nulos rellenados con %s.", col, n_nulls, strategy)
        return df

    def _encode_categoricals(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        cat_cols = [
            c for c in df.select_dtypes(exclude=[np.number]).columns
            if c != target_col
        ]
        if not cat_cols:
            return df
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        logger.info("Columnas categóricas codificadas: %s", cat_cols)
        return df

    def _remove_outliers_safe(self, df: pd.DataFrame, target_col: str,
                               threshold: float = 3.5) -> pd.DataFrame:
        """
        Eliminación SEGURA de outliers usando z-score con umbral más permisivo.

        CORRECCIÓN del bug anterior:
        - Umbral subido de 3.0 a 3.5 para ser menos agresivo
        - Verifica que queden suficientes filas antes de aplicar
        - Si quedarían menos de 50 filas, omite la eliminación completamente
        - Aplica por columna individualmente en lugar de todas a la vez

        Args:
            df: DataFrame de entrada
            target_col: Columna a excluir
            threshold: Z-score máximo permitido (default 3.5)

        Returns:
            DataFrame limpio o el original si no es seguro filtrar
        """
        num_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != target_col
        ]
        if not num_cols:
            return df

        n_original = len(df)

        # Verificar mínimo viable antes de filtrar
        if n_original < 100:
            logger.warning(
                "Dataset muy pequeño (%d filas). Omitiendo eliminación de outliers.",
                n_original
            )
            return df

        try:
            from scipy import stats as scipy_stats

            # Calcular z-scores solo para columnas numéricas
            z_scores = np.abs(scipy_stats.zscore(df[num_cols], nan_policy='omit'))

            # Máscara: filas donde TODAS las columnas están dentro del umbral
            mask = (z_scores < threshold).all(axis=1)

            n_resultado = mask.sum()

            # Seguridad: si quedarían menos del 30% de los datos, no filtrar
            if n_resultado < n_original * 0.30:
                logger.warning(
                    "Filtro de outliers eliminaría el %.0f%% de los datos (%d→%d filas). "
                    "Omitiendo para preservar el dataset.",
                    (1 - n_resultado / n_original) * 100,
                    n_original,
                    n_resultado,
                )
                return df

            # Seguridad: si quedarían menos de 50 filas absolutas, no filtrar
            if n_resultado < 50:
                logger.warning(
                    "Filtro de outliers dejaría solo %d filas. Omitiendo.",
                    n_resultado
                )
                return df

            n_removed = n_original - n_resultado
            if n_removed > 0:
                logger.info(
                    "Outliers eliminados: %d filas (%.1f%% del dataset).",
                    n_removed,
                    n_removed / n_original * 100,
                )

            return df[mask].reset_index(drop=True)

        except Exception as e:
            logger.warning("Detección de outliers omitida por error: %s", e)
            return df

    def fit_transform(
        self, df: pd.DataFrame, target_col: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Aplica el pipeline completo de preprocesamiento.

        Pasos:
            1. Eliminar duplicados
            2. Manejar nulos
            3. Codificar categóricas
            4. Eliminar outliers (con verificación de seguridad)
            5. Codificar target
            6. Selección de características
            7. Dividir en train/test
            8. Normalizar

        Args:
            df: DataFrame completo
            target_col: Columna objetivo

        Returns:
            Tupla (X_train, X_test, y_train, y_test)
        """
        logger.info("Iniciando pipeline de preprocesamiento...")
        logger.info("Dataset inicial: %d filas × %d columnas", *df.shape)

        # 1. Duplicados
        df = self._drop_duplicates(df)

        # 2. Nulos
        df = self._handle_nulls(df)

        # 3. Categóricas
        df = self._encode_categoricals(df, target_col)

        # 4. Outliers (ahora con verificación de seguridad)
        df = self._remove_outliers_safe(df, target_col)

        # Verificación crítica: asegurarse de que quedan filas suficientes
        if len(df) < 20:
            raise ValueError(
                f"El dataset quedó con solo {len(df)} filas después del preprocesamiento. "
                f"Necesitas al menos 20 filas para entrenar. "
                f"Revisa que el archivo tenga suficientes datos válidos."
            )

        logger.info("Dataset tras preprocesamiento: %d filas", len(df))

        # 5. Separar features y target
        y_raw = df[target_col].values
        X_df = df.drop(columns=[target_col])
        self.feature_names = X_df.columns.tolist()

        # 6. Codificar target
        y_encoded = self.label_encoder.fit_transform(y_raw)
        self.class_names = list(self.label_encoder.classes_)
        logger.info("Clases detectadas: %s", self.class_names)

        X = X_df.values.astype(float)

        # 7. Selección de características
        k = min(self.k_best_features, X.shape[1]) if self.k_best_features > 0 else X.shape[1]
        if k < X.shape[1] and self.k_best_features > 0:
            selector = SelectKBest(f_classif, k=k)
            X = selector.fit_transform(X, y_encoded)
            selected_mask = selector.get_support()
            self.selected_features = [
                self.feature_names[i]
                for i, sel in enumerate(selected_mask) if sel
            ]
            logger.info("Características seleccionadas (%d): %s", k, self.selected_features)
        else:
            self.selected_features = self.feature_names

        # 8. Dividir train/test estratificado
        # Verificar que hay suficientes muestras por clase para estratificar
        unique, counts = np.unique(y_encoded, return_counts=True)
        min_class_count = counts.min()

        if min_class_count < 2:
            logger.warning(
                "Alguna clase tiene menos de 2 muestras. "
                "Usando división sin estratificación."
            )
            stratify = None
        else:
            stratify = y_encoded

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify,
        )

        # 9. Normalizar (solo fit en train)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self._fitted = True
        logger.info(
            "Preprocesamiento completo. Train: %d | Test: %d",
            len(X_train), len(X_test)
        )
        return X_train, X_test, y_train, y_test

    def transform_new(self, df: pd.DataFrame,
                      target_col: Optional[str] = None) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Llama fit_transform() antes de transform_new().")

        if target_col and target_col in df.columns:
            df = df.drop(columns=[target_col])

        missing = set(self.selected_features) - set(df.columns)
        if missing:
            raise ValueError(f"Columnas faltantes en nuevos datos: {missing}")

        X = df[self.selected_features].values.astype(float)
        return self.scaler.transform(X)

    def decode_labels(self, y_encoded: np.ndarray) -> np.ndarray:
        return self.label_encoder.inverse_transform(y_encoded)

    def get_preprocessing_summary(self) -> Dict[str, Any]:
        return {
            "test_size": self.test_size,
            "random_state": self.random_state,
            "features_used": self.selected_features,
            "n_features": len(self.selected_features),
            "class_names": self.class_names,
            "scaler_mean": self.scaler.mean_.tolist() if self._fitted else None,
            "scaler_std": self.scaler.scale_.tolist() if self._fitted else None,
        }