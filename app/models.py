"""
Módulo de modelos de Machine Learning para clasificación QoS.
Implementa, entrena, compara y selecciona el mejor clasificador.
"""

import numpy as np
import pandas as pd
import pickle
import os
import time
from typing import Dict, Any, Tuple, List, Optional
import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# XGBoost es opcional: si no está instalado, se omite silenciosamente
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Definición de modelos disponibles
# ---------------------------------------------------------------------------

def build_model_catalog(random_state: int = 42) -> Dict[str, Any]:
    """
    Construye el catálogo de modelos con sus hiperparámetros por defecto.

    Args:
        random_state: Semilla para reproducibilidad

    Returns:
        Diccionario {nombre_modelo: instancia_sklearn}
    """
    catalog = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=5,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=5,
            class_weight="balanced",
            random_state=random_state,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            metric="euclidean",
            n_jobs=-1,
        ),
        "SVM": SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=random_state,
        ),
    }

    if XGBOOST_AVAILABLE:
        catalog["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
        )
        logger.info("XGBoost disponible y añadido al catálogo.")
    else:
        logger.info("XGBoost no disponible. Se omite.")

    return catalog


# ---------------------------------------------------------------------------
# Clase principal del entrenador
# ---------------------------------------------------------------------------

class QoSModelTrainer:
    """
    Entrena múltiples modelos, los compara y guarda el mejor.

    Atributos:
        results: Diccionario con métricas de cada modelo
        best_model_name: Nombre del modelo con mejor accuracy
        best_model: Instancia del mejor modelo entrenado
    """

    def __init__(self, models_dir: str = "models", random_state: int = 42):
        """
        Args:
            models_dir: Directorio donde guardar modelos .pkl
            random_state: Semilla de aleatoriedad
        """
        self.models_dir = models_dir
        self.random_state = random_state
        self.results: Dict[str, Dict[str, Any]] = {}
        self.best_model_name: Optional[str] = None
        self.best_model = None
        self.class_names: List[str] = []
        os.makedirs(models_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Entrenamiento y evaluación
    # ------------------------------------------------------------------

    def _evaluate_model(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        class_names: List[str],
    ) -> Dict[str, Any]:
        """
        Evalúa un modelo ya entrenado sobre el conjunto de test.

        Returns:
            Diccionario con accuracy, precision, recall, f1, confusion_matrix
            y classification_report detallado.
        """
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "recall":    round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "f1_score":  round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(
                y_test, y_pred,
                target_names=class_names,
                zero_division=0,
            ),
            "y_pred": y_pred,
        }
        return metrics

    def train_all(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        class_names: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Entrena y evalúa todos los modelos del catálogo.

        Args:
            X_train, X_test: Features normalizadas
            y_train, y_test: Etiquetas codificadas
            class_names: Lista de nombres de clases

        Returns:
            Diccionario con resultados por modelo
        """
        self.class_names = class_names
        catalog = build_model_catalog(self.random_state)
        all_results: Dict[str, Dict[str, Any]] = {}

        for name, model in catalog.items():
            logger.info("Entrenando: %s ...", name)
            t0 = time.time()

            try:
                model.fit(X_train, y_train)
                elapsed = round(time.time() - t0, 2)
                metrics = self._evaluate_model(model, X_test, y_test, class_names)
                metrics["train_time_sec"] = elapsed
                metrics["model"] = model
                all_results[name] = metrics

                logger.info(
                    "  %-15s  acc=%.4f  f1=%.4f  tiempo=%.2fs",
                    name, metrics["accuracy"], metrics["f1_score"], elapsed
                )

            except Exception as e:
                logger.error("Error entrenando %s: %s", name, e)
                all_results[name] = {"error": str(e)}

        self.results = all_results
        self._select_best_model()
        return all_results

    def _select_best_model(self) -> None:
        """Selecciona el modelo con mayor accuracy (sin errores)."""
        valid = {
            k: v for k, v in self.results.items()
            if "error" not in v
        }
        if not valid:
            raise RuntimeError("Ningún modelo se entrenó correctamente.")

        self.best_model_name = max(valid, key=lambda k: valid[k]["accuracy"])
        self.best_model = self.results[self.best_model_name]["model"]
        logger.info(
            "Mejor modelo: %s (accuracy=%.4f)",
            self.best_model_name,
            self.results[self.best_model_name]["accuracy"],
        )

    # ------------------------------------------------------------------
    # Predicción
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones con el mejor modelo entrenado.

        Args:
            X: Features normalizadas

        Returns:
            Array con etiquetas numéricas predichas
        """
        if self.best_model is None:
            raise RuntimeError("Primero llama a train_all().")
        return self.best_model.predict(X)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Retorna probabilidades de clase si el modelo las soporta.

        Args:
            X: Features normalizadas

        Returns:
            Matriz de probabilidades o None
        """
        if hasattr(self.best_model, "predict_proba"):
            return self.best_model.predict_proba(X)
        return None

    # ------------------------------------------------------------------
    # Importancia de características
    # ------------------------------------------------------------------

    def get_feature_importances(
        self, feature_names: List[str]
    ) -> Optional[pd.DataFrame]:
        """
        Extrae la importancia de características del mejor modelo
        (solo disponible para modelos basados en árboles y XGBoost).

        Args:
            feature_names: Lista de nombres de columnas de entrada

        Returns:
            DataFrame ordenado por importancia o None
        """
        if self.best_model is None:
            return None

        if hasattr(self.best_model, "feature_importances_"):
            importances = self.best_model.feature_importances_
            df = pd.DataFrame(
                {"feature": feature_names, "importance": importances}
            ).sort_values("importance", ascending=False)
            return df

        logger.info(
            "El modelo '%s' no expone feature_importances_.",
            self.best_model_name,
        )
        return None

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save_best_model(self, filename: str = "best_model.pkl") -> str:
        """
        Guarda el mejor modelo entrenado como archivo .pkl.

        Args:
            filename: Nombre del archivo (sin ruta)

        Returns:
            Ruta completa del archivo guardado
        """
        if self.best_model is None:
            raise RuntimeError("No hay modelo para guardar.")

        path = os.path.join(self.models_dir, filename)
        payload = {
            "model": self.best_model,
            "model_name": self.best_model_name,
            "class_names": self.class_names,
            "metrics": {
                k: v
                for k, v in self.results[self.best_model_name].items()
                if k not in ("model", "y_pred", "confusion_matrix", "classification_report")
            },
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

        logger.info("Modelo guardado en: %s", path)
        return path

    @staticmethod
    def load_model(path: str) -> Dict[str, Any]:
        """
        Carga un modelo desde un archivo .pkl.

        Args:
            path: Ruta al archivo .pkl

        Returns:
            Diccionario con modelo y metadatos
        """
        with open(path, "rb") as f:
            payload = pickle.load(f)
        logger.info("Modelo cargado: %s", payload.get("model_name"))
        return payload

    # ------------------------------------------------------------------
    # Reporte comparativo
    # ------------------------------------------------------------------

    def get_comparison_dataframe(self) -> pd.DataFrame:
        """
        Construye un DataFrame comparativo de todos los modelos entrenados.

        Returns:
            DataFrame con métricas por modelo, ordenado por accuracy desc
        """
        rows = []
        for name, res in self.results.items():
            if "error" in res:
                rows.append({
                    "Modelo": name, "Accuracy": None,
                    "Precision": None, "Recall": None,
                    "F1-Score": None, "Tiempo (s)": None,
                    "Error": res["error"],
                })
            else:
                rows.append({
                    "Modelo": name,
                    "Accuracy":   res["accuracy"],
                    "Precision":  res["precision"],
                    "Recall":     res["recall"],
                    "F1-Score":   res["f1_score"],
                    "Tiempo (s)": res["train_time_sec"],
                })

        df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False)
        return df.reset_index(drop=True)
