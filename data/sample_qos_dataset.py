"""
Generador de dataset de ejemplo para QoS en redes móviles.
Ejecutar este script para generar datos de entrenamiento sintéticos.
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)
N = 1000


def generate_qos_dataset(n_samples: int = 1000) -> pd.DataFrame:
    """
    Genera un dataset sintético de calidad de servicio (QoS)
    para redes móviles con 4 categorías de calidad.

    Args:
        n_samples: Número de muestras a generar

    Returns:
        DataFrame con variables QoS y etiqueta de calidad
    """

    data = []

    categories = {
        "Excelente": {
            "latencia_ms":       (5,   20),
            "jitter_ms":         (1,   5),
            "rtt_ms":            (10,  40),
            "perdida_paquetes":  (0.0, 0.5),
            "throughput_mbps":   (80,  150),
            "ancho_banda_mbps":  (90,  200),
            "intensidad_senal":  (-65, -50),
            "velocidad_descarga": (70, 120),
            "velocidad_subida":  (20,  50),
            "congestion_red":    (0,   10),
        },
        "Buena": {
            "latencia_ms":       (20,  60),
            "jitter_ms":         (5,   15),
            "rtt_ms":            (40,  100),
            "perdida_paquetes":  (0.5, 2.0),
            "throughput_mbps":   (40,  80),
            "ancho_banda_mbps":  (50,  90),
            "intensidad_senal":  (-80, -65),
            "velocidad_descarga": (30, 70),
            "velocidad_subida":  (10,  20),
            "congestion_red":    (10,  35),
        },
        "Regular": {
            "latencia_ms":       (60,  150),
            "jitter_ms":         (15,  35),
            "rtt_ms":            (100, 250),
            "perdida_paquetes":  (2.0, 5.0),
            "throughput_mbps":   (10,  40),
            "ancho_banda_mbps":  (15,  50),
            "intensidad_senal":  (-95, -80),
            "velocidad_descarga": (5,  30),
            "velocidad_subida":  (2,   10),
            "congestion_red":    (35,  65),
        },
        "Mala": {
            "latencia_ms":       (150, 500),
            "jitter_ms":         (35,  100),
            "rtt_ms":            (250, 800),
            "perdida_paquetes":  (5.0, 20.0),
            "throughput_mbps":   (0,   10),
            "ancho_banda_mbps":  (0,   15),
            "intensidad_senal":  (-110, -95),
            "velocidad_descarga": (0,  5),
            "velocidad_subida":  (0,   2),
            "congestion_red":    (65,  100),
        },
    }

    samples_per_class = n_samples // 4

    for categoria, rangos in categories.items():
        for _ in range(samples_per_class):
            row = {}
            for col, (low, high) in rangos.items():
                row[col] = round(np.random.uniform(low, high), 2)
            row["calidad_red"] = categoria
            data.append(row)

    df = pd.DataFrame(data).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_qos_dataset(1000)
    output_path = os.path.join(os.path.dirname(__file__), "qos_datos_ejemplo.xlsx")
    df.to_excel(output_path, index=False)
    print(f"Dataset generado: {output_path}")
    print(f"Shape: {df.shape}")
    print(df["calidad_red"].value_counts())
