import numpy as np
import pandas as pd
import os

np.random.seed(42)



RANGOS_BASE = {
    "Excelente": {
        "latencia_ms":       (5,   35),   
        "jitter_ms":         (1,   10),    
        "rtt_ms":            (10,  70),   
        "perdida_paquetes":  (0.0, 1.0),   
        "throughput_mbps":   (60,  150),  
        "ancho_banda_mbps":  (70,  200),
        "intensidad_senal":  (-70, -50),
        "velocidad_descarga": (55, 120),
        "velocidad_subida":  (15,  50),
        "congestion_red":    (0,   20),    
    },
    "Buena": {
        "latencia_ms":       (15,  80),    
        "jitter_ms":         (4,   22),
        "rtt_ms":            (30,  130),
        "perdida_paquetes":  (0.3, 3.5),
        "throughput_mbps":   (25,  85),
        "ancho_banda_mbps":  (30,  95),
        "intensidad_senal":  (-85, -62),
        "velocidad_descarga": (20, 75),
        "velocidad_subida":  (6,   22),
        "congestion_red":    (8,   45),
    },
    "Regular": {
        "latencia_ms":       (50,  200),   
        "jitter_ms":         (12,  50),
        "rtt_ms":            (90,  350),
        "perdida_paquetes":  (1.5, 8.0),
        "throughput_mbps":   (5,   45),
        "ancho_banda_mbps":  (8,   55),
        "intensidad_senal":  (-100, -78),
        "velocidad_descarga": (3,  38),
        "velocidad_subida":  (1,   12),
        "congestion_red":    (30,  75),
    },
    "Mala": {
        "latencia_ms":       (120, 600),   
        "jitter_ms":         (30,  120),
        "rtt_ms":            (200, 900),
        "perdida_paquetes":  (5.0, 25.0),
        "throughput_mbps":   (0,   18),
        "ancho_banda_mbps":  (0,   20),
        "intensidad_senal":  (-115, -95),
        "velocidad_descarga": (0,  10),
        "velocidad_subida":  (0,   3),
        "congestion_red":    (60,  100),
    },
}

DISTRIBUCION = {
    "Excelente": 0.20,
    "Buena":     0.40,
    "Regular":   0.28,
    "Mala":      0.12,
}

#en este se aplica el ruido gaussiano 

def _aplicar_ruido(valor: float, porcentaje_ruido: float = 0.12) -> float:
 
    std = abs(valor) * porcentaje_ruido
    ruido = np.random.normal(0, std)
    return max(0.0, valor + ruido)


def _aplicar_correlacion(fila: dict, categoria: str) -> dict:
#se aplica la correlacion entre variables  


    factor_congestion = fila["congestion_red"] / 100.0  


    fila["latencia_ms"] *= (1 + factor_congestion * 0.4)


    fila["throughput_mbps"] *= (1 - factor_congestion * 0.35)
    fila["throughput_mbps"] = max(0.1, fila["throughput_mbps"])


    factor_jitter = min(fila["jitter_ms"] / 100.0, 1.0)
    fila["perdida_paquetes"] *= (1 + factor_jitter * 0.3)

    factor_senal = (fila["intensidad_senal"] + 115) / 65.0  
    factor_senal = max(0.1, min(factor_senal, 1.0))
    fila["velocidad_descarga"] *= factor_senal
    fila["velocidad_subida"]   *= factor_senal

    return fila


def _generar_anomalia(fila: dict) -> dict:

    if np.random.random() < 0.06:   # 6% de probabilidad de anomalía
        tipo = np.random.choice(["pico_latencia", "perdida_rafaga", "congestion_pico"])

        if tipo == "pico_latencia":
            fila["latencia_ms"]  *= np.random.uniform(1.5, 3.0)
            fila["rtt_ms"]       *= np.random.uniform(1.5, 2.5)

        elif tipo == "perdida_rafaga":
            fila["perdida_paquetes"] += np.random.uniform(2.0, 8.0)
            fila["jitter_ms"]        *= np.random.uniform(1.3, 2.0)

        elif tipo == "congestion_pico":
            fila["congestion_red"]   = min(100, fila["congestion_red"] * np.random.uniform(1.5, 2.5))
            fila["throughput_mbps"]  *= np.random.uniform(0.3, 0.7)

    return fila


def _redondear_fila(fila: dict) -> dict:
    return {k: round(float(v), 2) for k, v in fila.items()}


def generate_qos_dataset(n_samples: int = 1200) -> pd.DataFrame:
#esta genra un dataset de prueba aplicando las tecnicas anteriormente mencionadas 
    data = []

    for categoria, proporcion in DISTRIBUCION.items():
        n_cat = int(n_samples * proporcion)
        rangos = RANGOS_BASE[categoria]

        for _ in range(n_cat):
            fila = {}


            for col, (low, high) in rangos.items():
                fila[col] = np.random.uniform(low, high)


            for col in fila:
                fila[col] = _aplicar_ruido(fila[col], porcentaje_ruido=0.12)


            fila = _aplicar_correlacion(fila, categoria)


            fila = _generar_anomalia(fila)


            fila = _redondear_fila(fila)
            fila["calidad_red"] = categoria
            data.append(fila)


    data = _agregar_casos_frontera(data, int(n_samples * 0.08))

    df = pd.DataFrame(data).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def _agregar_casos_frontera(data: list, n_frontera: int) -> list:
    fronteras = [
        ("Excelente", "Buena"),
        ("Buena",     "Regular"),
        ("Regular",   "Mala"),
    ]

    n_por_frontera = n_frontera // len(fronteras)

    for clase_a, clase_b in fronteras:
        rangos_a = RANGOS_BASE[clase_a]
        rangos_b = RANGOS_BASE[clase_b]

        for _ in range(n_por_frontera):
            fila = {}

            for col in rangos_a:

                if np.random.random() < 0.5:
                    low, high = rangos_a[col]

                    valor = np.random.uniform(
                        low + (high - low) * 0.6,  
                        high
                    )
                else:
                    low, high = rangos_b[col]

                    valor = np.random.uniform(
                        low,
                        low + (high - low) * 0.4   
                    )

                fila[col] = _aplicar_ruido(valor, porcentaje_ruido=0.08)

            fila = _redondear_fila(fila)


            fila["calidad_red"] = clase_b
            data.append(fila)

    return data


def get_dataset_stats(df: pd.DataFrame) -> None:

    print("=" * 55)
    print("DATASET QoS REALISTA — ESTADÍSTICAS")
    print("=" * 55)
    print(f"Total muestras : {len(df)}")
    print(f"Variables      : {len(df.columns) - 1}")
    print()
    print("Distribución de clases:")
    dist = df["calidad_red"].value_counts()
    for clase, count in dist.items():
        pct = count / len(df) * 100
        barra = "█" * int(pct / 2)
        print(f"  {clase:<12} {count:>4} ({pct:5.1f}%)  {barra}")
    print()
    print("Rangos reales de latencia_ms por clase:")
    for clase in ["Excelente", "Buena", "Regular", "Mala"]:
        subset = df[df["calidad_red"] == clase]["latencia_ms"]
        print(f"  {clase:<12} min={subset.min():6.1f}  max={subset.max():7.1f}  "
              f"media={subset.mean():6.1f}")
    print()
    print("Solapamiento detectado (latencia_ms):")
    exc_max = df[df["calidad_red"] == "Excelente"]["latencia_ms"].max()
    bue_min = df[df["calidad_red"] == "Buena"]["latencia_ms"].min()
    print(f"  Excelente max={exc_max:.1f}ms  |  Buena min={bue_min:.1f}ms")
    if exc_max > bue_min:
        print("  ✅ Solapamiento confirmado → modelo NO puede separar perfectamente")
    else:
        print("  ⚠️  Sin solapamiento detectado")
    print("=" * 55)


if __name__ == "__main__":
    df = generate_qos_dataset(1200)
    output_path = os.path.join(os.path.dirname(__file__), "qos_datos_ejemplo.xlsx")
    df.to_excel(output_path, index=False)
    get_dataset_stats(df)
    print(f"\nArchivo guardado: {output_path}")