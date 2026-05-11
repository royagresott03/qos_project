# 📡 QoS ML Analyzer
### Sistema de predicción de calidad de servicio en redes móviles mediante Machine Learning

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red?logo=streamlit)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4%2B-orange)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Descripción

**QoS ML Analyzer** es un sistema académico/profesional que aplica técnicas de
Machine Learning para clasificar automáticamente la calidad de servicio (QoS)
en redes móviles a partir de métricas como latencia, jitter, pérdida de
paquetes, throughput, entre otras.

El sistema clasifica la calidad en cuatro categorías:

| Clase      | Descripción                                        |
|------------|----------------------------------------------------|
| Excelente  | Latencia <20ms, pérdida <0.5%, throughput >80 Mbps |
| Buena      | Experiencia satisfactoria para la mayoría de usos  |
| Regular    | Degradación notable, videollamadas con problemas   |
| Mala       | Servicio inutilizable, alta pérdida de paquetes    |

---

## 🏗️ Estructura del Proyecto

```
qos_project/
│
├── data/
│   └── sample_qos_dataset.py     # Generador de datos sintéticos de ejemplo
│
├── models/
│   └── best_model.pkl            # Mejor modelo entrenado (generado)
│
├── reports/
│   ├── predicciones_qos.xlsx     # Predicciones exportadas
│   ├── reporte_metricas.xlsx     # Comparación de modelos
│   ├── classification_report.txt # Reporte de texto
│   └── resultados_qos.zip        # Todos los artefactos
│
├── graphs/
│   ├── heatmap_correlacion.png
│   ├── distribucion_clases.png
│   ├── histogramas_variables.png
│   ├── matriz_confusion.png
│   ├── importancia_variables.png
│   └── comparacion_modelos.png
│
├── app/
│   ├── streamlit_app.py          # Interfaz gráfica principal
│   ├── data_loader.py            # Carga y validación de datos
│   ├── preprocessor.py           # Pipeline de preprocesamiento
│   ├── models.py                 # Entrenamiento y comparación de modelos
│   ├── visualizations.py         # Gráficas profesionales
│   └── exporter.py               # Exportación de resultados
│
├── main.py                       # Entrada CLI (sin interfaz)
├── requirements.txt
└── README.md
```

---

## 🚀 Instalación y Ejecución

### 1. Clonar y crear entorno virtual

```bash
git clone https://github.com/tu-usuario/qos-ml-analyzer.git
cd qos-ml-analyzer

python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Ejecutar interfaz gráfica (recomendado)

```bash
streamlit run app/streamlit_app.py
```

Abre tu navegador en `http://localhost:8501`

### 4. Ejecutar en modo consola (CLI)

```bash
# Generar dataset de ejemplo
python main.py --generate-sample

# Analizar un archivo propio
python main.py --file data/mis_datos.xlsx --target calidad_red

# Opciones adicionales
python main.py --file data/mis_datos.xlsx \
               --target calidad_red \
               --test-size 0.25 \
               --random-state 123 \
               --k-features 8
```

---

## 📊 Variables QoS Soportadas

El sistema detecta automáticamente columnas numéricas. Las principales métricas esperadas son:

| Variable              | Unidad    | Descripción                          |
|-----------------------|-----------|--------------------------------------|
| `latencia_ms`         | ms        | Tiempo de ida del paquete            |
| `jitter_ms`           | ms        | Variación en la latencia             |
| `rtt_ms`              | ms        | Round-Trip Time                      |
| `perdida_paquetes`    | %         | Porcentaje de paquetes perdidos      |
| `throughput_mbps`     | Mbps      | Tasa de transferencia efectiva       |
| `ancho_banda_mbps`    | Mbps      | Capacidad del canal                  |
| `intensidad_senal`    | dBm       | Potencia de señal recibida           |
| `velocidad_descarga`  | Mbps      | Velocidad de bajada                  |
| `velocidad_subida`    | Mbps      | Velocidad de subida                  |
| `congestion_red`      | %         | Nivel de congestión de la red        |

La columna objetivo puede llamarse: `calidad_red`, `quality`, `label`, `clase`, `categoria`.

---

## 🤖 Modelos Implementados

| Modelo         | Tipo          | Ventaja principal                  |
|----------------|---------------|------------------------------------|
| Random Forest  | Ensamble      | Robusto, maneja outliers bien      |
| Decision Tree  | Árbol         | Interpretable, rápido              |
| KNN            | Instancia     | No paramétrico, simple             |
| SVM            | Kernel        | Excelente en alta dimensionalidad  |
| XGBoost*       | Gradient Boost| Estado del arte, muy preciso       |

\* Requiere `xgboost` instalado.

El sistema selecciona **automáticamente** el modelo con mayor accuracy.

---

## 🔄 Pipeline de Preprocesamiento

```
Dataset crudo
    │
    ▼
Eliminar duplicados
    │
    ▼
Imputación de nulos (mediana/moda)
    │
    ▼
Codificación one-hot (variables categóricas)
    │
    ▼
Eliminación de outliers extremos (z-score > 3)
    │
    ▼
Codificación del target (LabelEncoder)
    │
    ▼
Selección de características (SelectKBest + F-ANOVA)
    │
    ▼
División Train/Test estratificada (80/20)
    │
    ▼
Normalización StandardScaler (fit solo en train)
    │
    ▼
X_train, X_test, y_train, y_test
```

---

## 📈 Métricas de Evaluación

- **Accuracy** — Porcentaje de predicciones correctas
- **Precision** — De las predicciones positivas, cuántas son correctas
- **Recall** — De los casos reales, cuántos se detectaron
- **F1-Score** — Media armónica de precision y recall
- **Matriz de confusión** — Visualización detallada de errores por clase

---

## 📁 Formato del Dataset de Entrada

El archivo Excel/CSV debe tener **una fila de encabezado** con los nombres de las variables y **al menos 20 filas** de datos. Ejemplo mínimo:

```csv
latencia_ms,jitter_ms,perdida_paquetes,throughput_mbps,calidad_red
15.2,2.1,0.1,95.3,Excelente
120.5,25.3,4.5,18.2,Regular
```

---

## 🎓 Contexto Académico

Este proyecto está orientado a investigación en:

- Gestión de calidad en redes 4G/5G
- Clasificación automática de QoS con ML
- Análisis de KPIs en telecomunicaciones móviles
- Comparación de algoritmos de clasificación para datos de red

### Referencias

- ITU-T Y.1541 — Network performance objectives for IP-based services
- 3GPP TS 22.261 — Service requirements for the 5G system
- Pedregosa et al. (2011) — Scikit-learn: Machine Learning in Python

---

## 📝 Licencia

MIT License — libre uso académico y comercial con atribución.
