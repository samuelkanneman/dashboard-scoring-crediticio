# 📊 Tablero de Scoring Crediticio

Este repositorio contiene una aplicación construida con **Streamlit** diseñada para analizar, comparar y visualizar el rendimiento de diferentes modelos de *Scoring* Crediticio. 

> [!WARNING]
> **Aviso Importante sobre los Datos:**  
> Todos los datos utilizados y contenidos en este repositorio (`dataset_procesado.csv`) son **datos sintéticos generados artificialmente** para fines de demostración, pruebas y desarrollo. **No representan información de clientes reales** ni datos verídicos de ninguna entidad financiera.

## 🚀 Características Principales

- **Gestión Multimodelo:** Permite la evaluación simultánea de múltiples modelos de scoring (por ejemplo: `ML_Score`, `VZ_score_3t`, `NS_scoret`, `Score_Alt`, etc.).
- **Análisis por Deciles:** Calcula deciles dinámicos (ajustándose al tamaño de la muestra filtrada) o mantiene los deciles globales proporcionales al total de la población.
- **Métricas de Riesgo Preadaptadas:**
  - **Mora (+60 días):** Análisis del porcentaje de deudores en estado irregular.
  - **Curvas KS (Kolmogorov-Smirnov):** Para medir la máxima separación entre las distribuciones de clientes "buenos" y "malos".
  - **Curva ROC y AUC:** Integración rápida con `scikit-learn` para evaluar el nivel de discriminación de la herramienta de riesgo.
- **Filtros Flexibles:** Analiza segmentando por períodos de originación (cosechas/cohortes) y por tipo de cliente (Nuevos vs Existentes).
- **Visualización Interactiva:** Utiliza `plotly` y `seaborn` para gráficos dinámicos de distribuciones y curvas asintóticas.

## 📁 Estructura del Repositorio

- `scoring_dashboard.py`: Archivo principal que contiene toda la lógica y la interfaz de la aplicación de Streamlit.
- `dataset_procesado.csv`: Archivo de datos sintéticos de muestra con las inferencias de los modelos y variables de seguimiento (mora, periodos, score real).
- `requirements.txt`: Dependencias de Python requeridas para ejecutar la aplicación.
- `.devcontainer/`: Configuración del entorno de desarrollo estandarizado para VS Code / devcontainers o GitHub Codespaces.

## ⚙️ Instalación y Uso Local

1. **Clonar este repositorio:**
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd dashboard-scoring-crediticio-main
   ```

2. **Instalar las dependencias:**
   Se recomienda crear un entorno virtual previamente.
   ```bash
   pip install -r requirements.txt
   ```

3. **Ejecutar el Tablero:**
   ```bash
   streamlit run scoring_dashboard.py
   ```

4. El panel abrirá automáticamente tu navegador web en la dirección local por defecto `http://localhost:8501`.

## 🛠️ Tecnologías Utilizadas

- **[Streamlit](https://streamlit.io/):** Framework para el Frontend.
- **[Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/):** Procesamiento y cálculos de datos estadísticos.
- **[Plotly](https://plotly.com/python/):** Gráficos y visualizaciones interactivas.
- **[Scikit-Learn](https://scikit-learn.org/):** Cálculos estadísticos y métricas de machine learning. 
