# Plataforma Macro-Predictiva & Econométrica AR

Una plataforma analítica e interactiva desarrollada en **Python** y **Streamlit** diseñada para el análisis, diagnóstico y pronóstico de variables macroeconómicas clave de Argentina: **Dólar, Inflación (IPC) e Índice Merval**.

La aplicación integra desde estadística descriptiva y pruebas de estacionariedad hasta algoritmos de Machine Learning y Redes Neuronales Profundas (LSTM) para predecir el comportamiento futuro de estas series temporales.

---

## Características Principales y Módulos

El proyecto está estructurado en módulos independientes para facilitar el análisis escalonado de las series económicas:

### 1. Exploración de Datos (EDA)
- Visualización interactiva de series históricas (Base 100 = Dic 2016).
- Cálculo de estadísticas descriptivas (Media, Desvío, Min, Max, Volatilidad).
- Matrices de correlación de Pearson interactivas.

### 2. Análisis Econométrico Tradicional
- **Pruebas de Estacionariedad:** Evaluación rigurosa mediante pruebas Dickey-Fuller Aumentada (ADF) y KPSS.
- **Modelado ARIMA (p,d,q):** Ajuste de modelos autorregresivos integrados de media móvil con intervalos de confianza del 95%.
- **Suavizado Exponencial:** Proyecciones mediante el modelo Holt-Winters.

### 3. Machine Learning Benchmark
- Competencia automática de modelos de Machine Learning (Ensamble) para predicción de series usando ventanas deslizantes y rezagos (lags).
- Modelos evaluados: **Random Forest, XGBoost, LightGBM, Gradient Boosting y SVR**.
- Selección del mejor modelo en base a la métrica RMSE.

### 4. Deep Learning (Redes Neuronales LSTM)
- Implementación de Redes Neuronales Recurrentes (RNN - LSTM) utilizando **PyTorch**.
- Modelado secuencial para capturar patrones no lineales complejos en la evolución macroeconómica.
- Evaluación Out-of-Sample, curvas de aprendizaje (Loss MSE) y pronóstico a múltiples horizontes.

### 5. Descubrimiento de Patrones y Causalidad
- **Causalidad de Granger:** Análisis estadístico para determinar si una variable (ej. Dólar) causa los movimientos de otra (ej. IPC).
- **Clustering (K-Means):** Identificación no supervisada de "regímenes económicos" o fases de alta/baja volatilidad.
- **Reducción de Dimensionalidad:** Aplicación de PCA para variables múltiples (como IPCs regionales).

### 6. Análisis Estadístico y Dashboard Integral (`app-statistics.py`)
- Módulo enfocado en la estadística de riesgo, percentiles y asimetría (Skewness/Kurtosis).
- **Pass-Through:** Análisis del traspaso a precios con gráficos de rezagos (lags) del impacto del dólar en el IPC.
- Proyección multivariable a corto plazo utilizando modelos de Vectores Autorregresivos (**VAR**).
- Exportación automática de tablas a **PDF** y generación de reportes listos para imprimir.

---

## Arquitectura del Proyecto

```text
APP_ECONOMIA_ARG/
│
├── data/
│   └── dolar-index.xlsx           # Dataset histórico unificado
│
├── .streamlit/
│   └── config.toml                # Configuración de tema (Matrix/Dark theme)
│
├── app.py                         # Aplicación principal (Navegación e integración)
├── app-statistics.py              # Dashboard estadístico, VAR y PDF export
├── data_loader.py                 # Módulo de ingesta y limpieza de datos (Pandas)
├── eda.py                         # Módulo de Análisis Exploratorio (Plotly)
├── models_stat.py                 # Econometría: ADF, KPSS, ARIMA, Holt-Winters
├── models_ml.py                   # Ensamble ML: XGBoost, LightGBM, RF
├── models_nn.py                   # Deep Learning: Arquitectura PyTorch LSTM
├── pattern_discovery.py           # Granger Causality, PCA, K-Means
│
└── requirements.txt               # Dependencias del entorno
```

---

##  Instalación y Uso

1. **Clonar el repositorio:**
   bash
   git clone https://github.com/wgekko/app_economia_arg.git
   
   cd app_economia_arg


3. **Crear un entorno virtual (Recomendado):**
   bash
   python -m venv env
   source env/bin/activate  # En Windows: env\Scripts ctivate
   

4. **Instalar las dependencias:**
   bash
   pip install -r requirements.txt
   

5. **Ejecutar la plataforma:**
   Para iniciar el dashboard principal:
   bash
   streamlit run app.py   
   *(Si configuraste `app-statistics.py` dentro de la carpeta `pages/`, aparecerá automáticamente en el menú lateral).*

---

## Stack Tecnológico

- **Frontend & UI:** Streamlit, Streamlit Components.
- **Data Manipulation:** Pandas, NumPy.
- **Visualización:** Plotly Express, Plotly Graph Objects.
- **Machine Learning:** Scikit-Learn, XGBoost, LightGBM, CatBoost.
- **Deep Learning:** PyTorch.
- **Econometría y Estadística:** Statsmodels, SciPy.
- **Exportación:** FPDF (Generación de reportes en PDF).

---
*Desarrollado para la investigación y el análisis predictivo de la coyuntura económica Argentina.*


video demo



https://github.com/user-attachments/assets/93a0a109-9d99-47bf-9fd1-486804a62b1d



