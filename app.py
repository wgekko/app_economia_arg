import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go  
from models_statistics import run_statistics_app
from data_loader import DataLoader
from eda import ExploratoryDataAnalysis
from models_stat import StatisticalModels
from models_ml import MLModelBenchmark
from models_nn import NeuralNetworkForecaster
from pattern_discovery import PatternDiscovery


st.set_page_config(page_title="Plataforma Predictiva Macro AR", layout="wide", page_icon=":material/finance_mode:")

def main():
    st.subheader(":material/move_group: Plataforma Macro-Predictiva & Econométrica")
    
    loader = DataLoader("data/dolar-index.xlsx")
    df = loader.load_and_clean_data()

    if df.empty:
        st.stop()

    menu = st.sidebar.radio(
        "Seleccione Módulo Analítico", 
        [
            "1. Análisis Estadístico-Dashboard",
            "2. Exploración & EDA", 
            "3. Econometría (ADF/ARIMA/Holt-Winters)", 
            "4. Machine Learning Benchmark", 
            "5. Redes Neuronales (LSTM)",
            "6. Descubrimiento de Patrones & Causalidad"
        ]
    )

    if menu == "1. Análisis Estadístico-Dashboard":
        run_statistics_app()


    elif menu == "2. Exploración & EDA":
        st.subheader("Análisis Exploratorio")

        st.subheader("tabla de las variables expresado con formato de indice")
        st.info("la tabla parte de valor indice 100 en Dic-2016, y los valores se ajustan partiendo de esa fecha base para hacer los repectivos calculos de proyecciones y predicciones")
        st.info("los valores que se publican no los valores de las variables nominales sino loa valores ajustados de acuerdo a cada variable que ancla de ajuste")
        dfindex= pd.read_excel("data/dolar-index.xlsx")
        st.dataframe(dfindex, width='stretch')
        eda = ExploratoryDataAnalysis(df)
        eda.show_descriptive_stats()

    elif menu == "3. Econometría (ADF/ARIMA/Holt-Winters)":
        st.subheader(" Análisis Econométrico Tradicional & Diagnóstico de Estacionariedad")
        st.markdown("Evaluación estadística de series temporales con modelos ARIMA y Suavizado Exponencial.")
        st.info("los valores que se publican no los valores de las variables nominales sino loa valores ajustados de acuerdo a cada variable que ancla de ajuste")

        # Selección de parámetros principales
        c1, c2, c3 = st.columns(3)
        with c1:
            target = st.selectbox("Seleccione Variable Objetivo", df.columns, index=df.columns.get_loc('dólar-vta-real') if 'dólar-vta-real' in df.columns else 0)
        with c2:
            forecast_horizon = st.selectbox("Horizonte de Pronóstico (Meses)", [1, 3, 6, 12], index=2)
        with c3:
            hist_window = st.selectbox("Período Histórico Visualizado", ["Último Año (12 Meses)", "Últimos 2 Años (24 Meses)", "Histórico Completo"], index=0)

        stat = StatisticalModels(df[target])
        stat_results = stat.test_stationarity()

        # --- SECCIÓN 1: PRUEBAS DE ESTACIONARIEDAD ---
        st.subheader("Pruebas de Estacionariedad (ADF & KPSS)")
        
        # Tarjeta de Diagnóstico Principal
        if stat_results["ADF_Es_Estacionaria"] and stat_results["KPSS_Es_Estacionaria"]:
            st.success(f"🟢 **Diagnóstico:** {stat_results['Diagnostico']}")
        else:
            st.warning(f"🟡 **Diagnóstico:** {stat_results['Diagnostico']}")

        col_adf, col_kpss = st.columns(2)
        with col_adf:
            st.markdown("#### 🔹 Prueba Dickey-Fuller Aumentada (ADF)")
            st.caption("Hipótesis Nula ($H_0$): La serie posee raíz unitaria (No es estacionaria).")
            st.metric("Estadístico ADF", f"{stat_results['ADF_Stat']}")
            st.metric("p-valor ADF", f"{stat_results['ADF_pvalue']}", delta="Estacionaria (p < 0.05)" if stat_results["ADF_Es_Estacionaria"] else "No Estacionaria (p >= 0.05)")

        with col_kpss:
            st.markdown("#### 🔹 Prueba Kwiatkowski-Phillips-Schmidt-Shin (KPSS)")
            st.caption("Hipótesis Nula ($H_0$): La serie es estacionaria alrededor de una constante.")
            st.metric("Estadístico KPSS", f"{stat_results['KPSS_Stat']}")
            st.metric("p-valor KPSS", f"{stat_results['KPSS_pvalue']}", delta="Estacionaria (p > 0.05)" if stat_results["KPSS_Es_Estacionaria"] else "No Estacionaria (p <= 0.05)")

        st.divider()

        # --- SECCIÓN 2: MODELADO Y PRONÓSTICO ---
        st.subheader("Pronóstico Econométrico y Evaluación")
        st.info("los valores que se publican no los valores de las variables nominales sino loa valores ajustados de acuerdo a cada variable que ancla de ajuste")

        modelo_tipo = st.radio("Seleccione Modelo de Series Temporales", ["ARIMA(p, d, q)", "Holt-Winters Exponencial"], horizontal=True)

        if modelo_tipo == "ARIMA(p, d, q)":
            p_val = st.number_input("Parámetro AR (p - Autorregresivo)", min_value=0, max_value=5, value=1)
            d_val = st.number_input("Parámetro I (d - Diferenciación)", min_value=0, max_value=2, value=1)
            q_val = st.number_input("Parámetro MA (q - Media Móvil)", min_value=0, max_value=5, value=1)

            if st.button("Ejecutar Pronóstico ARIMA", width='stretch'):
                with st.spinner("Ajustando modelo ARIMA..."):
                    model, forecast_df, metrics = stat.fit_arima(order=(p_val, d_val, q_val), steps=forecast_horizon)
                    
                    # Filtrar período histórico para el gráfico según selección del usuario
                    series_hist = df[target].dropna()
                    if hist_window == "Último Año (12 Meses)":
                        df_plot = series_hist.tail(12)
                    elif hist_window == "Últimos 2 Años (24 Meses)":
                        df_plot = series_hist.tail(24)
                    else:
                        df_plot = series_hist

                    # Construcción del gráfico en Plotly
                    fig = go.Figure()

                    # 1. Serie histórica
                    fig.add_trace(go.Scatter(
                        x=df_plot.index, 
                        y=df_plot.values, 
                        mode='lines+markers', 
                        name='Histórico Real',
                        line=dict(color='#1f77b4', width=2.5)
                    ))

                    # 2. Serie del pronóstico (conectada al último dato histórico)
                    forecast_x = [df_plot.index[-1]] + list(forecast_df.index)
                    forecast_y = [df_plot.iloc[-1]] + list(forecast_df['pred'])

                    fig.add_trace(go.Scatter(
                        x=forecast_x, 
                        y=forecast_y, 
                        mode='lines+markers', 
                        name=f'Pronóstico ARIMA({p_val},{d_val},{q_val})',
                        line=dict(color='#ff7f0e', width=3, dash='dash')
                    ))

                    # 3. Banda de Confianza (95%)
                    upper_x = list(forecast_df.index)
                    upper_y = list(forecast_df['upper'])
                    lower_y = list(forecast_df['lower'])

                    fig.add_trace(go.Scatter(
                        x=upper_x + upper_x[::-1],
                        y=upper_y + lower_y[::-1],
                        fill='toself',
                        fillcolor='rgba(255, 127, 14, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo="skip",
                        name='Intervalo de Confianza 95%'
                    ))

                    fig.update_layout(
                        title=f"Proyección ARIMA a {forecast_horizon} Meses - {target} ({hist_window})",
                        xaxis_title="Fecha",
                        yaxis_title="Índice / Valor",
                        hovermode="x unified"
                    )

                    st.plotly_chart(fig, width='stretch')

                    # Despliegue de métricas y tabla de datos
                    m_col1, m_col2 = st.columns(2)
                    with m_col1:
                        st.markdown("#### Criterios de Calidad del Ajuste")
                        st.json(metrics)

                    with m_col2:
                        st.markdown("#### Tabla de Valores Proyectados")
                        st.dataframe(forecast_df.style.format("{:,.2f}"), width='stretch')

        else: # Holt-Winters
            if st.button("Ejecutar Pronóstico Holt-Winters", width='stretch'):
                with st.spinner("Ajustando modelo Holt-Winters..."):
                    model, forecast_df, metrics = stat.fit_holt_winters(steps=forecast_horizon)
                    
                    series_hist = df[target].dropna()
                    if hist_window == "Último Año (12 Meses)":
                        df_plot = series_hist.tail(12)
                    elif hist_window == "Últimos 2 Años (24 Meses)":
                        df_plot = series_hist.tail(24)
                    else:
                        df_plot = series_hist

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_plot.index, y=df_plot.values, 
                        mode='lines+markers', name='Histórico Real',
                        line=dict(color='#1f77b4', width=2.5)
                    ))

                    forecast_x = [df_plot.index[-1]] + list(forecast_df.index)
                    forecast_y = [df_plot.iloc[-1]] + list(forecast_df['pred'])

                    fig.add_trace(go.Scatter(
                        x=forecast_x, y=forecast_y, 
                        mode='lines+markers', name='Pronóstico Holt-Winters',
                        line=dict(color='#2ca02c', width=3, dash='dash')
                    ))

                    fig.update_layout(
                        title=f"Proyección Holt-Winters a {forecast_horizon} Meses - {target}",
                        hovermode="x unified"
                    )

                    st.plotly_chart(fig, width='stretch')
                    st.dataframe(forecast_df.style.format("{:,.2f}"), width='stretch')

    elif menu == "4. Machine Learning Benchmark":
        st.subheader("Ensamble & Competencia de Modelos ML")
        st.info("los valores que se publican no los valores de las variables nominales sino loa valores ajustados de acuerdo a cada variable que ancla de ajuste")
        target = st.selectbox("Variable Objetivo", df.columns)
        lags = st.slider("Lags (Rezagos)", 1, 12, 3)
        
        if st.button("Ejecutar Benchmark"):
            benchmark = MLModelBenchmark(df, target, lags)
            results_df, preds_dict, y_test, best_model = benchmark.evaluate_all()
            
            st.success(f"El mejor modelo es: **{best_model}**")
            st.dataframe(results_df)

    elif menu == "5. Redes Neuronales (LSTM)":
        st.subheader("Deep Learning: Redes Neuronales Recurrentes (LSTM)")
        st.markdown("Modelado secuencial avanzado de series temporales macroeconómicas.")
        st.info("los valores que se publican no los valores de las variables nominales sino loa valores ajustados de acuerdo a cada variable que ancla de ajuste")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            target = st.selectbox("Variable Objetivo", df.columns, index=df.columns.get_loc('dólar-vta-real') if 'dólar-vta-real' in df.columns else 0)
        with col2:
            horizon = st.selectbox("Horizonte de Pronóstico (Meses)", [1, 3, 6, 12], index=2)
        with col3:
            epochs = st.slider("Épocas de Entrenamiento", min_value=30, max_value=300, value=100, step=10)

        if st.button("Entrenar y Proyectar con LSTM", width='stretch'):
            with st.spinner("Entrenando Red Neuronal Recurrente LSTM..."):
                forecaster = NeuralNetworkForecaster(df[target], seq_length=6)
                df_loss, df_eval, df_future, metrics = forecaster.train_and_forecast(epochs=epochs, horizon=horizon)
                
                # --- CALCULOS EXECUTIVE SUMMARY ---
                last_real = df[target].dropna().iloc[-1]
                last_forecast = df_future['Pronóstico LSTM'].iloc[-1]
                pct_change = ((last_forecast - last_real) / last_real) * 100
                trend_label = "🟢 Alcista" if pct_change > 1 else ("🔴 Bajista" if pct_change < -1 else "🟡 Estacionaria")

                # --- TARJETAS KPI ---
                st.markdown("### Resumen de Proyección")
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric("Último Valor Real", f"{last_real:,.2f}")
                kpi2.metric(f"Proyección ({horizon}m)", f"{last_forecast:,.2f}", delta=f"{pct_change:+.2f}%")
                kpi3.metric("Tendencia Esperada", trend_label)
                kpi4.metric("Error Relativo (MAPE)", f"{metrics['MAPE (%)']}%")

                st.divider()

                # --- PESTAÑAS INTERACTIVAS ---
                tab1, tab2, tab3, tab4 = st.tabs([
                    "Proyección Futura", 
                    "Evaluación Out-of-Sample", 
                    "Curvas de Aprendizaje (Loss)", 
                    "Métricas & Diagnóstico"
                ])

                with tab1:
                    st.subheader(f"Proyección a {horizon} Meses para {target}")
                    fig_fut = go.Figure()
                    
                    # Serie Histórica Reciente
                    df_hist_tail = df[target].dropna().tail(24)
                    fig_fut.add_trace(go.Scatter(x=df_hist_tail.index, y=df_hist_tail, mode='lines+markers', name='Histórico Real'))
                    
                    # Serie Proyectada (unida al último punto histórico)
                    last_hist_date = df_hist_tail.index[-1]
                    future_x = [last_hist_date] + list(df_future.index)
                    future_y = [df_hist_tail.iloc[-1]] + list(df_future['Pronóstico LSTM'])
                    
                    fig_fut.add_trace(go.Scatter(
                        x=future_x, y=future_y, 
                        mode='lines+markers', 
                        name='Pronóstico LSTM', 
                        line=dict(dash='dash', color='orange', width=3)
                    ))
                    
                    fig_fut.update_layout(hovermode="x unified", title="Evolución Histórica Reciente + Proyección Futura")
                    st.plotly_chart(fig_fut, width='stretch')

                with tab2:
                    st.subheader("Ajuste Histórico en Datos de Prueba (Out-of-Sample)")
                    st.caption("Muestra la capacidad de la red para predecir períodos históricos no vistos durante el entrenamiento.")
                    
                    fig_eval = px.line(df_eval, title="Comparativa: Valores Reales vs Predicciones LSTM")
                    fig_eval.update_layout(hovermode="x unified")
                    st.plotly_chart(fig_eval, width='stretch')

                with tab3:
                    st.subheader("Evolución de la Función de Pérdida (Loss)")
                    st.caption("Evolución del Error Cuadrático Medio (MSE) en entrenamiento y validación.")
                    
                    fig_loss = px.line(df_loss, labels={"value": "MSE Loss", "index": "Época"}, title="Curvas de Aprendizaje (Training vs Validation)")
                    fig_loss.update_layout(hovermode="x unified")
                    st.plotly_chart(fig_loss, width='stretch')

                with tab4:
                    st.subheader("Métricas Estadísticas del Modelo")
                    m_col1, m_col2 = st.columns(2)
                    
                    with m_col1:
                        st.dataframe(pd.DataFrame([metrics]).T.rename(columns={0: "Valor"}), width='stretch')
                    
                    with m_col2:
                        st.markdown(f"""
                        **Interpretación de Métricas:**
                        * **MAE (Error Absoluto Medio):** En promedio, las predicciones se desvían **{metrics['MAE']}** unidades de los valores reales.
                        * **RMSE (Raíz del Error Cuadrático Medio):** Penaliza grandes desviaciones; se ubica en **{metrics['RMSE']}**.
                        * **MAPE:** El porcentaje de error medio absoluto es del **{metrics['MAPE (%)']}%**.
                        * **R² (Coeficiente de Determinación):** Indica la proporción de la varianza explicada por el modelo ($R^2 = {metrics['R2']}$).
                        """)

    elif menu == "6. Descubrimiento de Patrones & Causalidad":
        st.subheader("Causalidad de Granger y Clustering de Regímenes")
        st.info("los valores que se publican no los valores de las variables nominales sino loa valores ajustados de acuerdo a cada variable que ancla de ajuste")
        pat = PatternDiscovery(df)
        
        col1, col2 = st.columns(2)
        with col1:
            var_x = st.selectbox("Variable Causa (X)", df.columns, index=0)
        with col2:
            var_y = st.selectbox("Variable Efecto (Y)", df.columns, index=1)
            
        if st.button("Evaluar Causalidad"):
            res = pat.test_granger_causality(var_x, var_y)
            st.write(f"Resultados p-value para Granger ({var_x} ➔ {var_y}):")
            st.json(res)

if __name__ == "__main__":
    main()


#-----------------------------------------------------
# modelo basico de app 

# import streamlit as st
# from data_loader import DataLoader
# from eda import ExploratoryDataAnalysis
# from models_ml import MLModelBenchmark

# # Configuración de página
# st.set_page_config(page_title="Análisis Predictivo Económico AR", layout="wide")

# def main():
#     st.title("📊 Plataforma de Análisis Predictivo: Economía Argentina")
#     st.markdown("Análisis de Dólar, Merval e Inflación (Base 100 = Dic 2016)")

#     # 1. Carga de datos con la nueva ruta
#     loader = DataLoader("data/dolar-index.xlsx")
#     df = loader.load_and_clean_data()

#     if df.empty:
#         st.stop()

#     # Mapeo de columnas con sus descripciones para el usuario
#     column_descriptions = {
#         'dólar-cpra-evol': 'Dólar Compra Evolución (Variación nominal)',
#         'dólar-cpra-real': 'Dólar Compra Real (Descontando inflación/IPC)',
#         'dólar-vta-evol': 'Dólar Venta Evolución (Variación nominal)',
#         'dólar-vta-real': 'Dólar Venta Real (Descontando inflación/IPC)',
#         'merv$': 'Merval en Pesos (Variación nominal)',
#         'merv-dol': 'Merval en Dólares (Ajustado por tipo de cambio)',
#         'merv$-real': 'Merval Real (Ajustado por inflación/IPC)',
#         'merv-dol-real': 'Merval Dólar Real (Ajustado por tipo de cambio e inflación/IPC)',
#         'Ipc-Nacional': 'Inflación (IPC Nacional)'
#     }

#     # Barra lateral
#     st.sidebar.header("Configuración del Dashboard")
#     opcion = st.sidebar.selectbox(
#         "Seleccione un módulo",
#         ["1. Exploración de Datos (EDA)", "2. Modelos Predictivos (ML)"]
#     )

#     if opcion == "1. Exploración de Datos (EDA)":
#         st.header("🔍 Análisis Exploratorio")
#         eda = ExploratoryDataAnalysis(df)
        
#         # Permitir seleccionar usando las columnas exactas del dataset
#         variables = st.multiselect(
#             "Seleccione variables a visualizar", 
#             options=df.columns.tolist(), 
#             default=['dólar-vta-real', 'merv-dol', 'Ipc-Nacional'],
#             format_func=lambda x: column_descriptions.get(x, x) # Muestra la descripción si existe
#         )
#         if variables:
#             eda.plot_time_series(variables)
        
#         col1, col2 = st.columns(2)
#         with col1:
#             st.subheader("Estadísticas Descriptivas")
#             eda.show_descriptive_stats()
#         with col2:
#             st.subheader("Correlaciones")
#             eda.plot_correlation_matrix()

#     elif opcion == "2. Modelos Predictivos (ML)":
#         st.header("📈 Predicciones con Machine Learning")
#         target = st.selectbox(
#             "Seleccione la variable a predecir", 
#             options=df.columns.tolist(),
#             format_func=lambda x: column_descriptions.get(x, x)
#         )
#         lags = st.slider("Meses de rezago (Lags) para predecir", 1, 12, 3)
        
#         if st.button("Entrenar Modelo (Random Forest)"):
#             with st.spinner("Entrenando modelo..."):
#                 model = PredictiveModel(df, target, lags)
#                 metrics, fig = model.train_and_evaluate()
                
#                 st.plotly_chart(fig, width='stretch')
                
#                 kpi1, kpi2, kpi3 = st.columns(3)
#                 kpi1.metric("MAE (Error Absoluto Medio)", f"{metrics['MAE']:.2f}")
#                 kpi2.metric("RMSE (Raíz del Error Cuadrático)", f"{metrics['RMSE']:.2f}")
#                 kpi3.metric("R² (Bondad de ajuste)", f"{metrics['R2']:.2f}")

# if __name__ == "__main__":
#     main()