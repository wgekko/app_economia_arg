import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import VAR
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import plotly.figure_factory as ff
import streamlit.components.v1 as components

# Configuración de página
#st.set_page_config(page_title="Análisis Macro: Dólar vs Inflación vs Merval", layout="wide", page_icon=":material/analytics:")


def run_statistics_app():

        # Título y Descripción
        st.subheader("Análisis Estadístico: Dinámica del Dólar, IPC y Merval")
        st.markdown("Este dashboard interactivo aplica modelos de estadística descriptiva, probabilidad y Machine Learning sobre la serie histórica del Dólar, IPC regional y el índice Merval.")

        # Carga de datos
        @st.cache_data
        def load_data():
            try:
                df = pd.read_excel('data/dolar-index.xlsx')
                df['fecha'] = pd.to_datetime(df['fecha'])
                df = df.sort_values('fecha').reset_index(drop=True)
                # Limpiar nombres de columnas para que coincidan con la solicitud
                df = df.rename(columns={'Región Pampeana-Nivel general': 'Región Ipc-Pampeana'}) 
                return df
            except Exception as e:
                st.error(f"Error al cargar el archivo 'dolar-index.xlsx': {e}")
                return pd.DataFrame()

        df = load_data()

        if not df.empty:
            
            # ------------------ SIDEBAR ------------------
            st.sidebar.header("Filtros Temporales")
            min_date = df['fecha'].min().date()
            max_date = df['fecha'].max().date()
            
            start_date, end_date = st.sidebar.slider(
                "Seleccionar Rango de Fechas",
                min_value=min_date, max_value=max_date,
                value=(min_date, max_date)
            )
            
            df_filtered = df[(df['fecha'].dt.date >= start_date) & (df['fecha'].dt.date <= end_date)].copy()
            
            # Variables principales
            var_dolar = 'Dolar-venta'
            var_ipc = 'TotalNacional-Nivelgeneral'
            var_merv = 'indMerv'
            
            df_filtered = df_filtered.dropna(subset=[var_dolar, var_ipc])

            # ------------------ SECCIÓN 1: EVOLUCIÓN HISTÓRICA ------------------
            st.write("---")
            st.subheader("1. Evolución Histórica: Variación Mensual y Acumulada")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Variación Mensual (Dólar vs IPC vs Merv)")
                fig_line = px.line(df_filtered, x='fecha', y=[var_dolar, var_ipc, var_merv],
                                labels={'value': 'Variación Mensual (%)', 'variable': 'Indicador'},
                                color_discrete_map={var_dolar: '#2CA02C', var_ipc: '#D62728' , var_merv: "#7C62F3"})
                fig_line.update_layout(hovermode="x unified", legend_title="")
                st.plotly_chart(fig_line, width='stretch')

            with col2:
                st.subheader("Evolución Acumulada (Base 100)")
                # Cálculo de acumulados
                df_filtered['Dolar_Acumulado'] = (1 + df_filtered[var_dolar]/100).cumprod() * 100
                df_filtered['IPC_Acumulado'] = (1 + df_filtered[var_ipc]/100).cumprod() * 100
                df_filtered['Merv_Acumulado'] = (1 + df_filtered[var_merv]/100).cumprod() * 100

                
                fig_acum = px.line(df_filtered, x='fecha', y=['Dolar_Acumulado', 'IPC_Acumulado', 'Merv_Acumulado'],
                                labels={'value': 'Índice (Base 100)', 'variable': 'Indicador'},
                                color_discrete_map={'Dolar_Acumulado': '#2CA02C', 'IPC_Acumulado': '#D62728', 'Merv_Acumulado':  "#7C62F3"})
                fig_acum.update_layout(hovermode="x unified", legend_title="")
                st.plotly_chart(fig_acum, width='stretch')

            # ------------------ SECCIÓN 2: ESTADÍSTICA DESCRIPTIVA ------------------
            st.write("---")
            st.subheader("2. Estadística Descriptiva y Distribución")
            
            col_stat1, col_stat2 = st.columns([1, 2])
            
            with col_stat1:

                st.subheader("Métricas Clave")
                st.write("---")
                st.write("")
                stats_df = pd.DataFrame({
                    'Métrica': ['Media (%)', 'Mediana (%)', 'Desv. Estándar (Volatilidad)', 'Coef. Variación', 'Rango Intercuartílico (IQR)', 'Asimetría (Skewness)', 'Curtosis (Fat Tails)'],
                    'Dólar': [
                        df_filtered[var_dolar].mean(), df_filtered[var_dolar].median(), df_filtered[var_dolar].std(),
                        df_filtered[var_dolar].std() / df_filtered[var_dolar].mean() if df_filtered[var_dolar].mean() != 0 else np.nan,
                        df_filtered[var_dolar].quantile(0.75) - df_filtered[var_dolar].quantile(0.25),
                        df_filtered[var_dolar].skew(), df_filtered[var_dolar].kurtosis()
                    ],
                    'IPC': [
                        df_filtered[var_ipc].mean(), df_filtered[var_ipc].median(), df_filtered[var_ipc].std(),
                        df_filtered[var_ipc].std() / df_filtered[var_ipc].mean() if df_filtered[var_ipc].mean() != 0 else np.nan,
                        df_filtered[var_ipc].quantile(0.75) - df_filtered[var_ipc].quantile(0.25),
                        df_filtered[var_ipc].skew(), df_filtered[var_ipc].kurtosis()
                    ],
                    'Merv': [
                                    df_filtered[var_merv].mean(), df_filtered[var_ipc].median(), df_filtered[var_ipc].std(),
                                    df_filtered[var_merv].std() / df_filtered[var_ipc].mean() if df_filtered[var_ipc].mean() != 0 else np.nan,
                                    df_filtered[var_merv].quantile(0.75) - df_filtered[var_ipc].quantile(0.25),
                                    df_filtered[var_merv].skew(), df_filtered[var_ipc].kurtosis()
                                ]
                })
                st.dataframe(stats_df.style.format(precision=2))
                
            with col_stat2:
                st.subheader("Histogramas y Diagrama de Caja (Identificación de Shocks)")
                tab1, tab2 = st.tabs(["Histograma (Distribución)", "Boxplot (Outliers)"])
                
                with tab1:
                    hist_data = [df_filtered[var_dolar].dropna(), df_filtered[var_ipc].dropna(), df_filtered[var_merv].dropna()]
                    group_labels = ['Dólar', 'IPC', 'Merv']
                    fig_hist = ff.create_distplot(hist_data, group_labels, bin_size=1, colors=['#2CA02C', '#D62728', "#7C62F3"])
                    fig_hist.update_layout(xaxis_title="Variación Mensual (%)", yaxis_title="Densidad")
                    st.plotly_chart(fig_hist, width='stretch')
                    
                with tab2:
                    fig_box = go.Figure()
                    fig_box.add_trace(go.Box(y=df_filtered[var_dolar], name='Dólar', marker_color='#2CA02C'))
                    fig_box.add_trace(go.Box(y=df_filtered[var_ipc], name='IPC', marker_color='#D62728'))
                    fig_box.add_trace(go.Box(y=df_filtered[var_ipc], name='Merv', marker_color="#7C62F3"))
                    fig_box.update_layout(yaxis_title="Variación Mensual (%)")
                    st.plotly_chart(fig_box, width='stretch')



            # ------------------ SECCIÓN 3: PATRONES Y TRASPASO A PRECIOS ------------------
            st.write("---")
            st.subheader("3. Análisis de Patrones y Traspaso a Precios (Pass-Through)")
            st.info("se analiza el traspaso a precios de la evolucion del dolar (no se incluye la participación del Indice Merval)")
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader("Correlación Cruzada (Rezagos)")
                lags = range(0, 7)
                corr_values = [df_filtered[var_dolar].shift(i).corr(df_filtered[var_ipc]) for i in lags]
                
                fig_lags = px.bar(x=[f"Mes t-{i}" for i in lags], y=corr_values,
                                labels={'x': 'Meses de Rezago del Dólar', 'y': 'Coeficiente de Correlación'})
                fig_lags.update_traces(marker_color='#1F77B4')
                st.plotly_chart(fig_lags, width='stretch')
                st.markdown("*Muestra en qué mes de rezago el impacto del dólar es más fuerte sobre la inflación.*")

            with col4:
                st.subheader("Dispersión y Tendencia (Dólar vs IPC)")
                fig_scatter = px.scatter(df_filtered, x=var_dolar, y=var_ipc, trendline="ols",
                                        labels={var_dolar: 'Variación Dólar (%)', var_ipc: 'Variación IPC (%)'})
                st.plotly_chart(fig_scatter, width='stretch')
                
            # ------------------ SECCIÓN 4: PROBABILIDAD Y RIESGO ------------------
            st.write("---")
            st.subheader("4. Probabilidad Empírica y Percentiles de Riesgo")
            
            col_p1, col_p2 = st.columns(2)
            
            with col_p1:
                st.subheader("Percentiles Históricos")
                percentiles = [50, 75, 90, 95, 99]
                perc_df = pd.DataFrame({
                    'Percentil': [f"P{p}" for p in percentiles],
                    'Dólar Máximo Esperado (%)': [np.percentile(df_filtered[var_dolar].dropna(), p) for p in percentiles],
                    'IPC Máximo Esperado (%)': [np.percentile(df_filtered[var_ipc].dropna(), p) for p in percentiles],
                    'Merv Máximo Esperado (%)': [np.percentile(df_filtered[var_merv].dropna(), p) for p in percentiles],
                })
                st.write("###")
                st.dataframe(perc_df.style.format(precision=2))
                st.markdown("*Ej: P95 indica que el 95% de los meses la variación fue menor a ese valor.*")
                
            with col_p2:
                st.subheader("Análisis Regional (Variabilidad del IPC)")
                st.info("se grafica la evolución del indice de cada región con base 100=12/2016")
                # Filtrar columnas de IPC regional
                cols_regionales = [c for c in df_filtered.columns if 'Ipc-' in c and 'Nacional' not in c]
                if cols_regionales:
                    fig_regional = px.box(df_filtered, y=cols_regionales, 
                                        labels={'value': 'Variación Mensual (%)', 'variable': 'Región'})
                    st.plotly_chart(fig_regional, width='stretch')

            # ------------------ SECCIÓN 5: ANÁLISIS MULTIVARIABLE, REGIONAL Y BURSÁTIL ------------------
            st.write("---")
            st.subheader("5. Análisis Comparativo y Estadístico Multivariable")
            
            cols_dolar_req = ['dólar-cpra-evol', 'dólar-cpra-real', 'Dolar-venta', 'dólar-vta-evol', 'dólar-vta-real']
            cols_ipc_req = ['Ipc-Nacional', 'Ipc-GBA', 'Región Ipc-Pampeana', 'Ipc-Noroeste', 'Ipc-Noreste', 'Ipc-Cuyo', 'Ipc-Patagonia']
            cols_merval_req = ['merv$', 'merv-dol', 'merv$-real', 'merv-dol-real']
            
            available_cols_dolar = [c for c in cols_dolar_req if c in df_filtered.columns]
            available_cols_ipc = [c for c in cols_ipc_req if c in df_filtered.columns]
            available_cols_merval = [c for c in cols_merval_req if c in df_filtered.columns]
            
            todas_las_variables = available_cols_dolar + available_cols_ipc + available_cols_merval

            st.subheader("Evolución Histórica Comparativa")
            variables_seleccionadas = st.multiselect(
                "Selecciona las variables a comparar:",
                options=todas_las_variables,
                default=['dólar-vta-evol', 'Ipc-Nacional','merv$' ] if 'dólar-vta-evol' in todas_las_variables and 'Ipc-Nacional' in todas_las_variables and 'merv$' in todas_las_variables else todas_las_variables[:2]
            )
            
            if variables_seleccionadas:
                fig_multi = px.line(df_filtered, x='fecha', y=variables_seleccionadas,
                                labels={'value': 'Valor / Índice', 'variable': 'Indicador'})
                fig_multi.update_layout(hovermode="x unified")
                st.plotly_chart(fig_multi, width='stretch')

            st.subheader("Cálculos Estadísticos Descriptivos (Variables Seleccionadas)")
            desc_stats = pd.DataFrame()
            for col in todas_las_variables:
                serie = df_filtered[col].dropna()
                if not serie.empty:
                    desc_stats.loc[col, 'Media'] = serie.mean()
                    desc_stats.loc[col, 'Mediana'] = serie.median()
                    desc_stats.loc[col, 'Mínimo'] = serie.min()
                    desc_stats.loc[col, 'Máximo'] = serie.max()
                    desc_stats.loc[col, 'Desv. Est.'] = serie.std()
                    mean_val = serie.mean()
                    desc_stats.loc[col, 'Coef. Var.'] = (serie.std() / mean_val) if mean_val != 0 else np.nan
                    desc_stats.loc[col, 'Asimetría'] = serie.skew()
                    desc_stats.loc[col, 'Curtosis'] = serie.kurtosis()
                    desc_stats.loc[col, 'Percentil 95'] = np.percentile(serie, 95)

            if not desc_stats.empty:
                st.dataframe(desc_stats.style.format(precision=2))
                
            st.subheader("Comparativa de Dispersión y Outliers (Boxplots)")
            tab_multi1, tab_multi2, tab_multi3, tab_multi4 = st.tabs(["Variables del Dólar", "Índices de Precios", "Merval", "Matriz de Correlación"])
            
            with tab_multi1:
                if available_cols_dolar:
                    fig_box_dolar = px.box(df_filtered, y=available_cols_dolar, 
                                        labels={'value': 'Valor / Variación', 'variable': 'Indicador'})
                    st.plotly_chart(fig_box_dolar, width='stretch')
                    
            with tab_multi2:
                if available_cols_ipc:
                    fig_box_ipc = px.box(df_filtered, y=available_cols_ipc,
                                        labels={'value': 'Índice de Precios', 'variable': 'Región'})
                    st.plotly_chart(fig_box_ipc, width='stretch')
                    
            with tab_multi3:
                if available_cols_merval:
                    fig_box_merval = px.box(df_filtered, y=available_cols_merval,
                                        labels={'value': 'Índice Merval', 'variable': 'Indicador'})
                    st.plotly_chart(fig_box_merval, width='stretch')
                    
            with tab_multi4:
                if len(todas_las_variables) > 1:
                    corr_matrix = df_filtered[todas_las_variables].corr()
                    fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto",
                                        color_continuous_scale='RdBu_r')
                    st.plotly_chart(fig_corr, width='stretch')

            # ------------------ SECCIÓN 6: MACHINE LEARNING Y MODELOS AVANZADOS ------------------
            st.header("6. Machine Learning y Modelos Avanzados")
            st.markdown("Aplicación de modelos predictivos y algoritmos no supervisados para descubrir patrones complejos.")
            
            tab_ml1, tab_ml2, tab_ml3 = st.tabs([
                "Regímenes Económicos (K-Means)", 
                "Importancia de Variables (Random Forest)", 
                "Proyección Multivariable (Modelo VAR)"
            ])
            
            with tab_ml1:
                st.subheader("Agrupación por Regímenes (Clustering)")
                st.markdown("Identificación automática de 'fases' macroeconómicas basadas en la volatilidad del dólar y la inflación.")
                try:
                    # Preparar datos para clustering
                    cluster_data = df_filtered[[var_dolar, var_ipc]].dropna()
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(cluster_data)
                    
                    # Aplicar K-Means
                    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                    cluster_data['Régimen'] = kmeans.fit_predict(scaled_data)
                    cluster_data['Régimen'] = cluster_data['Régimen'].astype(str)
                    
                    fig_cluster = px.scatter(
                        cluster_data, x=var_dolar, y=var_ipc, color='Régimen',
                        title="Regímenes Económicos Históricos",
                        labels={var_dolar: "Dólar (%)", var_ipc: "IPC (%)"},
                        color_discrete_sequence=px.colors.qualitative.Set1
                    )
                    st.plotly_chart(fig_cluster, width='stretch')
                except Exception as e:
                    st.warning(f"No hay suficientes datos para el modelo de clustering. {e}")

            with tab_ml2:
                st.subheader("¿Qué variables predicen mejor la Inflación? (Random Forest)")
                try:
                    # Crear rezagos (lags) para el modelo
                    ml_df = df_filtered[['fecha', var_dolar, var_ipc]].copy()
                    if 'merv-dol' in df_filtered.columns:
                        ml_df['Merval_Dol'] = df_filtered['merv-dol']
                    
                    # Generar features (rezagos de 1 y 2 meses)
                    ml_df['Dolar_Lag1'] = ml_df[var_dolar].shift(1)
                    ml_df['Dolar_Lag2'] = ml_df[var_dolar].shift(2)
                    ml_df['IPC_Lag1'] = ml_df[var_ipc].shift(1)
                    
                    if 'Merval_Dol' in ml_df.columns:
                        ml_df['Merval_Lag1'] = ml_df['Merval_Dol'].shift(1)
                    
                    ml_df = ml_df.dropna()
                    
                    # Separar variables predictoras (X) y objetivo (y)
                    y = ml_df[var_ipc]
                    X = ml_df.drop(columns=['fecha', var_ipc, var_dolar])
                    
                    if len(X) > 10:
                        rf = RandomForestRegressor(n_estimators=100, random_state=42)
                        rf.fit(X, y)
                        
                        # Graficar importancia
                        importances = pd.DataFrame({'Variable': X.columns, 'Importancia': rf.feature_importances_})
                        importances = importances.sort_values('Importancia', ascending=True)
                        
                        fig_rf = px.bar(importances, x='Importancia', y='Variable', orientation='h',
                                        title="Peso de los factores históricos en el IPC actual")
                        st.plotly_chart(fig_rf, width='stretch')
                    else:
                        st.info("Se necesitan más meses de datos para entrenar el modelo Random Forest.")
                except Exception as e:
                    st.warning(f"Error al ejecutar Random Forest: {e}")

                with tab_ml3:
                        st.subheader("Proyección a Futuro (Pronóstico VAR a 3 meses)")
                        st.markdown("Modelo econométrico Vector Autorregresivo para predecir interacciones a corto plazo.")
                        try:
                            var_data = df_filtered[['fecha', var_dolar, var_ipc]].dropna().set_index('fecha')
                            if len(var_data) > 20:
                                
                                # --- AQUÍ ESTÁ EL CAMBIO SUGERIDO ---
                                # Convertimos el índice a DatetimeIndex explícito y forzamos la frecuencia mensual (MS)
                                var_data.index = pd.DatetimeIndex(var_data.index)
                                var_data = var_data.asfreq('MS')
                                # ------------------------------------

                                # Entrenar modelo VAR
                                model = VAR(var_data)
                                results = model.fit(maxlags=2, ic='aic')
                                
                                # Proyectar 3 meses
                                lag_order = results.k_ar
                                forecast = results.forecast(var_data.values[-lag_order:], steps=3)
                                
                                # Armar DataFrame de proyección
                                last_date = var_data.index[-1]
                                future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, 4)]
                                df_forecast = pd.DataFrame(forecast, index=future_dates, columns=[f'Proy_{var_dolar}', f'Proy_{var_ipc}'])
                                
                                # Unir historia reciente y proyección para graficar
                                hist_recent = var_data.tail(12)
                                
                                fig_var = go.Figure()
                                # Historia
                                fig_var.add_trace(go.Scatter(x=hist_recent.index, y=hist_recent[var_ipc], mode='lines+markers', name='IPC (Histórico)'))
                                fig_var.add_trace(go.Scatter(x=hist_recent.index, y=hist_recent[var_dolar], mode='lines+markers', name='Dólar (Histórico)'))
                                # Proyección
                                fig_var.add_trace(go.Scatter(x=df_forecast.index, y=df_forecast[f'Proy_{var_ipc}'], mode='lines+markers', name='IPC (Proyección)', line=dict(dash='dash')))
                                fig_var.add_trace(go.Scatter(x=df_forecast.index, y=df_forecast[f'Proy_{var_dolar}'], mode='lines+markers', name='Dólar (Proyección)', line=dict(dash='dash')))
                                
                                st.plotly_chart(fig_var, width='stretch')
                                st.dataframe(df_forecast.round(2), width='stretch')
                            else:
                                st.info("El modelo VAR requiere al menos 20 registros históricos continuos para hacer una proyección confiable.")
                        except Exception as e:
                            st.warning(f"Error al calcular el modelo VAR. Verifica que las series sean estacionarias. {e}")

            
            # ------------------ SECCIÓN 7: EXPORTACIÓN A PDF ------------------
            st.markdown("---")
            st.subheader("7. Exportar Información a PDF")
            st.markdown("Puedes descargar los cálculos estadísticos tabulares o bien imprimir todo el Dashboard con sus gráficos.")
            
            col_export1, col_export2 = st.columns(2)
            
            with col_export1:
                st.subheader("Descargar Tabla Estadística")
                try:
                    from fpdf import FPDF, XPos, YPos
                    
                    def generar_pdf_estadisticas(df):
                        pdf = FPDF(orientation='L') # Formato horizontal
                        pdf.add_page()
                        pdf.set_font("helvetica", 'B', 14)
                        pdf.cell(0, 10, "Resultados Estadisticos (Dolar, IPC y Merval)", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
                        pdf.ln(5)
                        
                        # Preparamos el dataframe
                        df_print = df.reset_index().round(2).fillna("N/A")
                        df_print = df_print.rename(columns={'index': 'Variable'})
                        cols = list(df_print.columns)
                        
                        # Ancho dinámico
                        col_width = 270 / len(cols)
                        
                        # Headers
                        pdf.set_font("helvetica", 'B', 8)
                        for col in cols:
                            pdf.cell(col_width, 8, str(col)[:15], border=1, align='C')
                        pdf.ln()
                        
                        # Filas
                        pdf.set_font("helvetica", '', 8)
                        for _, row in df_print.iterrows():
                            for item in row:
                                pdf.cell(col_width, 8, str(item)[:15], border=1, align='C')
                            pdf.ln()
                            
                        # Generar output
                        output = pdf.output()
                        if isinstance(output, str):
                            return output.encode('latin-1', 'replace')
                        return bytes(output)

                    if not desc_stats.empty:
                        pdf_bytes = generar_pdf_estadisticas(desc_stats)
                        st.download_button(
                            label="Descargar Tabla en PDF",
                            data=pdf_bytes,
                            file_name="estadisticas_macro.pdf",
                            mime="application/pdf"
                        )
                        st.caption("Solo exporta la tabla de cálculos estadísticos (Sección 5).")
                        
                except ImportError:
                    st.warning("La librería nativa de PDF no está detectada en tu entorno. Instálala ejecutando: `pip install fpdf2`")
                    if not desc_stats.empty:
                        csv = desc_stats.to_csv().encode('utf-8')
                        st.download_button("Descargar Tabla como CSV (Alternativa)", data=csv, file_name="estadisticas.csv", mime="text/csv")
                        
                with col_export2:
                        st.subheader("Guardar Dashboard Completo")
                        st.markdown("Utiliza este botón para **guardar todo el tablero en PDF** (incluyendo gráficos) a través de las opciones nativas de impresión de tu navegador.")
                        
                        # Pasamos el HTML como el primer argumento directamente, sin 'srcdoc='
                        st.iframe(
                            '''
                            <script>
                            function printDashboard() {
                                window.parent.print();
                            }
                            </script>
                            <button onclick="printDashboard()" style="
                                background-color: #FF4B4B;
                                color: white;
                                border: none;
                                padding: 10px 20px;
                                text-align: center;
                                text-decoration: none;
                                display: inline-block;
                                font-size: 16px;
                                margin: 4px 2px;
                                cursor: pointer;
                                border-radius: 4px;
                                font-family: sans-serif;
                                font-weight: bold;
                                width: 100%;
                            ">Imprimir / Guardar Dashboard (PDF)</button>
                            ''',
                            height=70
                        )
        