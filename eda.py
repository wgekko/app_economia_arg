import streamlit as st
import pandas as pd
import plotly.express as px

class ExploratoryDataAnalysis:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def show_raw_data(self):
        """Muestra la tabla de datos original formateada y con opciones de filtrado."""
        st.subheader("📋 Dataset Base de Series Económicas (Base 100 = Dic 2016)")
        
        # Filtro interactivo por fecha
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Fecha Desde", self.df.index.min(), min_value=self.df.index.min(), max_value=self.df.index.max())
        with col2:
            end_date = st.date_input("Fecha Hasta", self.df.index.max(), min_value=self.df.index.min(), max_value=self.df.index.max())

        df_filtered = self.df.loc[start_date:end_date]

        # Despliegue de la tabla con formato numérico limpio
        st.dataframe(
            df_filtered.style.format("{:,.2f}"),
            width='stretch',
            height=380
        )
        
        # Botones de descarga y resumen de observaciones
        c1, c2 = st.columns([3, 1])
        with c1:
            st.caption(f"Mostrando **{len(df_filtered)}** observaciones mensuales de un total de **{len(self.df)}** registros.")
        with c2:
            csv = df_filtered.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Exportar CSV",
                data=csv,
                file_name="series_economicas_argentina.csv",
                mime="text/csv",
                width='stretch'
            )

    def show_descriptive_stats(self):
        """Muestra las estadísticas descriptivas de las series."""
        st.subheader("Estadísticas Descriptivas (Media, Desvío, Min, Max)")
        st.dataframe(self.df.describe().T.style.format("{:,.2f}"), width='stretch')

    def plot_time_series(self, columns: list):
        """Gráfico interactivo de series temporales."""
        fig = px.line(
            self.df, 
            y=columns, 
            title="Evolución Histórica Comparativa",
            labels={'value': 'Índice (Dic 2016 = 100)', 'fecha': 'Fecha', 'variable': 'Variables'}
        )
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, width='stretch')

    def plot_correlation_matrix(self):
        """Matriz de correlación de Pearson."""
        st.subheader("🔥 Matriz de Correlaciones")
        corr = self.df.corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r")
        st.plotly_chart(fig, width='stretch')



#---------------------------------------------------------------------------------
#modelo sin publicar los datos base 


# import plotly.express as px
# import plotly.graph_objects as go
# import streamlit as st
# import pandas as pd

# class ExploratoryDataAnalysis:
#     def __init__(self, df: pd.DataFrame):
#         self.df = df

#     def plot_time_series(self, columns: list):
#         """Genera un gráfico interactivo de series temporales."""
#         fig = px.line(
#             self.df, 
#             y=columns, 
#             title="Evolución Histórica de Variables (Base 100 = Dic 2016)",
#             labels={'value': 'Índice', 'fecha': 'Fecha', 'variable': 'Variables'}
#         )
#         fig.update_layout(hovermode="x unified")
#         st.plotly_chart(fig, width='stretch')

#     def show_descriptive_stats(self):
#         """Muestra estadísticas descriptivas de las variables seleccionadas."""
#         st.dataframe(self.df.describe().T, width='stretch')

#     def plot_correlation_matrix(self):
#         """Genera un Heatmap de correlaciones."""
#         corr = self.df.corr()
#         fig = px.imshow(corr, text_auto=True, aspect="auto", title="Matriz de Correlaciones")
#         st.plotly_chart(fig, width='stretch')