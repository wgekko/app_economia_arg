import pandas as pd
import streamlit as st

class DataLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath

    @st.cache_data(show_spinner=False)
    def load_and_clean_data(_self) -> pd.DataFrame:
        """Carga y limpia el archivo Excel de variables económicas."""
        try:
            # Ruta de acceso a los datos
            df = pd.read_excel(_self.filepath, sheet_name='data')
            
            # Asegurar que la columna fecha sea datetime y setearla como índice
            df['fecha'] = pd.to_datetime(df['fecha'])
            df.set_index('fecha', inplace=True)
            
            # Manejo de valores nulos (Sintaxis corregida para la nueva versión de Pandas)
            df.ffill(inplace=True)
            df.bfill(inplace=True)
            
            return df
        except Exception as e:
            st.error(f"Error al cargar los datos desde {_self.filepath}: {e}")
            return pd.DataFrame()