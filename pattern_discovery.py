import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

class PatternDiscovery:
    def __init__(self, df: pd.DataFrame):
        self.df = df.dropna()

    def test_granger_causality(self, var_x: str, var_y: str, max_lag: int = 4) -> dict:
        """Determina si var_x causa en el sentido de Granger a var_y."""
        data_sub = self.df[[var_y, var_x]]
        res = grangercausalitytests(data_sub, maxlag=max_lag, verbose=False)
        
        p_values = {f"Lag {lag}": round(res[lag][0]['ssr_ftest'][1], 4) for lag in range(1, max_lag + 1)}
        return p_values

    def apply_pca(self, columns: list, n_components: int = 2):
        """Aplica PCA para resumir múltiples indicadores (ej: IPCs regionales)."""
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(self.df[columns])
        explained_variance = pca.explained_variance_ratio_
        
        pca_df = pd.DataFrame(
            components, 
            columns=[f"PC{i+1}" for i in range(n_components)], 
            index=self.df.index
        )
        return pca_df, explained_variance

    def cluster_economic_regimes(self, columns: list, n_clusters: int = 3):
        """Agrupa los períodos históricos en regímenes económicos (ej: Alta/Baja volatilidad)."""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(self.df[columns])
        
        result_df = self.df[columns].copy()
        result_df['Régimen'] = [f"Régimen {c+1}" for c in clusters]
        return result_df