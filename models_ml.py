import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import plotly.graph_objects as go

class MLModelBenchmark:
    def __init__(self, df: pd.DataFrame, target_col: str, lags: int = 6):
        self.df = df.copy()
        self.target_col = target_col
        self.lags = lags
        
    def _create_lag_features(self):
        df_feat = self.df[[self.target_col]].copy()
        for i in range(1, self.lags + 1):
            df_feat[f'lag_{i}'] = df_feat[self.target_col].shift(i)
        df_feat.dropna(inplace=True)
        
        X = df_feat.drop(columns=[self.target_col])
        y = df_feat[self.target_col]
        
        split_idx = int(len(X) * 0.8)
        return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]

    def evaluate_all(self):
        X_train, X_test, y_train, y_test = self._create_lag_features()
        
        models = {
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "XGBoost": xgb.XGBRegressor(n_estimators=100, random_state=42),
            "LightGBM": lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "SVR": SVR(kernel='rbf', C=100)
        }
        
        results = []
        predictions_dict = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            predictions_dict[name] = pd.Series(preds, index=y_test.index)
            
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
            r2 = r2_score(y_test, preds)
            
            results.append({
                "Modelo": name,
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
                "MAPE (%)": round(mape, 2),
                "R²": round(r2, 4)
            })

        results_df = pd.DataFrame(results).sort_values(by="RMSE")
        best_model_name = results_df.iloc[0]["Modelo"]
        
        return results_df, predictions_dict, y_test, best_model_name


#-------------------------------------------------------------------------
# modelo basico de prediccion Random Forest


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import plotly.graph_objects as go
# import streamlit as st

# class PredictiveModel:
#     def __init__(self, df: pd.DataFrame, target_col: str, lags: int = 3):
#         self.df = df.copy()
#         self.target_col = target_col
#         self.lags = lags
#         self.model = RandomForestRegressor(n_estimators=100, random_state=42)

#     def create_features(self):
#         """Crea variables rezagadas (lags) para series temporales."""
#         for i in range(1, self.lags + 1):
#             self.df[f'lag_{i}'] = self.df[self.target_col].shift(i)
#         self.df.dropna(inplace=True)
        
#         X = self.df[[f'lag_{i}' for i in range(1, self.lags + 1)]]
#         y = self.df[self.target_col]
#         return train_test_split(X, y, test_size=0.2, shuffle=False)

#     def train_and_evaluate(self):
#         X_train, X_test, y_train, y_test = self.create_features()
#         self.model.fit(X_train, y_train)
#         predictions = self.model.predict(X_test)

#         # Métricas
#         mae = mean_absolute_error(y_test, predictions)
#         rmse = np.sqrt(mean_squared_error(y_test, predictions))
#         r2 = r2_score(y_test, predictions)

#         # Visualización
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines', name='Real'))
#         fig.add_trace(go.Scatter(x=y_test.index, y=predictions, mode='lines', name='Predicción RF', line=dict(dash='dash')))
#         fig.update_layout(title=f"Predicción vs Realidad: {self.target_col}")
        
#         return {"MAE": mae, "RMSE": rmse, "R2": r2}, fig