import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

class StatisticalModels:
    def __init__(self, series: pd.Series):
        self.series = series.dropna()

    def test_stationarity(self) -> dict:
        """Aplica pruebas ADF y KPSS para evaluar estacionariedad sin generar tipos NumPy."""
        # Prueba Augmented Dickey-Fuller (ADF)
        adf_res = adfuller(self.series)
        adf_stat = float(adf_res[0])
        adf_pval = float(adf_res[1])
        
        # Prueba KPSS
        kpss_res = kpss(self.series, regression='c', nlags="auto")
        kpss_stat = float(kpss_res[0])
        kpss_pval = float(kpss_res[1])

        adf_stationary = bool(adf_pval < 0.05)
        kpss_stationary = bool(kpss_pval > 0.05)

        # Diagnóstico integrado
        if adf_stationary and kpss_stationary:
            diag = "Estacionaria (No requiere diferenciación d=0)"
        elif not adf_stationary and not kpss_stationary:
            diag = "No Estacionaria (Se recomienda aplicar diferencia d=1 u orden superior)"
        elif adf_stationary and not kpss_stationary:
            diag = "Estacionaria en torno a la tendencia (Difference Stationary)"
        else:
            diag = "Estacionaria en torno a nivel (Trend Stationary)"

        return {
            "ADF_Stat": round(adf_stat, 4),
            "ADF_pvalue": round(adf_pval, 4),
            "ADF_Es_Estacionaria": adf_stationary,
            "KPSS_Stat": round(kpss_stat, 4),
            "KPSS_pvalue": round(kpss_pval, 4),
            "KPSS_Es_Estacionaria": kpss_stationary,
            "Diagnostico": diag
        }

    def fit_arima(self, order=(1, 1, 1), steps: int = 6):
        """Ajusta un modelo ARIMA(p,d,q) y retorna el pronóstico con intervalos de confianza."""
        model = ARIMA(self.series, order=order).fit()
        forecast_res = model.get_forecast(steps=steps)
        conf_int = forecast_res.conf_int(alpha=0.05)
        
        last_date = self.series.index[-1]
        future_dates = pd.date_range(start=last_date, periods=steps + 1, freq='MS')[1:]
        
        forecast_df = pd.DataFrame({
            'pred': forecast_res.predicted_mean.values,
            'lower': conf_int.iloc[:, 0].values,
            'upper': conf_int.iloc[:, 1].values
        }, index=future_dates)
        
        metrics = {
            "AIC": round(float(model.aic), 2),
            "BIC": round(float(model.bic), 2),
            "Log Likelihood": round(float(model.llf), 2)
        }
        
        return model, forecast_df, metrics

    def fit_holt_winters(self, steps: int = 6, seasonal_periods: int = 12):
        """Ajusta Suavizado Exponencial Holt-Winters."""
        has_enough_data = len(self.series) >= (2 * seasonal_periods)
        
        model = ExponentialSmoothing(
            self.series, 
            trend='add', 
            seasonal='add' if has_enough_data else None, 
            seasonal_periods=seasonal_periods if has_enough_data else None
        ).fit()
        
        last_date = self.series.index[-1]
        future_dates = pd.date_range(start=last_date, periods=steps + 1, freq='MS')[1:]
        forecast_vals = model.forecast(steps)
        
        forecast_df = pd.DataFrame({
            'pred': forecast_vals.values
        }, index=future_dates)
        
        metrics = {
            "AIC": round(float(model.aic), 2) if hasattr(model, 'aic') else "N/A",
            "BIC": round(float(model.bic), 2) if hasattr(model, 'bic') else "N/A"
        }
        
        return model, forecast_df, metrics



# import pandas as pd
# import numpy as np
# from statsmodels.tsa.stattools import adfuller, kpss
# from statsmodels.tsa.holtwinters import ExponentialSmoothing
# from statsmodels.tsa.arima.model import ARIMA
# import plotly.graph_objects as go
# 
# class StatisticalModels:
    # def __init__(self, series: pd.Series):
        # self.series = series.dropna()
# 
    # def test_stationarity(self) -> dict:
        # """Aplica pruebas ADF y KPSS para determinar estacionariedad."""
        #Prueba Augmented Dickey-Fuller
        # adf_result = adfuller(self.series)
        # adf_pvalue = adf_result[1]
        # 
        #Prueba KPSS
        # kpss_result = kpss(self.series, regression='c', nlags="auto")
        # kpss_pvalue = kpss_result[1]
# 
        # return {
            # "ADF Statistic": round(adf_result[0], 4),
            # "ADF p-value": round(adf_pvalue, 4),
            # "ADF Estacionaria (p < 0.05)": adf_pvalue < 0.05,
            # "KPSS Statistic": round(kpss_result[0], 4),
            # "KPSS p-value": round(kpss_pvalue, 4),
            # "KPSS Estacionaria (p > 0.05)": kpss_pvalue > 0.05
        # }
# 
    # def fit_holt_winters(self, steps: int = 6):
        # """Ajusta un modelo de Suavizado Exponencial Holt-Winters."""
        # model = ExponentialSmoothing(
            # self.series, 
            # trend='add', 
            # seasonal='add', 
            # seasonal_periods=12
        # ).fit()
        # 
        # forecast = model.forecast(steps)
        # return model, forecast
# 
    # def fit_arima(self, order=(1, 1, 1), steps: int = 6):
        # """Ajusta un modelo ARIMA(p, d, q)."""
        # model = ARIMA(self.series, order=order).fit()
        # forecast_res = model.get_forecast(steps=steps)
        # 
        # forecast_df = pd.DataFrame({
            # 'pred': forecast_res.predicted_mean,
            # 'lower': forecast_res.conf_int().iloc[:, 0],
            # 'upper': forecast_res.conf_int().iloc[:, 1]
        # })
        # return model, forecast_df