import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=64, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

class NeuralNetworkForecaster:
    def __init__(self, series: pd.Series, seq_length: int = 6):
        self.series_orig = series.dropna()
        self.dates = self.series_orig.index
        self.values = self.series_orig.values.reshape(-1, 1)
        self.seq_length = seq_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaled_data = self.scaler.fit_transform(self.values)

    def _prepare_sequences(self, test_ratio: float = 0.2):
        x, y = [], []
        for i in range(len(self.scaled_data) - self.seq_length):
            x.append(self.scaled_data[i:i+self.seq_length])
            y.append(self.scaled_data[i+self.seq_length])
            
        X_arr, y_arr = np.array(x), np.array(y)
        split_idx = int(len(X_arr) * (1 - test_ratio))
        
        X_train, X_test = X_arr[:split_idx], X_arr[split_idx:]
        y_train, y_test = y_arr[:split_idx], y_arr[split_idx:]
        
        target_dates = self.dates[self.seq_length:]
        dates_train = target_dates[:split_idx]
        dates_test = target_dates[split_idx:]

        return (
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32),
            dates_train,
            dates_test
        )

    def train_and_forecast(self, epochs: int = 80, lr: float = 0.01, horizon: int = 6):
        X_train, X_test, y_train, y_test, dates_train, dates_test = self._prepare_sequences()
        
        model = LSTMModel()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_losses, val_losses = [], []

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            y_pred_train = model(X_train)
            loss = criterion(y_pred_train, y_train)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            # Pérdida de validación
            model.eval()
            with torch.no_grad():
                y_pred_val = model(X_test)
                val_loss = criterion(y_pred_val, y_test)
                val_losses.append(val_loss.item())

        # Evaluación en set de prueba
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test).numpy()
            
        test_preds_orig = self.scaler.inverse_transform(test_preds).flatten()
        y_test_orig = self.scaler.inverse_transform(y_test.numpy()).flatten()

        # Métricas de precisión
        mae = mean_absolute_error(y_test_orig, test_preds_orig)
        rmse = np.sqrt(mean_squared_error(y_test_orig, test_preds_orig))
        mape = np.mean(np.abs((y_test_orig - test_preds_orig) / (y_test_orig + 1e-8))) * 100
        r2 = r2_score(y_test_orig, test_preds_orig)

        # Proyección futura autoregresiva
        future_preds = []
        current_seq = self.scaled_data[-self.seq_length:].reshape(1, self.seq_length, 1)
        current_seq_tensor = torch.tensor(current_seq, dtype=torch.float32)

        with torch.no_grad():
            for _ in range(horizon):
                next_val = model(current_seq_tensor)
                future_preds.append(next_val.numpy()[0, 0])
                next_val_reshaped = next_val.unsqueeze(1)
                current_seq_tensor = torch.cat((current_seq_tensor[:, 1:, :], next_val_reshaped), dim=1)

        future_preds_orig = self.scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()

        # Generar fechas futuras (mensuales)
        last_date = self.dates[-1]
        future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq='MS')[1:]

        df_eval = pd.DataFrame({
            'Real': y_test_orig,
            'Predicción': test_preds_orig
        }, index=dates_test)

        df_future = pd.DataFrame({
            'Pronóstico LSTM': future_preds_orig
        }, index=future_dates)

        metrics = {
            'MAE': round(float(mae), 2),
            'RMSE': round(float(rmse), 2),
            'MAPE (%)': round(float(mape), 2),
            'R2': round(float(r2), 4)
        }

        df_loss = pd.DataFrame({
            'Entrenamiento': train_losses,
            'Validación': val_losses
        })

        return df_loss, df_eval, df_future, metrics

#----------------------------------------------------------------------------------
# modelo basico de redes neuronales

# import torch
# import torch.nn as nn
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler

# class LSTMModel(nn.Module):
#     def __init__(self, input_size=1, hidden_layer_size=64, output_size=1):
#         super().__init__()
#         self.hidden_layer_size = hidden_layer_size
#         self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
#         self.linear = nn.Linear(hidden_layer_size, output_size)

#     def forward(self, input_seq):
#         lstm_out, _ = self.lstm(input_seq)
#         predictions = self.linear(lstm_out[:, -1, :])
#         return predictions

# class NeuralNetworkForecaster:
#     def __init__(self, series: pd.Series, seq_length: int = 6):
#         self.series = series.dropna().values.reshape(-1, 1)
#         self.seq_length = seq_length
#         self.scaler = MinMaxScaler(feature_range=(0, 1))
#         self.scaled_data = self.scaler.fit_transform(self.series)

#     def _prepare_sequences(self):
#         x, y = [], []
#         for i in range(len(self.scaled_data) - self.seq_length):
#             x.append(self.scaled_data[i:i+self.seq_length])
#             y.append(self.scaled_data[i+self.seq_length])
#         return torch.tensor(np.array(x), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)

#     def train_lstm(self, epochs: int = 100, lr: float = 0.01):
#         X, y = self._prepare_sequences()
#         model = LSTMModel()
#         loss_function = nn.MSELoss()
#         optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#         loss_history = []
#         for epoch in range(epochs):
#             model.train()
#             optimizer.zero_grad()
#             y_pred = model(X)
#             single_loss = loss_function(y_pred, y)
#             single_loss.backward()
#             optimizer.step()
#             loss_history.append(single_loss.item())

#         return model, loss_history