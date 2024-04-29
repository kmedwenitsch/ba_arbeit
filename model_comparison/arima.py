import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

# CSV-Datei einlesen
df_train = pd.read_csv('../input_data/Neudörfl_Production_Training_AT0090000000000000000X312X009800E.csv', parse_dates=['timestamp'])
df_test = pd.read_csv('../input_data/Neudörfl_Production_Test_AT0090000000000000000X312X009800E.csv', parse_dates=['timestamp'])

# Extrahieren von Merkmalen aus dem Zeitstempel
df_train['timestamp'] = pd.to_datetime(df_train['timestamp'])
df_train.set_index('timestamp', inplace=True)

# Modellinitialisierung
model = ARIMA(df_train['AT0090000000000000000X312X009800E'], order=(5,1,0))
model_fit = model.fit()

# Vorhersagen für den 30.12.2024 in 15-minütigen Zeitabständen
future_timestamps = pd.date_range(start='2023-12-30 00:00:00', end='2023-12-31 23:45:00', freq='15min')
future_predictions = model_fit.forecast(steps=len(future_timestamps))

# Berechnung der Metriken
y_true = df_test.loc[(df_test['timestamp'] >= '30.12.2023 00:00:00') & (df_test['timestamp'] <= '31.12.2023 23:45:00'), 'AT0090000000000000000X312X009800E'].values

print("Anzahl der Vorhersagen:", len(future_predictions))
print("Anzahl der tatsächlichen Werte:", len(y_true))

mse = mean_squared_error(y_true, future_predictions)
mae = mean_absolute_error(y_true, future_predictions)
mape = np.mean(np.abs((y_true - future_predictions) / y_true)) * 100
rmse = np.sqrt(mse)


print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("Mean Absolute Percentage Error (MAPE):", mape)
print("Root Mean Squared Error (RMSE):", rmse)

# Grafische Darstellung der Inputdaten
plt.figure(figsize=(12, 6))
plt.plot(future_timestamps, future_predictions, color='green', linestyle='--', label='Predicted Values')
plt.title('Energy Production Prediction for 30.12.2023 and 31.12.2023')
plt.xlabel('Timestamp')
plt.ylabel('Energy Production')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
