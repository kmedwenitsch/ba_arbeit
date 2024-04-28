import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# CSV-Datei einlesen
data = pd.read_csv('../input_data/Neudörfl_Production_bis_12_2023_AT0090000000000000000X312X009800E.csv', parse_dates=['timestamp'])

# Extrahieren von Merkmalen aus dem Zeitstempel
data['hour'] = data['timestamp'].dt.hour  # Stunde des Tages
data['weekday'] = data['timestamp'].dt.weekday  # Wochentag (Montag=0, Sonntag=6)
data['month'] = data['timestamp'].dt.month  # Monat

# Modellinitialisierung
model = LinearRegression()

# Trainieren des Modells
X = data[['hour', 'weekday', 'month']]
y = data['AT0090000000000000000X312X009800E']
model.fit(X, y)

# Vorhersagen für den 1.1.2024 in 15-minütigen Zeitabständen
future_timestamps = pd.date_range(start='2024-01-01 00:00:00', end='2024-01-01 23:45:00', freq='15min')
future_features = pd.DataFrame({
    'hour': future_timestamps.hour,
    'weekday': future_timestamps.weekday,
    'month': future_timestamps.month
})
future_predictions = model.predict(future_features)

# Berechnung der Metriken
y_true = data.loc[(data['timestamp'] >= '30.12.2023') & (data['timestamp'] < '31.12.2023'), 'AT0090000000000000000X312X009800E'].values

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
plt.title('Energy Consumption Prediction for 1.1.2024')
plt.xlabel('Timestamp')
plt.ylabel('Energy Consumption')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
