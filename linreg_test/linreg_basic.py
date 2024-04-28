import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

# Berechnung von MAPE (Mean Absolute Percentage Error)
from sklearn.model_selection import train_test_split


def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Beispielzeitreihe erstellen
df = pd.read_csv("../input_data/Neudörfl_Production_bis_12_2023_AT0090000000000000000X312X009800E.csv")
df['timestamp'] = pd.to_datetime(df["timestamp"])

y = df["AT0090000000000000000X312X009800E"]
# np.random.seed(0)
# X = np.arange(1, 101).reshape(-1, 1)  # Zeitpunkte
# y = 2 * X.squeeze() + np.random.normal(0, 5, size=X.shape[0])  # Verkaufszahlen mit Rauschen

df['hour'] = df['timestamp'].dt.hour  # Stunde des Tages
df['weekday'] = df['timestamp'].dt.weekday  # Wochentag (Montag=0, Sonntag=6)
df['month'] = df['timestamp'].dt.month  # Monat

X = df[['hour', 'weekday', 'month']]  # Features

# Aufteilung der Daten in Trainingsdaten und Testdaten
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Modellinitialisierung und Training
model = LinearRegression()
model.fit(X_train, y_train)

# Vorhersage für zukünftige Zeitpunkte
# future_X = pd.date_range("2024-01-01", periods=96, freq="15min")
# future_X = np.arange(101, 121).reshape(-1, 1)  # Zukünftige Zeitpunkte
y_pred = model.predict(X_test)




# Berechnung der Metriken
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = calculate_mape(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("MSE:", mse)
print("MAE:", mae)
print("MAPE:", mape)
print("RMSE: ", rmse)

# Erstellen Sie Features für zukünftige Zeitpunkte
future_timestamps = pd.date_range(start='2024-01-01 00:00:00', end='2024-01-01 23:45:00', freq='15min')  # Beispiel: Vorhersage für 10 Stunden
future_features = pd.DataFrame({
    'timestamp': future_timestamps,
    'hour': future_timestamps.hour,
    'weekday': future_timestamps.weekday,
    'month': future_timestamps.month
})

# Vorhersagen für zukünftige Zeitpunkte treffen
future_predictions = model.predict(future_features[['hour', 'weekday', 'month']])

print("Future Predictions:")


# Visualisierung der Ergebnisse
plt.figure(figsize=(10, 6))
plt.scatter(future_timestamps, future_predictions, marker='o', color='b', linestyle='-', label='Predictions')
plt.title('Energy Consumption Predictions for Future Time Period')
plt.xlabel('Timestamp')
plt.ylabel('Energy Consumption')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
