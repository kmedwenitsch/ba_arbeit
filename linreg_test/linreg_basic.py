import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

# Berechnung von MAPE (Mean Absolute Percentage Error)
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Beispielzeitreihe erstellen
df = pd.read_csv("./input_data/Neudörfl_Production_bis_12_2023_AT0090000000000000000X312X009800E.csv")
X = df["timestamp"]
y = df["AT0090000000000000000X312X009800E"]
# np.random.seed(0)
# X = np.arange(1, 101).reshape(-1, 1)  # Zeitpunkte
# y = 2 * X.squeeze() + np.random.normal(0, 5, size=X.shape[0])  # Verkaufszahlen mit Rauschen

# Modellinitialisierung und Training
model = LinearRegression()
model.fit(X, y)

# Vorhersage für zukünftige Zeitpunkte
future_X = pd.date_range("2024-01-01", periods=96, freq="15min")
# future_X = np.arange(101, 121).reshape(-1, 1)  # Zukünftige Zeitpunkte
predicted_y = model.predict(future_X)

"""
# Visualisierung der Ergebnisse
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Daten')
plt.plot(X, model.predict(X), color='red', label='Lineare Regression')
plt.plot(future_X, predicted_y, color='green', linestyle='--', label='Prognose für zukünftige Zeitpunkte')
plt.xlabel('Zeit')
plt.ylabel('Verkaufszahlen')
plt.title('Lineare Regression für Zeitreihenprognose')
plt.legend()
plt.grid(True)
plt.show()
"""


# Berechnung der Metriken
mse = mean_squared_error(y, model.predict(X))
mae = mean_absolute_error(y, model.predict(X))
mape = calculate_mape(y, model.predict(X))
rmse = np.sqrt(mean_squared_error(y, model.predict(X)))

print("MSE:", mse)
print("MAE:", mae)
print("MAPE:", mape)
print("RMSE: ", rmse)
