import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from workalendar.europe import Austria

# CSV-Datei einlesen
data = pd.read_csv('../input_data/Neudörfl_Production_full_AT0090000000000000000X312X009800E.csv', parse_dates=['timestamp'])

# Extrahieren von Merkmalen aus dem Zeitstempel
data['timestamp'] = pd.to_datetime(data['timestamp'])
data['hour'] = data['timestamp'].dt.hour
data['day_of_week'] = data['timestamp'].dt.dayofweek
data['month'] = data['timestamp'].dt.month

# Festlegen von Feiertagen
cal = Austria()
holidays = cal.holidays(2023)  # Annahme: Das Jahr 2023 für Feiertage in Österreich

# Hinzufügen einer Spalte für Feiertage
data['is_holiday'] = data['timestamp'].dt.date.astype('datetime64').isin(holidays).astype(int)

data.set_index('timestamp', inplace=True)

# Zufälliges Aufteilen der Daten in Trainings- und Testsets
X = data[['hour', 'day_of_week', 'month', 'is_holiday']].values
y = data['AT0090000000000000000X312X009800E'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Modellinitialisierung und Training
model = XGBRegressor()
model.fit(X_train, y_train)

# Vorhersagen für den Testdatensatz
predictions = model.predict(X_test)

# Berechnung der Metriken für die Vorhersagen
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mse)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)

# Plot der Vorhersagen
plt.figure(figsize=(12, 6))
plt.plot(y_test, color='blue', label='Actual')
plt.plot(predictions, color='green', linestyle='--', label='Predicted')
plt.title('Energy Production Prediction (XGBoost)')
plt.xlabel('Time')
plt.ylabel('Energy Production')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
