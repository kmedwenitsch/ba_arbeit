import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error

# CSV-Datei einlesen
data = pd.read_csv('../input_data/Neudörfl_Production_bis_21042024_AT0090000000000000000X312X009800E.csv', parse_dates=['timestamp'])

# Extrahieren von Merkmalen aus dem Zeitstempel
data['timestamp'] = pd.to_datetime(data['timestamp'])
data['hour'] = data['timestamp'].dt.hour
data['day_of_week'] = data['timestamp'].dt.dayofweek
data['month'] = data['timestamp'].dt.month

# Manuell erstellte Liste von Feiertagen für Österreich
holidays_2023 = [
    '2023-01-01',  # Neujahr
    '2023-01-06',  # Heilige Drei Könige
    '2023-04-14',  # Karfreitag
    '2023-04-17',  # Ostermontag
    '2023-05-01',  # Tag der Arbeit
    '2023-05-25',  # Christi Himmelfahrt
    '2023-06-04',  # Pfingstsonntag
    '2023-06-05',  # Pfingstmontag
    '2023-06-15',  # Fronleichnam
    '2023-10-26',  # Nationalfeiertag
    '2023-11-01',  # Allerheiligen
    '2023-12-08',  # Mariä Empfängnis
    '2023-12-25',  # Weihnachten
    '2023-12-26'   # Stephanitag
]

holidays_2024 = [
    '2024-01-01',  # Neujahr
    '2024-01-06',  # Heilige Drei Könige
    '2024-03-29',  # Karfreitag
    '2024-04-01',  # Ostermontag
    '2024-05-01',  # Tag der Arbeit
    '2024-05-09',  # Christi Himmelfahrt
    '2024-05-19',  # Pfingstsonntag
    '2024-05-20',  # Pfingstmontag
    '2024-06-06',  # Fronleichnam
    '2024-10-26',  # Nationalfeiertag
    '2024-11-01',  # Allerheiligen
    '2024-12-08',  # Mariä Empfängnis
    '2024-12-25',  # Weihnachten
    '2024-12-26'   # Stephanitag
]

# Hinzufügen einer Spalte für Feiertage
data['is_holiday'] = data['timestamp'].dt.date.astype('datetime64[ns]').isin(holidays_2023 + holidays_2024).astype(int)

data.set_index('timestamp', inplace=True)

# Zufälliges Aufteilen der Daten in Trainings- und Testsets
X = data[['hour', 'day_of_week', 'month', 'is_holiday']].values
y = data['AT0090000000000000000X312X009800E'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Modellinitialisierung und Training
model = ElasticNet()
model.fit(X_train, y_train)

# Vorhersagen für den Testdatensatz
predictions = model.predict(X_test)

# Clippen der Vorhersagen auf positive Werte
predictions = np.clip(predictions, a_min=0, a_max=None)
print(predictions)

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
plt.title('Energy Production Prediction')
plt.xlabel('Time')
plt.ylabel('Energy Production')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
