import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import holidays
import matplotlib.pyplot as plt

# Lade den Datensatz
data = pd.read_csv("../input_data/Neudörfl_Production_bis_21042024_gesamt.csv")

# Iteriere über die Spalten der Eingabedaten
for col in data.columns:
    # Überspringe die erste Spalte "timestamp"
    if col == "timestamp":
        continue

    # Setze NaN-Werte in der aktuellen Spalte auf -999
    data[col] = data[col].fillna(-999)

# Extrahiere Features aus dem Zeitstempel
data['timestamp'] = pd.to_datetime(data['timestamp'], format='%d.%m.%Y %H:%M:%S')
data['hour'] = data['timestamp'].dt.hour
data['weekday'] = data['timestamp'].dt.weekday
data['month'] = data['timestamp'].dt.month

# Liste österreichischer Feiertage für die Jahre 2023 und 2024
at_holidays = holidays.Austria(years=[2023, 2024])

# Erstelle Feature für Feiertage
data['holiday'] = data['timestamp'].apply(lambda x: int(x in at_holidays))

# Spalte für die gesamte Energieproduktion
total_energy_column = 'Total_Energy'

# Spalten für individuelle Energieproduktion
individual_energy_columns = list(data.columns[1:-4])  # Alle außer der ersten (Zeitstempel) und letzten 4 Spalten

# Aggregiere die Einzelzeitreihen zur Gesamterzeugungsleistung abzgl. der None Werte
data[total_energy_column] = data[individual_energy_columns].apply(lambda x: x[x != -999].sum(), axis=1)

# Splitte Daten in Trainings- und Testdaten
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Spaltenauswahl ohne 'timestamp' und 'Total_Energy'
feature_columns = ['hour', 'weekday', 'month', 'holiday', total_energy_column]

# Skaliere die Daten
feature_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler = MinMaxScaler(feature_range=(0, 1))
train_X_data_scaled = feature_scaler.fit_transform(train_data[feature_columns])
train_y_data_scaled = target_scaler.fit_transform(train_data[[total_energy_column]])
test_X_data_scaled = feature_scaler.transform(test_data[feature_columns])
test_y_data_scaled = target_scaler.transform(test_data[[total_energy_column]])

def create_dataset(X, y, time_steps=1):
    """
    Diese Funktion erstellt Trainings- und Testdaten für das LSTM-Modell.

    Parameters:
        X (array): Die Features.
        y (DataFrame): Die Zielvariablen.
        time_steps (int): Die Anzahl der vergangenen Zeitpunkte, die als Features verwendet werden sollen.

    Returns:
        Xs (array): Das Array der Features.
        ys (array): Das Array der Zielvariablen.
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps, -1])  # Die Zielvariable ist die Gesamterzeugungsleistung (letzte Spalte)
    return np.array(Xs), np.array(ys)

# Parameter für das LSTM-Modell
time_steps = 96  # Anzahl der vergangenen Zeitschritte, die als Features berücksichtigt werden sollen für die Prognosen
n_features = len(feature_columns)  # Anzahl der Features

# Trainingsdaten
X_train, y_train = create_dataset(train_X_data_scaled, train_y_data_scaled, time_steps)

# Testdaten
X_test, y_test = create_dataset(test_X_data_scaled, test_y_data_scaled, time_steps)

print(f"Length of y_test: {len(X_train)}")
print(f"Length of y_train: {len(y_train)}")
print(f"Length of X_test: {len(X_test)}")
print(f"Length of y_test: {len(y_test)}")

# Funktion zum Trainieren des LSTM-Modells
def train_lstm_model(X_train, y_train):
    model = Sequential()
    model.add(LSTM(units=100, activation='relu', return_sequences=True, input_shape=(time_steps, n_features)))
    model.add(LSTM(units=100, activation='relu'))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    return model

# Trainiere das LSTM-Modell
model = train_lstm_model(X_train, y_train)

print(f"Length of X_train: {len(y_train)}")

# Prognose der Gesamterzeugungsleistung
total_energy_pred_scaled = model.predict(X_test)
print(f"Length of total_energy_pred_scaled: {len(total_energy_pred_scaled)}")
total_energy_pred = target_scaler.inverse_transform(total_energy_pred_scaled).flatten()
print(f"Length of total_energy_pred: {len(total_energy_pred)}")

df = pd.DataFrame(total_energy_pred, y_test)
df.to_csv('output.csv')

# Berechne die Metriken
mse_total = mean_squared_error(y_test, total_energy_pred)
mae_total = mean_absolute_error(y_test, total_energy_pred)
rmse_total = np.sqrt(mse_total)

print("Prognose für die Gesamterzeugungsleistung:")
print("MSE:", mse_total)
print("MAE:", mae_total)
print("RMSE:", rmse_total)

# Plotte die Prognoseergebnisse
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Real')
plt.plot(total_energy_pred, label='Total Energy Forecast')
plt.xlabel('Time Steps')
plt.ylabel('Energy Production (kWh)')
plt.legend()
plt.title('Total Energy Forecast vs. Real')
plt.show()

# Scatterplot der Vorhersagen gegen die tatsächlichen Werte
plt.figure(figsize=(10, 6))
plt.scatter(y_test, total_energy_pred, alpha=0.3)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('Actual Total Energy Production (kWh)')
plt.ylabel('Predicted Total Energy Production (kWh)')
plt.title('Scatter Plot of Actual vs Predicted Total Energy Production')
plt.show()
