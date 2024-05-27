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
# Spalten für individuelle Energieproduktion
individual_energy_columns = list(data.columns[1:-5])  # Alle außer der ersten (Zeitstempel) und letzten 5 Spalten

# Behalte nur die relevanten Features, einschließlich individueller Energieproduktionsspalten
feature_columns = ['hour', 'weekday', 'month', 'holiday'] + individual_energy_columns

# Splitte Daten in Trainings- und Testdaten
train_data, test_data = train_test_split(data[feature_columns], test_size=0.2, random_state=42)

# Skaliere die Daten
scaler = MinMaxScaler(feature_range=(0, 1))
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

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
        ys.append(y[i + time_steps, -len(individual_energy_columns):])  # Die Zielvariable ist die Gesamterzeugungsleistung (letzte Spalte)
    return np.array(Xs), np.array(ys)

# Parameter für das LSTM-Modell
time_steps = 96  # Anzahl der vergangenen Zeitschritte, die als Features berücksichtigt werden sollen für die Prognosen
n_features = len(feature_columns)  # Anzahl der Features

# Trainingsdaten
X_train, y_train = create_dataset(train_data_scaled, time_steps)

# Testdaten
X_test, y_test = create_dataset(test_data_scaled, time_steps)

print(f"Length of y_test: {len(X_train)}")
print(f"Length of y_train: {len(y_train)}")
print(f"Length of X_test: {len(X_test)}")
print(f"Length of y_test: {len(y_test)}")

# Funktion zum Trainieren des LSTM-Modells
def train_lstm_model(X_train, y_train):
    model = Sequential()
    model.add(LSTM(units=100, activation='relu', return_sequences=True, input_shape=(time_steps, n_features)))
    model.add(LSTM(units=100, activation='relu'))
    model.add(Dense(units=len(individual_energy_columns)))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
    return model

# Trainiere das LSTM-Modell
model = train_lstm_model(X_train, y_train)

print(f"Length of X_train: {len(y_train)}")

# Prognose der Gesamterzeugungsleistung
individual_energy_pred_scaled = model.predict(X_test)
print(f"Length of total_energy_pred_scaled: {len(individual_energy_pred_scaled)}")
individual_energy_pred = scaler.inverse_transform(individual_energy_pred_scaled).flatten()
print(f"Length of total_energy_pred: {len(individual_energy_pred)}")

# Exportiere die Prognosen und die echten Werte in eine CSV-Datei
df = pd.DataFrame(individual_energy_pred, columns=individual_energy_columns)
df['Actual'] = pd.DataFrame(y_test, columns=individual_energy_columns)
df.to_csv('output.csv', index=False)

metrics = {'MSE': [], 'MAE': [], 'RMSE': []}
for col in individual_energy_columns:
    mse = mean_squared_error(y_test[:, individual_energy_columns.index(col)], individual_energy_pred[:, individual_energy_columns.index(col)])
    mae = mean_absolute_error(y_test[:, individual_energy_columns.index(col)], individual_energy_pred[:, individual_energy_columns.index(col)])
    rmse = np.sqrt(mse)
    metrics['MSE'].append(mse)
    metrics['MAE'].append(mae)
    metrics['RMSE'].append(rmse)
    print(f"Prognose für {col}:")
    print("MSE:", mse)
    print("MAE:", mae)
    print("RMSE:", rmse)

# Berechne den Durchschnitt der Metriken über alle individuellen Energieproduktionsspalten
avg_mse = np.mean(metrics['MSE'])
avg_mae = np.mean(metrics['MAE'])
avg_rmse = np.mean(metrics['RMSE'])

print("\nDurchschnittliche Prognosemetriken für alle individuellen Energieproduktionsspalten:")
print("Durchschnittlicher MSE:", avg_mse)
print("Durchschnittlicher MAE:", avg_mae)
print("Durchschnittlicher RMSE:", avg_rmse)


# Plotte die Prognoseergebnisse für die ersten 5 individuellen Energieproduktionsspalten
plt.figure(figsize=(10, 6))
for i, col in enumerate(individual_energy_columns[:5]):
    plt.subplot(3, 2, i+1)
    plt.plot(y_test[:, i], label='Real')
    plt.plot(individual_energy_pred[:, i], label='Forecast')
    plt.xlabel('Time Steps')
    plt.ylabel(f'Energy Production ({col}) (kWh)')
    plt.legend()
    plt.title(f'{col} Forecast vs. Real')

plt.tight_layout()
plt.show()