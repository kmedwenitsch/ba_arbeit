import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from datetime import datetime
import holidays
import matplotlib.pyplot as plt

# Lade den Datensatz
data = pd.read_csv("Neudörfl_Production_bis_21042024_gesamt.csv")

# Extrahiere Features aus dem Zeitstempel
data['timestamp'] = pd.to_datetime(data['timestamp'], format='%d.%m.%Y %H:%M:%S')
data['hour'] = data['timestamp'].dt.hour
data['weekday'] = data['timestamp'].dt.weekday
data['month'] = data['timestamp'].dt.month

# Liste österreichischer Feiertage für die Jahre 2023 und 2024
at_holidays = holidays.Austria(years=[2023, 2024])

# Erstelle Feature für Feiertage
data['holiday'] = data['timestamp'].apply(lambda x: int(x in at_holidays))


# Funktion zum Erstellen von Trainings- und Testdaten
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)


# Parameter für das LSTM-Modell
time_steps = 96  # Anzahl der Zeitschritte pro Tag (96 15-Minuten-Schritte)
n_features = 1  # Anzahl der Features pro Zeitschritt
epochs = 50
batch_size = 32


# Funktion zum Trainieren des LSTM-Modells
def train_lstm_model(X_train, y_train):
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', input_shape=(time_steps, n_features)))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return model


# Funktion zur Vorhersage der Energieproduktion
def forecast(model, X_test):
    return model.predict(X_test)


# Funktion zur Auswertung der Prognoseergebnisse
def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mse, mae, rmse


# Splitte Daten in Trainings- und Testdaten
train_data = data[data['timestamp'] < '2024-04-22']
test_data = data[data['timestamp'] >= '2024-04-22']

X_train = train_data.drop(['timestamp'], axis=1)
X_test = test_data.drop(['timestamp'], axis=1)

y_train = train_data['Total_Energy']
y_test = test_data['Total_Energy']

# Skaliere die Daten
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Erstelle Trainings- und Testdatensätze für das LSTM-Modell
X_train_final, y_train_final = create_dataset(X_train_scaled, y_train, time_steps)
X_test_final, y_test_final = create_dataset(X_test_scaled, y_test, time_steps)

# Trainiere das LSTM-Modell
model = train_lstm_model(X_train_final, y_train_final)

# Prognose der Gesamterzeugungsleistung
total_energy_pred = forecast(model, X_test_final)

# Auswertung der Prognose
mse_total, mae_total, rmse_total = evaluate(y_test_final, total_energy_pred)
print("Prognose für die Gesamterzeugungsleistung:")
print("MSE:", mse_total)
print("MAE:", mae_total)
print("RMSE:", rmse_total)

# Speichere Prognoseergebnisse in CSV-Datei
test_data['Total_Energy_Pred'] = np.ravel(total_energy_pred)
test_data.to_csv('total_energy_forecast.csv', index=False)

# Prognose für einzelne Energieproduktionsdaten
individual_preds = []
for column in range(1, 30):
    y_train = train_data.iloc[:, column]
    y_test = test_data.iloc[:, column]

    y_train_final = np.ravel(y_train[time_steps:])
    y_test_final = np.ravel(y_test[time_steps:])

    model = train_lstm_model(X_train_final, y_train_final)
    pred = forecast(model, X_test_final)
    individual_preds.append(pred)

    mse, mae, rmse = evaluate(y_test_final, pred)
    print(f"Prognose für Spalte {column}:")
    print("MSE:", mse)
    print("MAE:", mae)
    print("RMSE:", rmse)


# Speichere Prognoseergebnisse für einzelne Energieproduktionsdaten in CSV-Datei
individual_preds = np.array(individual_preds).T
for i, column in enumerate(range(1, 30)):
    test_data[f'Energy_{column}_Pred'] = individual_preds[:, i]
test_data.to_csv('individual_energy_forecasts.csv', index=False)

# Plotte die Prognoseergebnisse
plt.figure(figsize=(10, 6))
plt.plot(y_test_final, label='Real')
plt.plot(total_energy_pred, label='Total Energy Forecast')
plt.xlabel('Time Steps')
plt.ylabel('Energy Production (kWh)')
plt.legend()
plt.title('Total Energy Forecast vs. Real')
plt.show()
