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

# Spalte für die gesamte Energieproduktion
total_energy_column = 'Total_Energy'

# Spalten für individuelle Energieproduktion
individual_energy_columns = list(data.columns[1:-5])  # Alle außer der ersten (Zeitstempel) und letzten 5 Spalten

# Aggregiere die Einzelzeitreihen zur Gesamterzeugungsleistung
data[total_energy_column] = data[individual_energy_columns].sum(axis=1)

# Splitte Daten in Trainings- und Testdaten
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Skaliere die Daten
scaler = MinMaxScaler(feature_range=(0, 1))
# in training und test daten sind alle Spalten bis auf timestamp und summierte energie vorhanden
# also alle individuellen residuallastwerte und die features dazu
train_data_scaled = scaler.fit_transform(train_data.drop(['timestamp', total_energy_column], axis=1))
test_data_scaled = scaler.transform(test_data.drop(['timestamp', total_energy_column], axis=1))


# Funktion zum Erstellen von Trainings- und Testdaten
def create_dataset(X1, X2, X3, y, time_steps=1):
    """
    Diese Funktion erstellt Trainings- und Testdaten für das LSTM-Modell.

    Parameters:
        X1 (array): Die historischen Features (individuelle Energieproduktionsdaten).
        X2 (array): Die anderen Features (Stunde, Wochentag, Feiertag, Monat).
        X3 (array): Der Zeitstempel.
        y (DataFrame): Die Zielvariablen.
        time_steps (int): Die Anzahl der vergangenen Zeitpunkte, die als Features verwendet werden sollen.

    Returns:
        Xs (array): Das Array der Features.
        ys (array): Das Array der Zielvariablen.
    """
    Xs, ys = [], []
    for i in range(len(X1) - time_steps):
        # Extrahiere die letzten 'time_steps' historischen Features
        v1 = X1[i:(i + time_steps)]
        # Extrahiere die anderen Features
        v2 = X2[i + time_steps]
        # Extrahiere den Zeitstempel
        v3 = X3[i + time_steps]
        # Kombiniere historische, andere Features und Zeitstempel
        v = np.concatenate((v1, v2, v3.reshape(-1, 1)), axis=1)
        Xs.append(v)
        # Der nächste Zeitschritt wird als Zielvariable verwendet
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)


# Parameter für das LSTM-Modell
time_steps = 96  # Anzahl der vergangenen Zeitschritte, die als Features berücksichtigt werden sollen für die Prognosen
n_features_hist = len(individual_energy_columns)  # Anzahl der historischen Features pro Zeitschritt (29 Einzelzeitreihen)
n_features_other = 4  # Anzahl der anderen Features (Stunde, Wochentag, Feiertag, Monat)
n_features_timestamp = 1  # Anzahl der Features für den Zeitstempel
epochs = 50
batch_size = 32


# Funktion zum Trainieren des LSTM-Modells
def train_lstm_model(X_train, y_train):
    """
    Diese Funktion trainiert das LSTM-Modell.

    Parameters:
        X_train (array): Die Trainingsdaten.
        y_train (array): Die Zielvariablen der Trainingsdaten.

    Returns:
        model (Sequential): Das trainierte LSTM-Modell.
    """
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', input_shape=(time_steps, n_features)))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return model


# Trainingsdaten
X_train_hist, X_train_other, X_train_timestamp, y_train = train_data_scaled[:, :n_features_hist], train_data_scaled[:, n_features_hist:-1], train_data_scaled[:, -1:], train_data[total_energy_column]
X_train, y_train = create_dataset(X_train_hist, X_train_other, X_train_timestamp, y_train, time_steps)


# Testdaten
X_test_hist, X_test_other, X_test_timestamp, y_test = test_data_scaled[:, :n_features_hist], test_data_scaled[:, n_features_hist:-1], test_data_scaled[:, -1:], test_data[total_energy_column]
X_test, y_test = create_dataset(X_test_hist, X_test_other, X_test_timestamp, y_test, time_steps)

# Trainiere das LSTM-Modell
model = train_lstm_model(X_train, y_train)

# Prognose der Gesamterzeugungsleistung
total_energy_pred = model.predict(X_test)

# Auswertung der Prognose
mse_total, mae_total, rmse_total = mean_squared_error(y_test, total_energy_pred),\
                                   mean_absolute_error(y_test, total_energy_pred),\
                                   np.sqrt(mean_squared_error(y_test, total_energy_pred))
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
