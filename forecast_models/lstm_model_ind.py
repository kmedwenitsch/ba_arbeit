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
individual_energy_columns = list(data.columns[1:-4])  # Alle außer der ersten (Zeitstempel) und letzten 5 Spalten

# Splitte Daten in Trainings- und Testdaten
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)


def create_dataset(X, y, time_steps=1):
    """
    Diese Funktion erstellt Trainings- und Testdaten für das LSTM-Modell.

    Parameters:
        X (array): Die Features.
        y (array): Die Zielvariablen.
        time_steps (int): Die Anzahl der vergangenen Zeitpunkte, die als Features verwendet werden sollen.

    Returns:
        Xs (array): Das Array der Features.
        ys (array): Das Array der Zielvariablen.
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])  # Die Zielvariable ist die aktuelle individuelle Spalte
    return np.array(Xs), np.array(ys)


# Parameter für das LSTM-Modell
time_steps = 96  # Anzahl der vergangenen Zeitschritte, die als Features berücksichtigt werden sollen für die Prognosen

# Trainiere individuelle LSTM-Modelle für jede Spalte in individual_energy_columns
models = {}
mse_list = []
mae_list = []
rmse_list = []

for column in individual_energy_columns:
    try:
        print(f"Training model for column: {column}")
        # Trainingsdaten für die aktuelle Spalte
        train_data_column = train_data.dropna(subset=[column])
        test_data_column = test_data.dropna(subset=[column])

        # Skaliere die Features für die aktuelle Spalte
        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaler = MinMaxScaler(feature_range=(0, 1))

        train_X_data_scaled = feature_scaler.fit_transform(
            train_data_column[['hour', 'weekday', 'month', 'holiday'] + [column]])
        train_y_data_scaled = target_scaler.fit_transform(train_data_column[[column]])

        test_X_data_scaled = feature_scaler.transform(
            test_data_column[['hour', 'weekday', 'month', 'holiday'] + [column]])
        test_y_data_scaled = target_scaler.transform(test_data_column[[column]])

        # Trainingsdaten und Zielvariablen
        y_train_column_scaled = train_y_data_scaled
        y_test_column_scaled = test_y_data_scaled
        X_train_column, y_train_column = create_dataset(train_X_data_scaled, y_train_column_scaled, time_steps)
        X_test_column, y_test_column = create_dataset(test_X_data_scaled, y_test_column_scaled, time_steps)

        # Erstelle und trainiere das LSTM-Modell für die aktuelle Spalte
        model = Sequential()
        model.add(LSTM(units=100, activation='relu', return_sequences=True,
                       input_shape=(time_steps, train_X_data_scaled.shape[1])))
        model.add(LSTM(units=100, activation='relu'))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train_column, y_train_column, epochs=10, batch_size=32, verbose=0)

        # Prognose für die aktuelle Spalte
        column_pred_scaled = model.predict(X_test_column)
        column_pred = target_scaler.inverse_transform(column_pred_scaled).flatten()

        # Rückskalieren der Zielvariablen
        y_test_column = target_scaler.inverse_transform(y_test_column.reshape(-1, 1)).flatten()

        # Überprüfen auf NaN-Werte vor der Berechnung der Metriken
        if np.isnan(column_pred).any() or np.isnan(y_test_column).any():
            print(f"Berechnung der Metriken für Spalte {column} nicht möglich, enthält NaN-Werte.")
            continue

        # Berechne die Metriken für die aktuelle Spalte
        mse_column = mean_squared_error(y_test_column, column_pred)
        mae_column = mean_absolute_error(y_test_column, column_pred)
        rmse_column = np.sqrt(mse_column)

        mse_list.append(mse_column)
        mae_list.append(mae_column)
        rmse_list.append(rmse_column)

        models[column] = model

    except Exception as e:
        print(f"Berechnung der Metriken für Spalte {column} nicht möglich: {e}")
        continue

# Berechne den Durchschnitt der Metriken über alle Modelle
avg_mse = np.mean(mse_list)
avg_mae = np.mean(mae_list)
avg_rmse = np.mean(rmse_list)

print("Average Metrics:")
print("Average MSE:", avg_mse)
print("Average MAE:", avg_mae)
print("Average RMSE:", avg_rmse)
