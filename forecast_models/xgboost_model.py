import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import holidays
import matplotlib.pyplot as plt

# Lade den Datensatz
data = pd.read_csv("../input_data/Neudörfl_Production_bis_21042024_gesamt.csv")

# Iteriere über die Spalten der Eingabedaten
for col in data.columns:
    # Überspringe die erste Spalte "timestamp"
    if col == "timestamp":
        continue
    # Entferne NaN-Werte in der aktuellen Spalte, während die Zeitstempel beibehalten werden
    data[col] = data[col].fillna(-999)

# Dataframe ausgeben in csv Datei
data.to_csv('data.csv')

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

# Aggregiere die Einzelzeitreihen zur Gesamterzeugungsleistung abzgl. der None Werte
data[total_energy_column] = data[individual_energy_columns].apply(lambda x: x[x != -999].sum(), axis=1)
print(data[total_energy_column].to_string())

# Splitte Daten in Trainings- und Testdaten
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Skaliere die Daten
scaler = MinMaxScaler(feature_range=(0, 1))
# in training und test daten sind alle Spalten bis auf timestamp und summierte energie vorhanden
# also alle individuellen residuallastwerte und die features dazu
train_data_scaled = scaler.fit_transform(train_data.drop(['timestamp', total_energy_column], axis=1))
test_data_scaled = scaler.transform(test_data.drop(['timestamp', total_energy_column], axis=1))

# Funktion zur Erstellung des Datensatzes für XGBoost
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)].flatten())
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

# Parameter für das Modell
time_steps = 96  # Anzahl der vergangenen Zeitschritte, die als Features berücksichtigt werden sollen für die Prognosen

# Trainingsdaten
X_train, y_train = create_dataset(train_data_scaled, train_data[total_energy_column], time_steps)

# Testdaten
X_test, y_test = create_dataset(test_data_scaled, test_data[total_energy_column], time_steps)

# Funktion zum Trainieren des XGBoost-Modells
def train_xgboost_model(X_train, y_train):
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(X_train, y_train)
    return model

# Trainiere das XGBoost-Modell
model = train_xgboost_model(X_train, y_train)

# Prognose der Gesamterzeugungsleistung
total_energy_pred = model.predict(X_test)

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
