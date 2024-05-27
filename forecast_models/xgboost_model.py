import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
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

    # Setze NaN-Werte in der aktuellen Spalte auf -999
    data[col] = data[col].fillna(-999)

# Extrahiere Features aus dem Zeitstempel
data['timestamp'] = pd.to_datetime(data['timestamp'], format='%d.%m.%Y %H:%M:%S')
data['hour'] = data['timestamp'].dt.hour
data['weekday'] = data['timestamp'].dt.weekday
data['month'] = data['timestamp'].dt.month

at_holidays = holidays.Austria(years=[2023, 2024])
data['holiday'] = data['timestamp'].apply(lambda x: int(x in at_holidays))

total_energy_column = 'Total_Energy'
individual_energy_columns = list(data.columns[1:-5])

data[total_energy_column] = data[individual_energy_columns].apply(lambda x: x[x != -999].sum(), axis=1)

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
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps, -1])  # Die Zielvariable ist die Gesamterzeugungsleistung (letzte Spalte)
    return np.array(Xs), np.array(ys)

time_steps = 96
n_features = len(feature_columns)  # Anzahl der Features

X_train, y_train = create_dataset(train_X_data_scaled, train_y_data_scaled, time_steps)
X_test, y_test = create_dataset(test_X_data_scaled, test_y_data_scaled, time_steps)

# Umformen der Daten
n_samples, time_steps, n_features = X_train.shape
X_train = X_train.reshape((n_samples, time_steps * n_features))

n_samples, time_steps, n_features = X_test.shape
X_test = X_test.reshape((n_samples, time_steps * n_features))

# Anpassung der Hyperparameter
param_grid = {
    'n_estimators': [50],
    'learning_rate': [0.05],
    'max_depth': [3],
    'subsample': [1.0],
    'colsample_bytree': [1.0],
    'gamma': [0.05]
}

# XGBoost-Modell mit angepassten Hyperparametern
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1)  # Verwende alle verfügbaren Kerne
# xgb_model.fit(X_train, y_train)

# Grid Search mit reduziertem Suchraum für schnelleres Training
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Prognose der Gesamterzeugungsleistung
total_energy_pred_scaled = best_model.predict(X_test)
total_energy_pred = target_scaler.inverse_transform(total_energy_pred_scaled.reshape(-1, 1)).flatten()

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

plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Real')
plt.plot(total_energy_pred, label='Total Energy Forecast')
plt.xlabel('Time Steps')
plt.ylabel('Energy Production (kWh)')
plt.legend()
plt.title('Total Energy Forecast vs. Real')
plt.show()
