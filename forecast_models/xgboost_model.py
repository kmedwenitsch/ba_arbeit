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
    if col == "timestamp":
        continue
    data[col] = data[col].fillna(-999)

data.to_csv('data.csv')

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

scaler = MinMaxScaler(feature_range=(0, 1))
train_data_scaled = scaler.fit_transform(train_data.drop(['timestamp', total_energy_column], axis=1))
test_data_scaled = scaler.transform(test_data.drop(['timestamp', total_energy_column], axis=1))

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)].flatten())
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 96
X_train, y_train = create_dataset(train_data_scaled, train_data[total_energy_column], time_steps)
X_test, y_test = create_dataset(test_data_scaled, test_data[total_energy_column], time_steps)

# Anpassung der Hyperparameter
param_grid = {
    'n_estimators': [150],
    'learning_rate': [0.05],
    'max_depth': [5],
    'subsample': [1.0],
    'colsample_bytree': [1.0],
    'gamma': [0.1]
}

# XGBoost-Modell mit angepassten Hyperparametern
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1)  # Verwende alle verfügbaren Kerne

# Grid Search mit reduziertem Suchraum für schnelleres Training
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Prognose der Gesamterzeugungsleistung
total_energy_pred = best_model.predict(X_test)

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
