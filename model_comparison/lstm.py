import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# CSV-Datei einlesen
df = pd.read_csv('../input_data/Neudörfl_Production_full_AT0090000000000000000X312X009800E.csv', parse_dates=['timestamp'])
# df_train = pd.read_csv('../input_data/Neudörfl_Production_Training_AT0090000000000000000X312X009800E.csv', parse_dates=['timestamp'])
# df_test = pd.read_csv('../input_data/Neudörfl_Production_Test_AT0090000000000000000X312X009800E.csv', parse_dates=['timestamp'])

# Extrahieren von Merkmalen aus dem Zeitstempel
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month
df.set_index('timestamp', inplace=True)

# Normalisierung der Daten
scaler = MinMaxScaler()
df_normalized = scaler.fit_transform(df)


# Funktion zum Aufteilen der Daten in Trainings- und Testsets
def split_data(data, split_ratio=0.8):
    split_index = int(len(data) * split_ratio)
    train_data = data[:split_index]
    test_data = data[split_index:]
    return train_data, test_data

# Aufteilen der Daten in Trainings- und Testsets
train_data, test_data = split_data(df_normalized)


# Funktion zum Erstellen von Sequenzen für das LSTM-Modell
def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        sequence = data[i:i+sequence_length]
        sequences.append(sequence)
    return np.array(sequences)


# Hyperparameter für das LSTM-Modell
sequence_length = 24  # Länge der Eingabesequenz
num_features = 4  # Anzahl der Merkmale in den Daten (Energieproduktion, Stunde, Wochentag, Monat)
hidden_units = 50  # Anzahl der versteckten Einheiten im LSTM
batch_size = 64  # Batch-Größe für das Training
epochs = 10  # Anzahl der Epochen für das Training

# Erstellen von Sequenzen für das LSTM-Modell
X_train = create_sequences(train_data, sequence_length)
y_train = train_data[sequence_length:]
X_test = create_sequences(test_data, sequence_length)
y_test = test_data[sequence_length:]

# Modellinitialisierung und Training
model = Sequential()
model.add(LSTM(hidden_units, input_shape=(sequence_length, num_features)))
model.add(Dense(num_features))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

# Vorhersagen für den Testdatensatz
predictions = model.predict(X_test)

# Umkehrung der Normalisierung für die Vorhersagen
predictions_denormalized = scaler.inverse_transform(predictions)


mse = mean_squared_error(test_data[sequence_length:], predictions_denormalized)
mae = mean_absolute_error(test_data[sequence_length:], predictions_denormalized)
mape = np.mean(np.abs((test_data[sequence_length:] - predictions_denormalized) / test_data[sequence_length:])) * 100
rmse = np.sqrt(mse)


print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("Mean Absolute Percentage Error (MAPE):", mape)
print("Root Mean Squared Error (RMSE):", rmse)

# Plot der Vorhersagen
plt.figure(figsize=(12, 6))
plt.plot(df.index[sequence_length:], df['energy_consumption'][sequence_length:], color='blue', label='Historical Data')
plt.plot(df.index[-len(predictions_denormalized):], predictions_denormalized[:, 0], color='green', linestyle='--', label='Predicted Values')
plt.title('Energy Consumption Prediction (LSTM)')
plt.xlabel('Timestamp')
plt.ylabel('Energy Consumption')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

"""
# Grafische Darstellung der Inputdaten
plt.figure(figsize=(12, 6))
plt.plot(future_timestamps, predictions_denormalized, color='green', linestyle='--', label='Predicted Values')
plt.title('Energy Production Prediction for 30.12.2023')
plt.xlabel('Timestamp')
plt.ylabel('Energy Production')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
"""

