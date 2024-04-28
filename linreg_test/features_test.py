
"""
# Konvertierung der gesamten Spalte in datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Extrahieren von Merkmalen aus dem Zeitstempel
data['hour'] = data['timestamp'].dt.hour  # Stunde des Tages
data['weekday'] = data['timestamp'].dt.weekday  # Wochentag (Montag=0, Sonntag=6)
data['month'] = data['timestamp'].dt.month  # Monat
"""
