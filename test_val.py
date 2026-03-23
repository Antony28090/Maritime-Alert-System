import pandas as pd
import numpy as np
from src.models import TrajectoryForecaster

df = pd.read_csv('data/vessel_data.csv')
trip_data = df[df['trip_id'] == df['trip_id'].unique()[0]]
coords = trip_data[['lat', 'lon']].values

lstm_model = TrajectoryForecaster()
lstm_model.load()

X_seq, y_next_true = lstm_model.create_sequences(coords, 5)

print("y_next_true first 3:", y_next_true[:3])

N, L, F = X_seq.shape
X_seq_flat = X_seq.reshape(N*L, F)
X_seq_scaled = lstm_model.scaler.transform(X_seq_flat).reshape(N, L, F)

print("X_seq_scaled first 1:", X_seq_scaled[0])

pred_scaled = lstm_model.model.predict(X_seq_scaled, verbose=0)
print("pred_scaled first 3:", pred_scaled[:3])

pred_unscaled = lstm_model.scaler.inverse_transform(pred_scaled)
print("pred_unscaled first 3:", pred_unscaled[:3])
