import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model, save_model
import joblib
import os
from src.config import *

class ZoneClassifier:
    def __init__(self, model_type='knn'):
        self.model_type = model_type
        if model_type == 'knn':
            self.model = KNeighborsClassifier(n_neighbors=5)
        else:
            self.model = LogisticRegression()
        self.scaler = StandardScaler()

    def train(self, data):
        # Features: distance_to_imbl (could also use lat/lon but distance is more direct)
        # Using Lat/Lon is better for "geofencing" logic if boundary is complex.
        # But for this simulation, distance or Lat/Lon works. Let's use Lat/Lon for "realism"
        X = data[['lat', 'lon']]
        y = data['zone']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model.fit(X_train_scaled, y_train)
        print(f"Zone Classifier ({self.model_type}) Accuracy: {self.model.score(X_test_scaled, y_test)}")

    def predict(self, lat, lon):
        X_new = pd.DataFrame([[lat, lon]], columns=['lat', 'lon'])
        X_scaled = self.scaler.transform(X_new)
        return self.model.predict(X_scaled)[0]
    
    def save(self, path='models/zone_model.pkl'):
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)
        
    def load(self, path='models/zone_model.pkl'):
        loaded = joblib.load(path)
        self.model = loaded['model']
        self.scaler = loaded['scaler']

class TrajectoryForecaster:
    def __init__(self, lookback=LSTM_LOOKBACK):
        self.lookback = lookback
        self.model = None
        self.scaler = MinMaxScaler()
        
    def create_sequences(self, data, lookback):
        X, y = [], []
        # Calculate deltas for the entire sequence first
        deltas = np.diff(data, axis=0)
        # deltas length is len(data) - 1
        # we need 'lookback' deltas to predict the next delta
        for i in range(len(deltas) - lookback):
            X.append(deltas[i:(i + lookback)])
            y.append(deltas[i + lookback]) 
        return np.array(X), np.array(y)

    def train(self, df):
        sequences_X = []
        sequences_y = []
        
        # Fit scaler on DELTAS first
        coords = df[['lat', 'lon']].values
        
        # We need to compute all deltas across trips to fit the scaler properly
        all_deltas = []
        for trip_id, group in df.groupby('trip_id'):
            trip_data = group[['lat', 'lon']].values
            if len(trip_data) > 1:
                all_deltas.append(np.diff(trip_data, axis=0))
        
        if all_deltas:
            all_deltas_concat = np.vstack(all_deltas)
            self.scaler.fit(all_deltas_concat)
        
        for trip_id, group in df.groupby('trip_id'):
            trip_data = group[['lat', 'lon']].values
            if len(trip_data) > self.lookback + 1:
                # Get deltas for this trip
                trip_deltas = np.diff(trip_data, axis=0)
                # Scale deltas
                trip_deltas_scaled = self.scaler.transform(trip_deltas)
                
                # We can reuse create_sequences on the scaled deltas
                # but change create_sequences to not take diff again
                # Actually, let's just build X, y here:
                for i in range(len(trip_deltas_scaled) - self.lookback):
                    sequences_X.append(trip_deltas_scaled[i:(i + self.lookback)])
                    sequences_y.append(trip_deltas_scaled[i + self.lookback])
                
        X_train = np.array(sequences_X)
        y_train = np.array(sequences_y)
        
        # Build LSTM
        self.model = Sequential()
        self.model.add(LSTM(50, activation='relu', input_shape=(self.lookback, 2)))
        self.model.add(Dense(2)) # Lat delta, Lon delta
        self.model.compile(optimizer='adam', loss='mse')
        
        print("Training LSTM with Deltas...")
        self.model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
        
    def predict_next(self, recent_path):
        """
        recent_path: list of last 'lookback' + 1 (lat, lon) absolute positions
        so we can compute 'lookback' deltas.
        """
        if len(recent_path) < self.lookback + 1:
            # Fallback if we don't have enough points: just repeat the last known delta
            if len(recent_path) >= 2:
                last_delta = np.array(recent_path[-1]) - np.array(recent_path[-2])
                return (np.array(recent_path[-1]) + last_delta).tolist()
            return recent_path[-1] # Can't do much
            
        recent_points = np.array(recent_path[-(self.lookback + 1):])
        recent_deltas = np.diff(recent_points, axis=0)
        
        input_scaled = self.scaler.transform(recent_deltas).reshape(1, self.lookback, 2)
        
        # Predict next delta in scaled space
        pred_delta_scaled = self.model.predict(input_scaled, verbose=0)[0]
        
        # Inverse transform delta
        pred_delta = self.scaler.inverse_transform([pred_delta_scaled])[0]
        
        # Add delta to the absolute last point
        last_point = recent_points[-1]
        next_point = last_point + pred_delta
        
        return next_point # [lat, lon]
        
    def save(self, path='models/lstm_model.keras'):
        self.model.save(path)
        joblib.dump(self.scaler, 'models/lstm_scaler.pkl')
        
    def load(self, path='models/lstm_model.keras'):
        self.model = load_model(path)
        self.scaler = joblib.load('models/lstm_scaler.pkl')
